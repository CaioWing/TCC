"""Gera o conjunto YOLO do MBG-V2 em recortes NATIVOS 640x640, a partir dos PNG a 1 fps
e de splits/splits.json (ver mbg_splits.py).

Por que recortes nativos: a memoria geografica roda a rede so nos recortes 640x640 que cobrem
celulas ainda nao vistas, e a estrategia de quadro inteiro usa os mesmos recortes (6x4 = 24 no
3840x2160; 7x4 = 28 no 4096x2160 da regiao A). Treinar na mesma escala da inferencia evita
trocar o detector entre estrategias.

Regras:
  - grade regular com sobreposicao minima (linspace de 0 a W-640 e de 0 a H-640);
  - caixa mantida se >= 40% da area original cai dentro do recorte (cortada na borda);
  - train/val: todos os recortes com objeto + 10% dos vazios (sorteio deterministico por hash);
  - test/test_speed: todos os recortes (falsos alarmes medidos sem selecao);
  - classes: 0-5 principais (avaliadas), 6-11 raras (rotuladas, nao avaliadas).
Saida: <root>/yolo_640/{images,labels}/<split>/<video>_f<frame>_x<x>_y<y>.{jpg,txt},
       <root>/yolo_640/recortes.csv e <root>/yolo_640/mbg_v2_640.yaml
"""
import hashlib, json, os, re, threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image

T = 640
MIN_VIS = 0.4
EMPTY_KEEP = 0.10
JPEG_Q = 95
CLASSES = ['watertank', 'potted_plant', 'storm_drain', 'pool', 'tire', 'bucket',
           'plastic_bag', 'bottle', 'small_trash_bin', 'puddle', 'large_trash_bin', 'dumpster']
CID = {c: i for i, c in enumerate(CLASSES)}


def grid(W, H):
    xs = np.round(np.linspace(0, W - T, int(np.ceil(W / T)))).astype(int)
    ys = np.round(np.linspace(0, H - T, int(np.ceil(H / T)))).astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


def keep_empty(key):
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < EMPTY_KEEP


def parse_flight(fdir):
    pngs = sorted(os.listdir(os.path.join(fdir, 'frames')))
    idx = {int(re.findall(r'\d+', s)[0]): s for s in pngs}
    xf = os.path.join(fdir, 'ann', os.listdir(os.path.join(fdir, 'ann'))[0])
    root = ET.parse(xf).getroot()
    osz = root.find('meta/task/original_size')
    W, H = int(osz.find('width').text), int(osz.find('height').text)
    boxes = {f: [] for f in idx}
    for tr in root.iter('track'):
        lab = tr.get('label'); tid = int(tr.get('id'))
        for b in tr.iter('box'):
            fr = int(b.get('frame'))
            if b.get('outside') == '1' or fr not in boxes:
                continue
            boxes[fr].append((CID[lab], tid, float(b.get('xtl')), float(b.get('ytl')),
                              float(b.get('xbr')), float(b.get('ybr'))))
    return os.path.splitext(os.path.basename(xf))[0], W, H, idx, boxes


def process_frame(args):
    split, vid, W, H, png, fr, bxs, out = args
    tiles, rows = grid(W, H), []
    b = np.array([x[2:] for x in bxs]) if bxs else np.zeros((0, 4))
    cls = [x[0] for x in bxs]; tids = [x[1] for x in bxs]
    img = None
    for (x0, y0) in tiles:
        name = f'{vid}_f{fr:05d}_x{x0:04d}_y{y0:04d}'
        lines, ntid = [], []
        if len(b):
            cx1 = np.clip(b[:, 0], x0, x0 + T); cy1 = np.clip(b[:, 1], y0, y0 + T)
            cx2 = np.clip(b[:, 2], x0, x0 + T); cy2 = np.clip(b[:, 3], y0, y0 + T)
            vis = ((cx2 - cx1) * (cy2 - cy1)) / np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6)
            for k in np.where(vis >= MIN_VIS)[0]:
                xc = ((cx1[k] + cx2[k]) / 2 - x0) / T; yc = ((cy1[k] + cy2[k]) / 2 - y0) / T
                w = (cx2[k] - cx1[k]) / T; h = (cy2[k] - cy1[k]) / T
                lines.append(f'{cls[k]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}'); ntid.append(tids[k])
        if not lines and split in ('train', 'val') and not keep_empty(name):
            continue
        ip = os.path.join(out, 'images', split, name + '.jpg')
        lp = os.path.join(out, 'labels', split, name + '.txt')
        if not os.path.exists(ip):
            if img is None:
                try:
                    img = Image.open(png).convert('RGB')
                except OSError as e:          # 1 PNG corrompido conhecido: voo 01, frame_3552
                    return [dict(split=split, video=vid, frame=fr, x0=-1, y0=-1, n_boxes=-1,
                                 track_ids='', file=f'ILEGIVEL: {png} ({e})')]
            img.crop((x0, y0, x0 + T, y0 + T)).save(ip, quality=JPEG_Q)
        with open(lp, 'w') as fh:
            fh.write('\n'.join(lines) + ('\n' if lines else ''))
        rows.append(dict(split=split, video=vid, frame=fr, x0=x0, y0=y0, n_boxes=len(lines),
                         track_ids=' '.join(map(str, ntid)), file=os.path.relpath(ip, out)))
    return rows


def build(root=r'E:\TCC\data\mbg_v2', workers=5, log=print):
    raw = os.path.join(root, 'raw'); out = os.path.join(root, 'yolo_640')
    spec = json.load(open(os.path.join(root, 'splits', 'splits.json'), encoding='utf-8'))
    v2s = {v: s for s, vs in spec['voos'].items() for v in vs}
    for s in v2s.values():
        for k in ('images', 'labels'):
            os.makedirs(os.path.join(out, k, s), exist_ok=True)
    allrows = []
    for fd in sorted(d for d in os.listdir(raw) if d.startswith('flight-mbg-v2-')):
        fdir = os.path.join(raw, fd)
        vid, W, H, idx, boxes = parse_flight(fdir)
        s = v2s[vid]
        jobs = [(s, vid, W, H, os.path.join(fdir, 'frames', idx[f]), f, boxes[f], out) for f in sorted(idx)]
        with ThreadPoolExecutor(workers) as ex:
            for r in ex.map(process_frame, jobs):
                allrows.extend(r)
        log(f'{vid} {s}: {sum(1 for r in allrows if r["video"] == vid)} recortes')
    M = pd.DataFrame(allrows); M.to_csv(os.path.join(out, 'recortes.csv'), index=False)
    with open(os.path.join(out, 'mbg_v2_640.yaml'), 'w', encoding='utf-8') as fh:
        fh.write(f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                 f'# teste de velocidade (regiao A): images/test_speed\nnames:\n')
        for i, c in enumerate(CLASSES):
            fh.write(f'  {i}: {c}\n')
    return M


if __name__ == '__main__':
    import sys, time
    t0 = time.time()
    M = build(log=lambda s: print(f'{time.time() - t0:7.0f}s {s}', flush=True))
    ok = M[M.n_boxes >= 0]
    print(ok.groupby('split').agg(recortes=('file', 'size'), com_objeto=('n_boxes', lambda s: (s > 0).sum()),
                                  caixas=('n_boxes', 'sum')).to_string())
    print('ilegiveis:', M[M.n_boxes < 0].file.tolist())
