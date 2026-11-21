"""Divisao treino/validacao/teste do MBG-V2 por REGIAO, pensada para a memoria geografica.

Criterios (na ordem em que restringem a busca):
  1. Unidade = regiao. Os dois voos de uma regiao se sobrepoem no solo (1-11%) e cada ponto
     aparece em ~190-283 quadros; dividir por quadro ou por voo vaza o mesmo objeto fisico.
  2. Regiao A (2019, 7,9 m/s, 4096x2160) fica fora do treino: teste de velocidade.
  3. Validacao e teste so com telemetria continua (maior falha de log dentro do video < 1,5 s),
     porque a memoria decide os recortes quadro a quadro a partir da pose.
  4. M fora do teste (a fragmentacao dos focos foi diagnosticada nela no piloto).
  5. Minimos no teste: >=40 pneus, >=40 baldes, >=25 caixas abertas, >=80 piscinas, >=80 vasos;
     na validacao: >=20 pneus e >=20 baldes; o treino fica com >=45% de cada classe.
  6. Entre as combinacoes viaveis, minimiza a soma das distancias L1 entre a proporcao de
     classes de cada conjunto e a do total B-M, mais 2*|f_teste-0,25| + 2*|f_val-0,15|.

Uso:  python mbg_splits.py --root E:\\TCC\\data\\mbg_v2
Saida: <root>/splits/{splits.json, voos.csv, regioes.csv, telemetria_falhas.csv}
"""
import argparse, json, os, re
from itertools import combinations
import numpy as np
import pandas as pd

MAIN = ['watertank', 'potted_plant', 'storm_drain', 'pool', 'tire', 'bucket']
RARE = ['plastic_bag', 'bottle', 'small_trash_bin', 'puddle', 'large_trash_bin', 'dumpster']
MAX_GAP_S = 1.5


def telemetry_gaps(raw, flights):
    rows = []
    for fd in flights:
        p = os.path.join(raw, fd, 'log')
        L = pd.read_csv(os.path.join(p, os.listdir(p)[0]),
                        usecols=['time(millisecond)', 'datetime(utc)', 'remainPowerPercent'])
        tm = L['time(millisecond)'].to_numpy() / 1000.0
        dt = np.diff(tm)
        region = fd.split('region')[1]
        fps = 24000 / 1001 if region == 'A' else 30000 / 1001
        xml = os.path.join(raw, fd, 'ann', os.listdir(os.path.join(raw, fd, 'ann'))[0])
        with open(xml, 'rb') as fh:
            head = fh.read(3000).decode('utf8', 'replace')
        stop = int(re.search(r'<stop_frame>(\d+)</stop_frame>', head).group(1))
        vdur = (stop + 1) / fps
        inside = (tm[:-1] - tm[0]) < vdur           # so falhas dentro da duracao do video
        dts = pd.to_datetime(L['datetime(utc)'], errors='coerce').dropna()
        rows.append(dict(flight=fd, region=region, date=str(dts.iloc[0].date()) if len(dts) else '',
                         video_s=round(vdur, 1), log_s=round(tm[-1] - tm[0], 1),
                         max_gap_in_video_s=round(float(dt[inside].max()), 2),
                         n_gaps_gt_1s=int((dt[inside] > 1.0).sum()),
                         zero_battery_rows=int((L.remainPowerPercent == 0).sum()),
                         duplicated_t_rows=int((dt == 0).sum())))
    return pd.DataFrame(rows)


def region_table(tracks):
    ct = pd.crosstab(tracks.region, tracks.label)
    R = ct.reindex(columns=MAIN, fill_value=0).copy()
    R['raras'] = ct.drop(columns=[c for c in MAIN if c in ct], errors='ignore').sum(1)
    R['trilhas'] = ct.sum(1)
    R['aberta'] = (tracks[(tracks.label == 'watertank') & (tracks.a_is_open == 'yes')]
                   .groupby('region').size().reindex(R.index).fillna(0).astype(int))
    R['lado_lt_32px'] = tracks[tracks.side < 32].groupby('region').size().reindex(R.index).fillna(0).astype(int)
    return R


def search(R, clean):
    cols = MAIN + ['raras']
    BM = R.drop(index='A')
    regs = list(BM.index)
    tot = BM[cols].sum(); ptot = tot / tot.sum()
    out = []
    cand_te = [r for r in clean if r != 'M']
    for te in combinations(cand_te, 3):
        for va in combinations([r for r in clean if r not in te], 2):
            tr = [r for r in regs if r not in te + va]
            T, V, Tr = BM.loc[list(te)].sum(), BM.loc[list(va)].sum(), BM.loc[tr].sum()
            if (T.tire < 40 or T.bucket < 40 or T.aberta < 25 or T.pool < 80 or T.potted_plant < 80
                    or V.tire < 20 or V.bucket < 20 or ((Tr[cols] / tot) < 0.45).any()):
                continue
            d = sum(np.abs(BM.loc[x][cols].sum() / BM.loc[x][cols].sum().sum() - ptot).sum()
                    for x in (tr, list(te), list(va)))
            fte = T.trilhas / BM.trilhas.sum(); fva = V.trilhas / BM.trilhas.sum()
            out.append((float(d + 2 * abs(fte - 0.25) + 2 * abs(fva - 0.15)), te, va, tr))
    out.sort(key=lambda z: z[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'E:\TCC\data\mbg_v2')
    a = ap.parse_args()
    raw = os.path.join(a.root, 'raw')
    flights = sorted(d for d in os.listdir(raw) if d.startswith('flight-mbg-v2-'))
    tracks = pd.read_csv(os.path.join(a.root, 'objetos_por_trilha.csv'))
    gaps = telemetry_gaps(raw, flights)
    clean_reg = gaps.groupby('region').max_gap_in_video_s.max()
    clean = [r for r in clean_reg.index if r != 'A' and clean_reg[r] < MAX_GAP_S]
    R = region_table(tracks)
    ranking = search(R, clean)
    score, te, va, tr = ranking[0]
    split_of = {r: 'train' for r in tr}
    split_of.update({r: 'val' for r in va}); split_of.update({r: 'test' for r in te}); split_of['A'] = 'test_speed'

    od = os.path.join(a.root, 'splits'); os.makedirs(od, exist_ok=True)
    gaps['split'] = gaps.region.map(split_of)
    gaps['n_png_1fps'] = [len(os.listdir(os.path.join(raw, f, 'frames'))) for f in gaps.flight]
    gaps['video'] = [os.path.splitext(os.listdir(os.path.join(raw, f, 'ann'))[0])[0] for f in gaps.flight]
    gaps.to_csv(os.path.join(od, 'voos.csv'), index=False)
    R['split'] = R.index.map(split_of)
    R['max_gap_s'] = clean_reg.reindex(R.index)
    R.to_csv(os.path.join(od, 'regioes.csv'))
    pd.DataFrame([dict(rank=i + 1, score=round(s, 4), test=''.join(t_), val=''.join(v_))
                  for i, (s, t_, v_, _) in enumerate(ranking[:20])]).to_csv(os.path.join(od, 'ranking_divisoes.csv'), index=False)
    spec = dict(
        criterio='divisao por regiao; ver docstring de code/mbg_splits.py',
        max_gap_s_val_teste=MAX_GAP_S,
        regioes={k: sorted(r for r, s in split_of.items() if s == k) for k in ['train', 'val', 'test', 'test_speed']},
        voos={k: gaps[gaps.split == k].video.tolist() for k in ['train', 'val', 'test', 'test_speed']},
        uso={'train': 'treino do detector (quadros 1 fps, recortes 640 nativos)',
             'val': 'parada antecipada do detector; ajuste de n_v*, margem das celulas e fusao de fragmentos (video completo + telemetria)',
             'test': 'avaliacao final: mAP por recorte, taxa de deteccao de focos unicos, energia por km2 (video completo + telemetria)',
             'test_speed': 'outra campanha (2019, 7,9 m/s, ~80 vistas por ponto): estresse da memoria com pouca redundancia'},
        classes_yolo=MAIN + RARE, classes_avaliadas=MAIN,
        score=round(score, 4))
    with open(os.path.join(od, 'splits.json'), 'w', encoding='utf-8') as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    print(json.dumps(spec['regioes'], ensure_ascii=False))


if __name__ == '__main__':
    main()
