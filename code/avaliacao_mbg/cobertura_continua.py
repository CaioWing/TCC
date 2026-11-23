"""Recortes por cobertura e intervalo fixo desde o quadro zero, com cota pareada."""
import argparse
import json

import numpy as np
from filelock import FileLock

from adaptativa import MOTOR, VOOS, gravar_npz, verificar_pesos
from dados import CACHE, OUT, folder
from memoria_sim import CELL, T, load_pose, tiles_grid

DESTINO = OUT / "cobertura_continua"
TEMP = CACHE / "cobertura_continua_v1"
METODOS = ("cobertura", "intervalo")


def plano(pose, tamanho, gsd):
    """Células fixas da regra calibrada. Só a pose atual afeta a prioridade."""
    n = len(pose)
    largura, altura = tamanho
    tiles = np.asarray(tiles_grid(largura, altura))
    intervalo = np.zeros((n, len(tiles)), bool)
    intervalo[::30] = True
    cobertura = np.zeros_like(intervalo)
    vistas = set()
    enviados = 0
    # O quadrado local contém toda a projeção do quadro em qualquer guinada.
    raio = int(np.ceil(np.hypot(largura, altura) * gsd / (2 * CELL))) + 1
    dj, di = np.mgrid[-raio:raio+1, -raio:raio+1]
    for inicio in range(0, n, 60):
        fim = min(inicio + 60, n)
        instantes = list(range(inicio, fim, 20))
        limite = intervalo[inicio:fim].sum()
        for passo, q in enumerate(instantes):
            cota = (passo+1)*limite//len(instantes) - passo*limite//len(instantes)
            e, norte, yaw = pose[q]
            i, j = di + int(np.floor(e / CELL)), dj + int(np.floor(norte / CELL))
            de, dn = (i * CELL - e) / gsd, (j * CELL - norte) / gsd
            c, s = np.cos(yaw), np.sin(yaw)
            u, v = largura/2 + c*de - s*dn, altura/2 - s*de - c*dn
            vis = (u >= 0) & (u < largura) & (v >= 0) & (v < altura)
            u, v = u[vis], v[vis]
            chaves = list(zip(i[vis].tolist(), j[vis].tolist()))
            novas = np.fromiter((k not in vistas for k in chaves), bool, len(chaves))
            areas = [np.flatnonzero((u >= x) & (u < x+T) & (v >= y) & (v < y+T))
                     for x, y in tiles]
            ordem = list((7*(np.arange(len(tiles)) + enviados)) % len(tiles))
            for _ in range(cota):
                tile = max(ordem, key=lambda t: novas[areas[t]].sum())
                ordem.remove(tile)
                ids = areas[tile]
                vistas.update(chaves[k] for k in ids[novas[ids]])
                novas[ids] = False
                cobertura[q, tile] = True
            enviados += cota
        assert cobertura[inicio:fim].sum() == limite
    return cobertura, intervalo


def preparar(video):
    with np.load(CACHE / f"{video}.npz") as base:
        n, fps, size, gsd = (base[k].copy() for k in ("total_frames","fps","size","gsd"))
        tiles = base["tiles"]
    log, = (folder(video) / "log").glob("*.csv")
    pose = np.column_stack(load_pose(log, int(n), float(fps)))
    assert np.array_equal(tiles, tiles_grid(*size))
    masks = np.stack(plano(pose, size, gsd), axis=-1)
    requested = masks.any(axis=2)
    frames = np.flatnonzero(requested.any(axis=1))
    gravar_npz(TEMP / f"{video}.npz", total_frames=n, fps=fps, pose=pose[frames],
               frames=frames, starts=np.array([0]), tiles=tiles, gsd=gsd, size=size,
               masks=masks[frames], requested=requested[frames])
    print(video, "recortes por estratégia", masks.sum(axis=(0,1)), flush=True)


def inferir():
    import torch
    from inferir import infer_video
    from motores import runner, setup
    config = json.loads((OUT / "trajetoria/protocolo.json").read_text(encoding="utf-8"))
    assert config["quadros_escolhidos"] == 6 and config["peso_escolhido"] == 0
    verificar_pesos()
    setup()
    with FileLock(str(TEMP / "inferencia.lock")), torch.inference_mode():
        run = runner(MOTOR)
        for video in VOOS:
            if not (TEMP / f"{video}.npz").exists():
                preparar(video)
            infer_video(video, MOTOR, run, cache=TEMP, sparse=True, leitura_continua=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["inferir", "avaliar"])
    args = ap.parse_args()
    TEMP.mkdir(exist_ok=True)
    if args.acao == "inferir":
        inferir()
    else:
        from alertas_cobertura import avaliar
        avaliar()
