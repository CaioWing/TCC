"""Memória por célula e faixa da imagem. Reproduz os voos desde o início."""
import argparse

import numpy as np
import pandas as pd

from dados import CACHE, OUT, ROOT, SPLITS, folder
from memoria_sim import T, simulate


METAS = (1, 2, 3, 4, 5)
DIR = OUT / "multivistas"
TEMP = CACHE / "multivistas"
BENCHMARK = ROOT / "data/yolo11n-jetson24-20261116/summary.csv"


def altura_seguinte(current, following):
    """Coordenada vertical no próximo quadro para os mesmos centros da grade do solo."""
    j0,j1,i0,i1,mask,_,_ = current
    v = np.full(mask.shape, -1.0)
    if following is not None:
        a0,a1,b0,b1,inside,_,next_v = following
        top,bottom = max(j0,a0), min(j1,a1)
        left,right = max(i0,b0), min(i1,b1)
        if top < bottom and left < right:
            block = np.s_[top-a0:bottom-a0,left-b0:right-b0]
            v[top-j0:bottom-j0,left-i0:right-i0] = np.where(inside[block], next_v[block], -1)
    return v


class MemoriaFaixas:
    def __init__(self, n, tiles, height):
        self.tiles = tiles
        self.height = height
        self.selected = {k: np.zeros((n,len(tiles)), dtype=bool) for k in METAS}
        self.maps = {}
        self.imaged = None

    def frame(self, frame, current, following, shape):
        j0,j1,i0,i1,inside,u,v = current
        if self.imaged is None:
            self.imaged = np.zeros(shape, dtype=bool)
            self.maps = {k: np.zeros(shape, dtype=np.uint8) for k in METAS}
        block = np.s_[j0:j1,i0:i1]
        self.imaged[block] |= inside
        next_v = altura_seguinte(current,following)
        # A faixa é um bit. Fora da pegada, o bit do quadro seguinte é zero.
        next_inside = next_v >= 0
        for k in METAS:
            band = np.clip((v * k / self.height).astype(int), 0, k-1)
            new_band = np.clip((next_v * k / self.height).astype(int), 0, k-1)
            leaving = inside & (~next_inside | (band != new_band))
            if not leaving.any():
                continue
            bits = np.left_shift(np.uint8(1),band.astype(np.uint8))
            state = self.maps[k][block]
            due = leaving & ((state & bits) == 0)
            if not due.any():
                continue
            x_due, y_due = u[due], v[due]
            covered = np.zeros(inside.shape, dtype=bool)
            for index,(x,y) in enumerate(self.tiles):
                if ((x_due>=x)&(x_due<x+T)&(y_due>=y)&(y_due<y+T)).any():
                    self.selected[k][frame,index] = True
                    covered |= (u>=x)&(u<x+T)&(v>=y)&(v<y+T)
            np.bitwise_or(state,bits,out=state,where=inside & covered)


def prepare(video):
    with np.load(CACHE / f"{video}.npz") as info:
        frames, tiles = info["frames"], info["tiles"]
        n, fps, gsd = int(info["total_frames"]), float(info["fps"]), float(info["gsd"])
        w,h = map(int,info["size"])
        original = info["masks"][:,:,4]
        original_control = info["masks"][:,:,3]
    log, = (folder(video) / "log").glob("*.csv")
    simulation = MemoriaFaixas(n,tiles.tolist(),h)
    base = np.zeros((n,len(tiles)), dtype=bool)
    indices = {tuple(tile): i for i,tile in enumerate(tiles)}
    def original_tile(policy, frame, x, y):
        if policy == "adiada":
            base[frame,indices[x,y]] = True
    simulate(log,n,fps,w,h,gsd,on_selected=original_tile,on_deferred_frame=simulation.frame)
    assert np.array_equal(base,simulation.selected[1]) and np.array_equal(base[frames],original)
    rows = []
    saved = {"frames":frames}
    bitcount = np.array([i.bit_count() for i in range(256)])
    bench = pd.read_csv(BENCHMARK).set_index(["config","mode"]).loc[("trt_fp16","e2e")]
    window = round(fps)
    for k, selection in simulation.selected.items():
        counts = bitcount[simulation.maps[k][simulation.imaged]]
        assert counts.min() >= 1 and counts.max() <= k
        step = max(1,round(len(tiles)*n/int(selection.sum())))
        control = np.broadcast_to((np.arange(n)%step == 0)[:,None],selection.shape)
        if k == 1:
            assert np.array_equal(control[frames],original_control)
        saved[f"geo{k}"],saved[f"tempo{k}"] = selection[frames],control[frames]
        per_window = np.convolve(selection.sum(axis=1),np.ones(window,dtype=int),"valid")*fps/window
        rate = selection.sum()*fps/n
        rows.append(dict(video=video,k=k,quadros=n,fps=fps,passo_temporal=step,
                         recortes_geo=selection.sum(),recortes_tempo=control.sum(),
                         taxa_geo=rate,taxa_tempo=control.sum()*fps/n,pico_1s_geo=per_window.max(),
                         fracao_janelas_acima_vazao_fp16=(per_window>bench.fps).mean(),
                         potencia_incremental_fp16_w=rate*bench.E_dyn_mJ_per_inf/1000,
                         recortes_geo_trechos=selection[frames].sum(),recortes_tempo_trechos=control[frames].sum(),
                         celulas=len(counts),celulas_com_k=(counts==k).sum(),
                         fracao_com_k=(counts==k).mean(),fracao_com_uma_ou_mais=(counts>=1).mean()))
    # Os passos temporais dependem da demanda geográfica de cada voo completo.
    # Na segunda reprodução, cada quadro temporal selecionado oferece todas as suas células.
    temporal_states = {k:np.zeros_like(simulation.maps[k]) for k in METAS}
    def temporal_frame(frame,current,following,shape):
        j0,j1,i0,i1,inside,_,v = current
        for row in rows:
            if frame % row["passo_temporal"] == 0:
                k = row["k"]
                band = np.clip((v*k/h).astype(int),0,k-1)
                bits = np.left_shift(np.uint8(1),band.astype(np.uint8))
                state = temporal_states[k][j0:j1,i0:i1]
                np.bitwise_or(state,bits,out=state,where=inside)
    simulate(log,n,fps,w,h,gsd,on_deferred_frame=temporal_frame)
    for row in rows:
        k = row["k"]
        counts = bitcount[temporal_states[k][simulation.imaged]]
        row["celulas_tempo_com_k"] = (counts==k).sum()
        row["celulas_tempo_com_uma_ou_mais"] = (counts>=1).sum()
    np.savez_compressed(TEMP / f"{video}.npz", **saved)
    pd.DataFrame(rows).to_csv(TEMP / f"{video}.csv",index=False)
    print(video,[(r["k"],int(r["recortes_geo_trechos"])) for r in rows],flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="*", default=SPLITS["voos"]["test"])
    args = p.parse_args()
    DIR.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    for name in args.videos:
        prepare(name)
