"""Simulacao geometrica da memoria geografica com a pose real da telemetria (MBG-V2).

Para cada quadro do video: pegada projetada no solo pela Eq. eq_georref (camera nadir, guinada
da camera = gimbalYawRaw/10, GSD ajustado do voo), grade de celulas de 0,5 m e grade fixa de
recortes 640x640 (6x4 em 3840x2160; 7x4 em 4096x2160). n_v* = 1.

Politicas:
  imediata: a cada quadro, processa todo recorte que contem alguma celula ainda nao analisada;
  adiada:   processa um recorte so quando contem celula nao analisada que sai da pegada no
            quadro seguinte (ultima oportunidade de ve-la).
Cada recorte processado marca como analisadas todas as celulas que cobre.
Referencia (skip_steps): quadro inteiro (todos os recortes) a cada k quadros; mede a fracao
da area imageada que nao aparece em nenhum quadro amostrado.

Nao usa o detector: mede apenas a demanda de passadas e a cobertura das celulas.
"""
import os
import numpy as np
import pandas as pd

CELL = 0.5
T = 640
A_WGS, E2 = 6378137.0, 6.69437999014e-3


def tiles_grid(W, H):
    xs = np.round(np.linspace(0, W - T, int(np.ceil(W / T)))).astype(int)
    ys = np.round(np.linspace(0, H - T, int(np.ceil(H / T)))).astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


def load_pose(log_csv, nfr, fps):
    L = pd.read_csv(log_csv, usecols=['time(millisecond)', 'latitude', 'longitude', 'gimbalYawRaw', 'remainPowerPercent'])
    L = L[L.remainPowerPercent > 0].groupby('time(millisecond)', as_index=False).mean(numeric_only=True)
    tl = (L['time(millisecond)'] - L['time(millisecond)'].iloc[0]).to_numpy() / 1000.0
    lat0 = np.deg2rad(L.latitude.mean()); lon0 = L.longitude.mean(); latm = L.latitude.mean()
    RM = A_WGS * (1 - E2) / (1 - E2 * np.sin(lat0) ** 2) ** 1.5
    RN = A_WGS / np.sqrt(1 - E2 * np.sin(lat0) ** 2)
    tf = np.arange(nfr) / fps
    N = np.deg2rad(np.interp(tf, tl, L.latitude) - latm) * RM
    E = np.deg2rad(np.interp(tf, tl, L.longitude) - lon0) * RN * np.cos(lat0)
    yaw = np.interp(tf, tl, np.unwrap(np.deg2rad(L.gimbalYawRaw.to_numpy() / 10.0)))
    return E, N, yaw


def grade_projecao(log_csv, nfr, fps, W, H, gsd_m, *, centros_na_origem=False):
    """Grade comum do voo e projeção dos centros das células em cada quadro."""
    E0, N0, yaw = load_pose(log_csv, nfr, fps)

    def img2ground(k, u, v):
        c, s = np.cos(yaw[k]), np.sin(yaw[k]); a = -(v - H / 2); b = u - W / 2
        return E0[k] + gsd_m * (s * a + c * b), N0[k] + gsd_m * (c * a - s * b)

    def ground2img(k, e, n):
        c, s = np.cos(yaw[k]), np.sin(yaw[k]); de = (e - E0[k]) / gsd_m; dn = (n - N0[k]) / gsd_m
        a = c * dn + s * de; b = -s * dn + c * de
        return b + W / 2, H / 2 - a

    cu = np.array([0, W, W, 0]); cv = np.array([0, 0, H, H])
    foot = np.array([np.column_stack(img2ground(k, cu, cv)) for k in range(nfr)])
    def eixo(coordenadas):
        if centros_na_origem:
            # A política de cobertura usa centros em múltiplos de CELL nos eixos locais.
            return CELL * np.arange(np.floor(coordenadas.min()/CELL) - 1,
                                    np.ceil(coordenadas.max()/CELL) + 2)
        return np.arange(coordenadas.min() - 2, coordenadas.max() + 2, CELL) + CELL / 2

    ge, gn = eixo(foot[..., 0]), eixo(foot[..., 1])
    extremo = "right" if centros_na_origem else "left"

    def raster(k):
        bb = foot[k]
        i0, i1 = np.searchsorted(ge, bb[:, 0].min()), np.searchsorted(ge, bb[:, 0].max(), side=extremo)
        j0, j1 = np.searchsorted(gn, bb[:, 1].min()), np.searchsorted(gn, bb[:, 1].max(), side=extremo)
        EE, NN = np.meshgrid(ge[i0:i1], gn[j0:j1])
        u, v = ground2img(k, EE, NN)
        return j0, j1, i0, i1, (u >= 0) & (u < W) & (v >= 0) & (v < H), u, v

    return (len(gn), len(ge)), raster


def simulate(log_csv, nfr, fps, W, H, gsd_m, skip_steps=(), on_deferred_tile=None,
             on_selected=None, on_deferred_frame=None):
    shape, raster = grade_projecao(log_csv, nfr, fps, W, H, gsd_m)
    tiles = tiles_grid(W, H)
    out = {}
    imaged = np.zeros(shape, bool)
    for pol in ('imediata', 'adiada'):
        vis = np.zeros(shape, np.int32); nt = np.zeros(nfr, np.int32)
        nxt = raster(0)
        for k in range(nfr):
            current = nxt
            j0, j1, i0, i1, m, u, v = current
            if k + 1 < nfr:
                nxt = raster(k + 1)
            if pol == 'adiada' and on_deferred_frame is not None:
                on_deferred_frame(k, current, nxt if k + 1 < nfr else None, vis.shape)
            if pol == 'imediata':
                imaged[j0:j1, i0:i1] |= m
            sub = vis[j0:j1, i0:i1]
            need = m & (sub < 1)
            if pol == 'adiada' and k + 1 < nfr:
                a0, a1, b0, b1, m2, _, _ = nxt
                inn = np.zeros(vis.shape, bool); inn[a0:a1, b0:b1] = m2
                need &= ~inn[j0:j1, i0:i1]
            if not need.any():
                continue
            uu, vv = u[need], v[need]
            for (x, y) in tiles:
                in_tile = (uu >= x) & (uu < x + T) & (vv >= y) & (vv < y + T)
                if in_tile.any():
                    sub[m & (u >= x) & (u < x + T) & (v >= y) & (v < y + T)] += 1
                    nt[k] += 1
                    if on_selected is not None:
                        on_selected(pol, k, x, y)
                    if pol == 'adiada' and on_deferred_tile is not None:
                        on_deferred_tile(k, x, y, uu[in_tile], vv[in_tile])
        a = vis[imaged]
        out[pol] = dict(recortes=int(nt.sum()), recortes_s=float(nt.sum() / (nfr / fps)),
                        frac_nao_analisada=float((a == 0).mean()), analises_med=float(np.median(a)))
    out['area_img_m2'] = float(imaged.sum() * CELL ** 2)
    out['n_tiles_quadro'] = len(tiles)
    if skip_steps:
        for kstep in skip_steps:
            seen = np.zeros(imaged.shape, bool)
            for k in range(0, nfr, kstep):
                j0, j1, i0, i1, m, _, _ = raster(k)
                seen[j0:j1, i0:i1] |= m
            out[f'skip_{kstep}'] = dict(recortes_s=len(tiles) * fps / kstep,
                                        frac_nao_vista=float((~seen[imaged]).mean()))
    return out
