"""Compara as metas de faixas e o ganho da segunda faixa nos seis voos."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from dados import CACHE, OUT, ROOT, folder
from figura_exemplo_temporal import fotos_do_video
from figuras_malha import amostra_de_celulas, contagem, itens_legenda, painel
from memoria_sim import grade_projecao
from metricas import CLASSES, box_ious, load_clips, selected_tiles
from multivistas import MemoriaFaixas, altura_seguinte
from paleta_tcc import BLUE, ORANGE
from seguimento import MOTOR, TEMP

DADOS = OUT / "seguimento"
COTAS = ("natural", "fixo_bruto")


def painel_metas(ax, parte, titulo, fixa):
    geo = parte[parte.metodo == "memoria"].set_index("k").sort_index()
    temporal = parte[parte.metodo == "temporal"]
    assert temporal.groupby("k").size().eq(4).all()
    assert parte.groupby("k").recortes.nunique().eq(1).all()
    fases = temporal.groupby("k").vp.agg(["min", "mean", "max"]).loc[geo.index]
    faixa = ax.fill_between(geo.index, fases["min"].to_numpy(), fases["max"].to_numpy(),
                            color=ORANGE, alpha=.19, linewidth=0)
    media, = ax.plot(geo.index, fases["mean"], color=ORANGE, ls="--", marker="s", ms=3.7)
    pontos, = ax.plot(geo.index, geo.vp, color=BLUE, marker="o", ms=4)
    ganho = int(geo.at[2, "vp"] - geo.at[1, "vp"])
    ax.annotate(f"{ganho:+d}", (2, geo.at[2, "vp"]), xytext=(2, -12),
                textcoords="offset points", ha="center", color=BLUE, weight="bold")
    ax.set_title(titulo, loc="left")
    ax.set_xlim(.78, 5.22)
    ax.set_ylim(0, 200)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xticks(geo.index)
    if fixa:
        (cota,) = geo.recortes.unique()
        ax.set_xlabel(f"Faixas da imagem\n({cota} recortes por meta)")
    else:
        cotas = [f"{n:,}".replace(",", ".") for n in geo.recortes]
        ax.set_xticklabels([f"{k}\n{n}" for k, n in zip(geo.index, cotas)])
        ax.set_xlabel("Faixas da imagem\ne total de recortes")
    ax.grid(axis="y", alpha=.22)
    return pontos, media, faixa


def painel_voos(ax, trechos, resumo):
    selecao = trechos[(trechos.metodo == "memoria") & trechos.orcamento.isin(COTAS)
                      & trechos.k.isin([1, 2])]
    totais = selecao.groupby(["orcamento", "k"]).vp.sum()
    agregado = resumo[resumo.metodo == "memoria"].set_index(["orcamento", "k"]).vp
    assert totais.eq(agregado.loc[totais.index]).all()
    por_voo = selecao.groupby(["video", "orcamento", "k"]).vp.sum().unstack("k")
    saldos = (por_voo[2] - por_voo[1]).unstack("orcamento").sort_index()
    y = np.arange(len(saldos))
    ax.axvline(0, color=".48", lw=.8, zorder=0)
    ax.hlines(y, saldos.fixo_bruto, saldos.natural, color="#abb6bf", lw=1)
    ax.scatter(saldos.natural, y, c=BLUE, edgecolors=BLUE, s=23, zorder=3)
    ax.scatter(saldos.fixo_bruto, y, facecolors="white", edgecolors=BLUE, s=23, zorder=3)
    rotulos = [video.removeprefix("video").replace("_region", " · ") for video in saldos.index]
    ax.set_yticks(y, rotulos)
    ax.set_ylim(len(saldos)-.45, -.65)
    ax.set_xlim(-9, 12.5)
    ax.set_xticks([-5, 0, 5, 10], ["−5", "0", "+5", "+10"])
    ax.set_title("(c) Saldo da segunda faixa", loc="left", pad=15)
    ax.set_xlabel("Acertos com duas faixas\nmenos acertos com uma")
    ax.grid(axis="x", alpha=.17)
    ax.text(.02, 1.01, "● Solicitada    ○ Fixa", transform=ax.transAxes,
            va="bottom", fontsize=7, color=BLUE)


def exemplo_do_seguimento(fig, eixos):
    """Reproduz um acerto exclusivo de duas faixas sob a cota solicitada."""
    video, quadro, trilha, trecho = "video23_regionL", 1413, 28, 1
    with np.load(CACHE / f"{video}.npz") as voo:
        n, fps, gsd = int(voo["total_frames"]), float(voo["fps"]), float(voo["gsd"])
        w, h = voo["size"]
        tiles = voo["tiles"]
    with np.load(TEMP / f"{video}.npz") as cache:
        frame, = np.flatnonzero(cache["frames"] == quadro)
        masks = {k: cache[f"geo{k}"][frame] for k in (1, 2)}
        assert not masks[1].any() and masks[2].sum() == 2

    clip = next(c for c in load_clips(video, MOTOR, cache=TEMP) if c["clip"] == trecho)
    idx, = np.flatnonzero(clip["frames"] == quadro)
    anotada, = clip["gt"][idx][clip["gt"][idx][:, 4] == trilha]
    cls = CLASSES.index("pool")
    assert anotada[5] == cls
    deteccoes, = selected_tiles({"tiles": [clip["tiles"][idx]]}, [masks[2]])
    classe = deteccoes[(deteccoes[:, 5] == cls) & (deteccoes[:, 4] >= .5)]
    iou = box_ious(classe, anotada[None])[:, 0]
    caixa, = classe[iou >= .5]
    pareamento = pd.read_csv(DADOS / "pareamento.csv")
    ref = pareamento.query("video == @video and trecho == @trecho and classe == 'pool' and trilha == @trilha")
    np.testing.assert_array_equal(ref[["natural_memoria1_-1", "natural_memoria2_-1"]].values, [[0, 1]])
    centro = (anotada[:2] + anotada[2:4])/2

    log, = (folder(video) / "log").glob("*.csv")
    forma, raster = grade_projecao(log, n, fps, w, h, gsd)
    memoria = MemoriaFaixas(n, tiles, h)
    atual = raster(0)
    for t in range(quadro):
        seguinte = raster(t+1)
        memoria.frame(t, atual, seguinte, forma)
        atual = seguinte
    j0, j1, i0, i1, dentro, u, v = atual
    seguinte = raster(quadro+1)
    prox = altura_seguinte(atual, seguinte)
    pedem = {}
    for k in (1, 2):
        faixa = np.clip((v*k/h).astype(int), 0, k-1)
        depois = np.clip((prox*k/h).astype(int), 0, k-1)
        sai = (prox < 0) | (faixa != depois)
        vistas = memoria.maps[k][j0:j1, i0:i1]
        pedem[k] = dentro & sai & ((vistas & (1 << faixa)) == 0)
    memoria.frame(quadro, atual, seguinte, forma)
    linhas, colunas = np.nonzero(pedem[2])
    indice = np.argmin((u[pedem[2]] - centro[0])**2 + (v[pedem[2]] - centro[1])**2)
    ancora = linhas[indice] + j0, colunas[indice] + i0
    foto, = fotos_do_video([quadro], video)
    for k, ax in zip((1, 2), eixos):
        np.testing.assert_array_equal(memoria.selected[k][quadro], masks[k])
        outros, ativos, _, _ = amostra_de_celulas(atual, pedem[k], tiles, masks[k], ancora)
        painel(ax, foto, (w, h), tiles[masks[k]], outros, ativos, centro, "P", duas_faixas=k==2)
        nome = "Uma faixa" if k == 1 else "Duas faixas"
        ax.set_title(f"{nome} · {contagem(int(masks[k].sum()))}", loc="left", fontsize=8.2, pad=2)
    x, y, x2, y2 = caixa[:4]
    eixos[1].add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, ec="white", lw=1.05,
                                 path_effects=[patheffects.withStroke(linewidth=2., foreground="#172633")]))
    fig.legend(handles=itens_legenda("Célula"), loc="lower center", ncol=3, frameon=False,
               fontsize=7.4, borderaxespad=0, handletextpad=.45)
    print(f"Voo 23, trecho {trecho}, quadro {quadro}, trilha {trilha}: "
          f"confiança {caixa[4]:.3f}, IoU {max(iou):.3f}.")


def desenhar():
    resumo = pd.read_csv(DADOS / "resumo.csv")
    trechos = pd.read_csv(DADOS / "por_trecho.csv")
    (referencias,) = resumo.referencias.unique()
    fig = plt.figure(figsize=(7.05, 5.25))
    grade = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, 1.02], left=.083, right=.986,
                             bottom=.655, top=.948, wspace=.34)
    eixos = [fig.add_subplot(pos) for pos in grade]
    h_geo, h_media, h_faixa = painel_metas(
        eixos[0], resumo[resumo.orcamento == COTAS[0]], "(a) Cota solicitada por meta", False)
    painel_metas(eixos[1], resumo[resumo.orcamento == COTAS[1]], "(b) Cota fixa entre metas", True)
    eixos[0].set_ylabel(f"Acertos entre {referencias} referências")
    eixos[1].tick_params(labelleft=False)
    painel_voos(eixos[2], trechos, resumo)
    fig.legend([h_geo, h_media, h_faixa],
               ["Saída de cada faixa", "Temporal por trecho (média)", "Extremos das quatro fases temporais"],
               frameon=False, loc="lower center", bbox_to_anchor=(.5, .479),
               ncol=3, fontsize=7.3, columnspacing=1.45)
    fig.text(.015, .462, "(d) Cota solicitada no quadro 1413 do voo 23", fontsize=8.7, va="top")
    imagens = fig.add_gridspec(1, 2, left=.012, right=.988, bottom=.056, top=.396, wspace=.06)
    exemplo_do_seguimento(fig, [fig.add_subplot(pos) for pos in imagens])
    fig.savefig(ROOT / "doc/figuras/fig_seguimento_multivistas.pdf",
                bbox_inches="tight", pad_inches=.04, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 8,
                         "axes.titlesize": 8.1, "axes.labelsize": 7.8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "lines.linewidth": 1.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True}):
        desenhar()
