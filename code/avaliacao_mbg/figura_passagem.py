"""Mostra a entrada e a saída de um ponto com a mesma seleção geográfica."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from dados import GSD, ROOT
from figura_exemplo_temporal import VIDEO, fotos_do_video
from memoria_sim import T
from passagens import DESTINO, MASCARAS, carregar_voo, melhor_recorte

QUADROS = (47, 48, 246, 247)
# Índices leste/norte na malha do voo, mantidos em todos os quadros.
REFERENCIAS = {(-16, 5): ("A", (9, 8)), (-14, 7): ("B", (10, -10))}
ENTRADA, VISIVEL, BORDA = "#ffb538", "#75e4ff", "#172633"


def pontos_no_quadro(eventos, pose, tamanho, quadro):
    visiveis = eventos.query("quadro_entrada <= @quadro <= quadro_saida").copy()
    largura, altura = tamanho
    leste, norte, yaw = pose[quadro]
    de, dn = (visiveis.leste_m - leste) / GSD, (visiveis.norte_m - norte) / GSD
    c, s = np.cos(yaw), np.sin(yaw)
    visiveis["u"] = largura / 2 + c * de - s * dn
    visiveis["v"] = altura / 2 - s * de - c * dn
    assert visiveis.u.between(0, largura, inclusive="left").all()
    assert visiveis.v.between(0, altura, inclusive="left").all()
    return visiveis


def desenhar():
    pose, tamanho, grade, _ = carregar_voo(VIDEO)
    largura, altura = tamanho
    eventos = pd.read_csv(DESTINO / "eventos.csv").query("video == @VIDEO")
    with np.load(MASCARAS / f"{VIDEO}.npz") as dados:
        mascara = dados["extremos"]
    fig, eixos = plt.subplots(2, 2, figsize=(7.15, 4.72))
    fig.subplots_adjust(left=.005, right=.995, top=.87, bottom=.008, wspace=.055, hspace=.24)

    estilos = [(True, "D", ENTRADA, "Ponto que pede envio"),
               (False, "o", VISIVEL, "Ponto sem pedido")]
    legenda = [Line2D([], [], ls="", marker=forma, mfc=cor, mec=BORDA, ms=5, label=nome)
               for _, forma, cor, nome in estilos]
    legenda.append(Patch(fill=False, ec="#c27713", lw=1.3, label="Recorte enviado"))
    fig.legend(handles=legenda, loc="upper center", ncol=3, frameon=False,
               fontsize=8.6, handletextpad=.4, columnspacing=1.6, borderaxespad=0)

    for ax, quadro, foto in zip(eixos.flat, QUADROS, fotos_do_video(QUADROS)):
        visiveis = pontos_no_quadro(eventos, pose, tamanho, quadro)
        entrada = visiveis.quadro_entrada == quadro
        saida = visiveis.quadro_saida == quadro
        pedido = entrada | saida
        enviados = np.flatnonzero(mascara[quadro])
        recortes = [melhor_recorte(p.u, p.v, grade) for p in visiveis[pedido].itertuples()]
        np.testing.assert_array_equal(np.unique(recortes), enviados)

        ax.imshow(foto, extent=(0, largura, altura, 0))
        for x, y in grade[enviados]:
            ax.add_patch(Rectangle((x, y), T, T, fill=False, ec=ENTRADA, lw=1.,
                                   path_effects=[patheffects.withStroke(linewidth=1.9, foreground=BORDA)]))
        for solicitado, forma, cor, _ in estilos:
            pontos = visiveis[pedido == solicitado]
            ax.scatter(pontos.u, pontos.v, marker=forma, s=20, facecolor=cor,
                       edgecolor=BORDA, linewidth=.8, zorder=3, clip_on=False)
        for ponto in visiveis.itertuples():
            if (ponto.i, ponto.j) in REFERENCIAS:
                letra, deslocamento = REFERENCIAS[ponto.i, ponto.j]
                ax.annotate(letra, (ponto.u, ponto.v), xytext=deslocamento, textcoords="offset points",
                            ha="center", va="center", fontsize=7.2, weight="bold", color="white",
                            bbox=dict(boxstyle="round,pad=.15", fc=BORDA, ec="white", lw=.65),
                            arrowprops=dict(arrowstyle="-", color="white", lw=.85, shrinkB=1.5),
                            zorder=4)
        contagem = f"{len(enviados)} recorte" + ("" if len(enviados) == 1 else "s")
        ax.set_title(f"Quadro {quadro}  ·  {len(visiveis)} pontos  ·  {contagem}",
                     fontsize=9.2, loc="left", pad=4)
        ax.set(xlim=(0, largura), ylim=(altura, 0))
        ax.set_axis_off()
        print(f"Quadro {quadro}: {len(visiveis)} pontos, {entrada.sum()} entradas, {saida.sum()} saídas, {contagem}.")

    fig.savefig(ROOT / "doc/figuras/fig_passagem_geografica.pdf",
                bbox_inches="tight", pad_inches=.045, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        desenhar()
