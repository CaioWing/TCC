"""Primeira vista e mudança de quadrante de um ponto no voo 09 completo."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from dados import CACHE, ROOT
from figura_exemplo_temporal import fotos_do_video
from figura_passagem import pontos_no_quadro
from memoria_sim import T
from passagens import DESTINO, carregar_voo, melhor_recorte

VIDEO = "video09_regionE"
QUADROS = (0, 92, 93, 94)
PONTO_A = (-15, 7)
PEDIDO = "#ffbd42"
OBSERVADO = "#75d9fb"
CONTORNO = "#152632"


def carregar_paineis():
    pose, tamanho, grade, _ = carregar_voo(VIDEO)
    eventos = pd.read_csv(DESTINO / "eventos.csv").query("video == @VIDEO")
    with np.load(CACHE / "quadrantes_v1" / f"{VIDEO}.npz") as dados:
        selecoes = np.zeros((len(pose), len(grade)), dtype=bool)
        selecoes[dados["frames"]] = dados["masks"][:, :, 0]

    vistos = set()
    paineis = []
    largura, altura = tamanho
    for q in range(QUADROS[-1] + 1):
        pontos = pontos_no_quadro(eventos, pose, tamanho, q)
        quadrantes = (pontos.u >= largura / 2).astype(int) + 2 * (pontos.v >= altura / 2).astype(int)
        chaves = list(zip(pontos.i, pontos.j, pontos.passagem, quadrantes))
        pontos["pedido"] = [chave not in vistos for chave in chaves]
        pedidos = [melhor_recorte(p.u, p.v, grade) for p in pontos[pontos.pedido].itertuples()]
        indices = np.flatnonzero(selecoes[q])
        # Nos quadros exibidos e em todo o seu histórico, cada pedido foi atendido.
        np.testing.assert_array_equal(np.unique(pedidos), indices)
        for indice in indices:
            x, y = grade[indice]
            cobertos = (x <= pontos.u) & (pontos.u < x + T) & (y <= pontos.v) & (pontos.v < y + T)
            vistos.update(chaves[k] for k in np.flatnonzero(cobertos))
        if q in QUADROS:
            paineis.append((pontos, grade[indices]))
    np.testing.assert_array_equal([len(tiles) for _, tiles in paineis], [18, 0, 1, 0])
    return tamanho, paineis


def desenhar():
    tamanho, paineis = carregar_paineis()
    largura, altura = tamanho
    fig, eixos = plt.subplots(2, 2, figsize=(7.25, 4.8))
    fig.subplots_adjust(left=.005, right=.995, top=.98, bottom=.10, wspace=.05, hspace=.21)
    estilos = [(True, "D", PEDIDO, "Ponto que pede recorte"),
               (False, "o", OBSERVADO, "Ponto já visto no quadrante")]
    legenda = [Line2D([], [], ls="", marker=marca, mfc=cor, mec=CONTORNO, ms=5, label=rotulo)
               for _, marca, cor, rotulo in estilos]
    legenda.append(Patch(fill=False, edgecolor="#ad630a", lw=1.5, label="Recorte enviado"))
    fig.legend(handles=legenda, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               handlelength=1.5, handletextpad=.45, columnspacing=1.7)
    titulos = ["Quadro 0 · 18 recortes", "Quadro 92 · nenhum recorte",
               "Quadro 93 · 1 recorte", "Quadro 94 · nenhum recorte"]
    fotos = fotos_do_video(QUADROS, VIDEO)
    for ax, foto, (pontos, enviados), titulo in zip(eixos.flat, fotos, paineis, titulos, strict=True):
        ax.imshow(foto, extent=(0, largura, altura, 0))
        estilo_eixo = dict(color="white", lw=.8, ls=(0, (3, 3)), zorder=2,
                           path_effects=[patheffects.withStroke(linewidth=1.8, foreground=CONTORNO)])
        ax.axvline(largura / 2, **estilo_eixo)
        ax.axhline(altura / 2, **estilo_eixo)
        for x, y in enviados:
            ax.add_patch(Rectangle((x, y), T, T, fill=False, ec=PEDIDO, lw=1.1, zorder=3,
                                   path_effects=[patheffects.withStroke(linewidth=1.95, foreground=CONTORNO)]))
        for pedido, forma, cor, _ in estilos:
            grupo = pontos[pontos.pedido == pedido]
            ax.scatter(grupo.u, grupo.v, marker=forma, s=17, facecolor=cor,
                       edgecolor=CONTORNO, linewidth=.75, zorder=4, clip_on=False)
        alvo = pontos[(pontos.i == PONTO_A[0]) & (pontos.j == PONTO_A[1])].iloc[0]
        ax.annotate("A", (alvo.u, alvo.v), xytext=(11, 10), textcoords="offset points",
                    ha="center", va="center", fontsize=8, weight="bold", color="white", zorder=5,
                    bbox=dict(boxstyle="round,pad=.14", fc=CONTORNO, ec="white", lw=.7),
                    arrowprops=dict(arrowstyle="-", color="white", lw=.9, shrinkB=2))
        ax.set_title(titulo, loc="left", fontsize=10.2, pad=4)
        ax.set(xlim=(0, largura), ylim=(altura, 0))
        ax.set_axis_off()
    fig.savefig(ROOT / "doc/figuras/fig_memoria_quadrantes.pdf", bbox_inches="tight",
                pad_inches=.035, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        desenhar()
