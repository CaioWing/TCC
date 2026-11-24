"""Convenções visuais dos exemplos geográficos sobre quadros reais."""
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from memoria_sim import T

ENVIO, VISIVEL, BORDA = "#ffb538", "#75e4ff", "#172633"


def itens_legenda(unidade):
    itens = [Line2D([], [], ls="", marker=forma, mfc=cor, mec=BORDA, ms=5, label=nome)
             for forma, cor, nome in [("D", ENVIO, f"{unidade} que pede envio"),
                                      ("o", VISIVEL, f"{unidade} sem pedido")]]
    itens.append(Patch(fill=False, ec="#c27713", lw=1.3, label="Recorte enviado"))
    return itens


def legenda(fig, unidade):
    fig.legend(handles=itens_legenda(unidade), loc="upper center", ncol=3, frameon=False, fontsize=8.5,
               borderaxespad=0, columnspacing=1.5, handletextpad=.45)


def painel(ax, foto, tamanho, recortes, visiveis, pedidos, referencia, letra, duas_faixas=False):
    w, h = tamanho
    ax.imshow(foto, extent=(0, w, h, 0))
    if duas_faixas:
        ax.axhline(h / 2, color="white", lw=.95, ls=(0, (4, 3)),
                   path_effects=[patheffects.withStroke(linewidth=2, foreground=BORDA)])
    for tile in recortes:
        ax.add_patch(Rectangle(tile, T, T, fill=False, ec=ENVIO, lw=1.,
                               path_effects=[patheffects.withStroke(linewidth=1.9, foreground=BORDA)]))
    for pontos, forma, cor, area in [(visiveis, "o", VISIVEL, 11), (pedidos, "D", ENVIO, 23)]:
        ax.scatter(*pontos.T, marker=forma, s=area, facecolor=cor, edgecolor=BORDA,
                   linewidth=.65, zorder=3, clip_on=False)
    u, v = referencia
    ax.annotate(letra, (u, v), xytext=(10 if u < .88*w else -10, -10 if v < h/2 else 10),
                textcoords="offset points", ha="center", va="center", fontsize=7.5,
                weight="bold", color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=.15", fc=BORDA, ec="white", lw=.65),
                arrowprops=dict(arrowstyle="-", color="white", lw=.8, shrinkB=1.5))
    ax.set(xlim=(0, w), ylim=(h, 0))
    ax.set_axis_off()


def contagem(n):
    return f"{n} recorte" + ("" if n == 1 else "s")


def amostra_de_celulas(atual, pendentes, grade, enviados, celula):
    """Amostra fixa do terreno e ao menos uma célula que justifica cada recorte."""
    j0, _, i0, _, dentro, u, v = atual
    j, i = np.indices(dentro.shape)
    ref = (celula[0] - j0, celula[1] - i0)
    assert dentro[ref]
    amostra = dentro & ((j + j0 - celula[0]) % 16 == 0) & ((i + i0 - celula[1]) % 16 == 0)
    gatilhos = amostra & pendentes
    for tile, enviado in zip(grade, enviados):
        x, y = tile
        contem = (x <= u) & (u < x + T) & (y <= v) & (v < y + T)
        candidatos = contem & pendentes
        assert candidatos.any() == enviado
        if enviado and not (gatilhos & contem).any():
            linhas, colunas = np.nonzero(candidatos)
            dist = (u[candidatos] - x - T/2)**2 + (v[candidatos] - y - T/2)**2
            idx = dist.argmin()
            gatilhos[linhas[idx], colunas[idx]] = True
    outros = amostra & ~pendentes
    return (np.column_stack((u[outros], v[outros])), np.column_stack((u[gatilhos], v[gatilhos])),
            (u[ref], v[ref]), bool(pendentes[ref]))
