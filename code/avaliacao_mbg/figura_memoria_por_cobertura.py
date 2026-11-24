"""Os dois primeiros envios da memória por cobertura no voo 09 completo."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from cobertura_continua import METODOS, TEMP
from dados import ROOT, folder
from figura_exemplo_temporal import fotos_do_video, mostrar_selecao
from figura_passagem import pontos_no_quadro
from memoria_sim import T, grade_projecao, load_pose
from paleta_tcc import BLUE
from passagens import DESTINO as DADOS_PASSAGENS

VIDEO = "video09_regionE"
QUADROS = (0, 20)
SEM_MARCA = "#a5540b"
PASSAGEM = "#ffbf40"


def exemplo():
    with np.load(TEMP / f"{VIDEO}.npz") as dados:
        n, fps, gsd = int(dados["total_frames"]), float(dados["fps"]), float(dados["gsd"])
        tamanho, recortes = dados["size"], dados["tiles"]
        selecoes = dados["masks"][:, :, METODOS.index("cobertura")]
        instantes = np.flatnonzero(selecoes.any(axis=1))[:2]
        np.testing.assert_array_equal(dados["frames"][instantes], QUADROS)
        selecoes = selecoes[instantes]
        np.testing.assert_array_equal(selecoes.sum(axis=1), [16, 16])

    log, = (folder(VIDEO) / "log").glob("*.csv")
    pose = np.column_stack(load_pose(log, n, fps))
    eventos = pd.read_csv(DADOS_PASSAGENS / "eventos.csv").query("video == @VIDEO")
    forma, projetar = grade_projecao(log, n, fps, *tamanho, gsd, centros_na_origem=True)
    memoria = np.zeros(forma, dtype=bool)
    mapas = []
    for quadro, selecao in zip(QUADROS, selecoes):
        j0, j1, i0, i1, visiveis, u, v = projetar(quadro)
        marcas = memoria[j0:j1, i0:i1]
        centros = [(x <= u) & (u < x + T) & (y <= v) & (v < y + T) for x, y in recortes]
        if quadro == 0:
            for posicao in np.flatnonzero(selecao):
                marcas |= centros[posicao]
        else:
            ganhos = np.array([np.count_nonzero(dentro & ~marcas) for dentro in centros])
            primeira = ganhos.argmax()
            # Neste envio o máximo é único, sem necessidade de reproduzir o desempate.
            assert (ganhos == ganhos[primeira]).sum() == 1 and selecao[primeira]
        pontos = pontos_no_quadro(eventos, pose, tamanho, quadro)
        mapas.append((u[visiveis], v[visiveis], marcas[visiveis], pontos))
    return tamanho, recortes, selecoes, mapas, recortes[primeira], int(ganhos[primeira])


def mostrar_mapa(ax, mapa, tamanho):
    u, v, marcadas, pontos = mapa
    ax.scatter(u[marcadas], v[marcadas], s=1.35, c=BLUE, marker="o", linewidths=0)
    ax.scatter(u[~marcadas], v[~marcadas], s=2.45, c=SEM_MARCA, marker="D", linewidths=0)
    ax.scatter(pontos.u, pontos.v, s=27, marker="^", facecolor=PASSAGEM,
               edgecolor="#172633", linewidth=.8, zorder=3)
    ax.set(xlim=(0, tamanho[0]), ylim=(tamanho[1], 0), aspect="equal", xticks=[], yticks=[])
    for borda in ax.spines.values():
        borda.set_color("#879098")
        borda.set_linewidth(.5)


def destacar_primeira(ax, origem):
    x, y = origem
    for cor, espessura in [("white", 3), ("#172633", 1.6)]:
        ax.add_patch(Rectangle((x, y), T, T, fill=False, ec=cor, lw=espessura, zorder=4))
    ax.text(x + T / 2, y + T / 2, "1", color="white", fontsize=8, weight="bold",
            ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="circle,pad=.18", fc="#172633", ec="white", lw=.65))


def desenhar():
    tamanho, recortes, selecoes, mapas, primeira, ganho = exemplo()
    fotos = fotos_do_video(QUADROS, VIDEO)
    fig, eixos = plt.subplots(2, 2, figsize=(7.1, 4.52))
    fig.subplots_adjust(left=.012, right=.985, bottom=.10, top=.93, wspace=.07, hspace=.30)
    a, b, c, d = eixos.flat
    mostrar_selecao(a, fotos[0], recortes[selecoes[0]])
    mostrar_mapa(b, mapas[0], tamanho)
    mostrar_mapa(c, mapas[1], tamanho)
    mostrar_selecao(d, fotos[1], recortes[selecoes[1]])
    destacar_primeira(d, primeira)
    for ax, titulo in zip(eixos.flat, ("(a) Quadro 0 · 16 recortes enviados",
                                      "(b) Quadro 0 · memória após o envio",
                                      "(c) Quadro 20 · memória antes do envio",
                                      "(d) Quadro 20 · 16 recortes enviados")):
        ax.set_title(titulo, loc="left", fontsize=10.1, pad=5)
    itens = [Line2D([], [], ls="", marker=forma, color=cor, ms=5.4, label=rotulo)
             for forma, cor, rotulo in [("o", BLUE, "Célula já marcada"),
                                        ("D", SEM_MARCA, "Célula sem marca")]]
    itens.append(Line2D([], [], ls="", marker="^", mfc=PASSAGEM, mec="#172633", mew=.8,
                        ms=5.8, label="Ponto de passagem"))
    itens.append(Line2D([], [], ls="", marker="$①$", color="#172633", ms=9.5,
                        label=f"Primeira escolha ({ganho} células novas)"))
    fig.legend(handles=itens, loc="lower center", ncol=4, frameon=False, fontsize=8.7,
               handlelength=1.25, handletextpad=.35, columnspacing=1.)
    fig.savefig(ROOT / "doc/figuras/fig_memoria_por_cobertura.pdf", bbox_inches="tight",
                pad_inches=.045, dpi=280)
    plt.close(fig)
    print(f"Primeira escolha do quadro 20: {primeira.tolist()}, {ganho} células sem marca.")


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        desenhar()
