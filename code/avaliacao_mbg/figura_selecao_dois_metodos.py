"""Recortes realmente solicitados pelos dois métodos no primeiro bloco do voo 09."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cobertura_continua import TEMP
from dados import ROOT
from figura_exemplo_temporal import fotos_do_video, mostrar_selecao


def desenhar():
    video = "video09_regionE"
    with np.load(TEMP / f"{video}.npz") as dados:
        bloco = dados["frames"] < 60
        quadros = dados["frames"][bloco]
        tiles = dados["tiles"]
        mascaras = dados["masks"][bloco]
    np.testing.assert_array_equal(quadros, [0, 20, 30, 40])
    np.testing.assert_array_equal(mascaras.sum(axis=(0, 1)), [48, 48])
    fotos = fotos_do_video(quadros, video)
    fig, eixos = plt.subplots(2, len(quadros), figsize=(7.4, 2.85))
    fig.subplots_adjust(left=.008, right=.992, top=.82, bottom=.02, wspace=.045, hspace=.37)
    for linha, nome in enumerate(["Memória por cobertura", "Um quadro a cada 30"]):
        for col, (quadro, foto) in enumerate(zip(quadros, fotos)):
            ax = eixos[linha, col]
            recortes = tiles[mascaras[col, :, linha]]
            mostrar_selecao(ax, foto, recortes)
            if linha == 0:
                ax.set_title(f"Quadro {quadro}", fontsize=9, pad=4)
            ax.text(.5, -.025, f"{len(recortes)} recortes", ha="center", va="top",
                    fontsize=8, transform=ax.transAxes)
        linha_y = eixos[linha, 0].get_position().y1
        fig.text(.008, linha_y + (.09 if linha == 0 else .025), nome, fontsize=10,
                 weight="semibold", va="bottom")
    fig.savefig(ROOT / "doc/figuras/fig_selecao_dois_metodos.pdf", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    desenhar()
