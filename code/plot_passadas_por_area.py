"""Regenera o exemplo analítico de carga de inferência do capítulo 2."""
from pathlib import Path
from paleta_tcc import BLUE, ORANGE, GREEN

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter


def main():
    velocity = np.linspace(1, 15, 300)
    tiles, fps, width, tile_area, views = 24, 30, 53.8, 80.3, 5
    all_tiles = 1e6 * tiles * fps / (velocity * width)
    sampled_tiles = all_tiles / 6
    geographic = 1e6 * views / tile_area
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        fig.subplots_adjust(left=0.12, right=0.71, bottom=0.16, top=0.96)
        ax.plot(velocity, all_tiles, color=BLUE, lw=2)
        ax.plot(velocity, sampled_tiles, color=GREEN, lw=2)
        ax.axhline(geographic, color=ORANGE, lw=2)
        for y, label, color in [
            (all_tiles[-1], "24 recortes por quadro\n(30 quadros/s)", BLUE),
            (sampled_tiles[-1], "24 recortes por amostra\n(1 quadro a cada 6)", GREEN),
            (geographic, "Memória geográfica ideal\n(5 observações por ponto)", ORANGE),
        ]:
            ax.text(15.2, y, label, color=color, va="center", fontsize=10)
        ax.set(xlim=(1, 15), ylim=(5e4, 2e7), yscale="log",
               xlabel="Velocidade de voo $V$ (m/s)",
               ylabel="Passadas da rede por km² imageado")
        ax.yaxis.set_major_locator(FixedLocator([1e5, 1e6, 1e7]))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda x, _: {1e5: "100 mil", 1e6: "1 milhão", 1e7: "10 milhões"}.get(x, "")))
        ax.spines[["right", "top"]].set_visible(False)
        output = Path(__file__).resolve().parents[1] / "doc/figuras/fig_passadas_por_area.pdf"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
