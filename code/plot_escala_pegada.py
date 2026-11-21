"""Ilustra a projeção central e as relações GSD(h), w(h) e L(h).

As curvas usam GSD = 1,4 cm/pixel a 40 m e imagem de 3840 x 2160
pixels. São relações geométricas para uma câmera fixa, não medições
em diferentes alturas. Execute com Python, NumPy e Matplotlib.
"""

from pathlib import Path
from paleta_tcc import BLUE, ORANGE, GREEN

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


DARK = "0.12"
GRAY = "0.55"


def comma(value, _position=None):
    return f"{value:g}".replace(".", ",")


def dimension(ax, start, end, label, label_position, color=DARK):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.1})
    ax.text(*label_position, label, ha="center", va="center", color=color)


def main():
    heights = np.linspace(0, 85, 171)
    gsd = heights * (0.014 / 40)
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(9.5, 8.5))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.7, 1],
                               left=0.09, right=0.97, top=0.93,
                               bottom=0.07, wspace=0.32, hspace=0.48)
        diagram = fig.add_subplot(grid[0, 0])
        longitudinal = fig.add_subplot(grid[0, 1])
        sampling = fig.add_subplot(grid[1, 0])
        footprint = fig.add_subplot(grid[1, 1])

        diagram.set_title("(a) Vista transversal: largura", loc="left", pad=15)
        diagram.set_xlim(-3.15, 2.8)
        diagram.set_ylim(-4.65, 2.1)
        diagram.axis("off")
        # Sensor and ground are parallel. Every ray crosses the origin.
        sensor_y, ground_y = 1.2, -3.4
        sensor_halfwidth = 0.75
        ground_halfwidth = sensor_halfwidth * abs(ground_y) / sensor_y
        diagram.plot([0, 0], [-4.2, 1.9], ls="--", color="0.7", lw=1)
        for x in [-sensor_halfwidth, sensor_halfwidth]:
            diagram.plot([x, x * ground_y / sensor_y],
                         [sensor_y, ground_y], color=GRAY, lw=1.3)
        diagram.plot([-sensor_halfwidth, sensor_halfwidth],
                     [sensor_y, sensor_y], color=DARK, lw=3)
        diagram.text(-0.95, 1.63, "Sensor", ha="center")
        diagram.plot([-2.45, 2.45], [ground_y, ground_y], color=DARK, lw=1.5)
        diagram.text(2.45, -3.16, "Solo", ha="right")
        diagram.scatter([0], [0], s=35, color=DARK, zorder=4)
        diagram.annotate("Centro de\nprojeção", xy=(0, 0), xytext=(0.78, 0.1),
                         ha="left", va="center", fontsize=10,
                         arrowprops={"arrowstyle": "-", "color": DARK})

        # Exaggerated pixel spacing, explicitly identified as not to scale.
        pixel = 0.34
        ground_pixel = pixel * abs(ground_y) / sensor_y
        diagram.plot([pixel, -ground_pixel], [sensor_y, ground_y],
                     color=ORANGE, lw=1.6)
        diagram.plot([0, pixel], [sensor_y, sensor_y], color=ORANGE, lw=4)
        diagram.plot([-ground_pixel, 0], [ground_y, ground_y], color=ORANGE, lw=4)
        # Full transverse field of view, between the two footprint edges.
        half_angle = np.arctan(sensor_halfwidth / sensor_y)
        angles = np.linspace(-half_angle, half_angle, 100)
        radius = 0.8
        diagram.plot(radius * np.sin(angles), -radius * np.cos(angles),
                     color=DARK, lw=1.2)
        diagram.text(0, -1.02, r"$\beta_x$", ha="center", va="center",
                     fontsize=12, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        dimension(diagram, (0, 1.52), (pixel, 1.52), "$p$", (pixel / 2, 1.85), ORANGE)
        dimension(diagram, (-ground_pixel, -3.7), (0, -3.7),
                  "$GSD$", (-ground_pixel / 2, -3.99), ORANGE)
        for level in [sensor_y, 0, ground_y]:
            diagram.plot([-2.75, -0.82 if level == sensor_y else 0],
                         [level, level], color="0.7", ls=":", lw=0.8)
        dimension(diagram, (-2.7, 0), (-2.7, sensor_y), "$f_c$", (-2.97, 0.6))
        dimension(diagram, (-2.7, ground_y), (-2.7, 0), "$h$", (-2.97, -1.7))
        dimension(diagram, (-ground_halfwidth, -4.35), (ground_halfwidth, -4.35),
                  "$w$", (0, -4.62))

        # Orthogonal section of the same viewing geometry: the y direction.
        longitudinal.set_title("(b) Vista longitudinal: comprimento", loc="left", pad=15)
        longitudinal.set_xlim(diagram.get_xlim())
        longitudinal.set_ylim(diagram.get_ylim())
        longitudinal.axis("off")
        sensor_halflength = sensor_halfwidth * 2160 / 3840
        ground_halflength = sensor_halflength * abs(ground_y) / sensor_y
        longitudinal.plot([0, 0], [-4.2, 1.9], ls="--", color="0.7", lw=1)
        for x in [-sensor_halflength, sensor_halflength]:
            longitudinal.plot([x, x * ground_y / sensor_y],
                              [sensor_y, ground_y], color=GRAY, lw=1.3)
        longitudinal.plot([-sensor_halflength, sensor_halflength],
                          [sensor_y, sensor_y], color=DARK, lw=3)
        longitudinal.text(-0.95, 1.63, "Sensor", ha="center")
        longitudinal.plot([-2.45, 2.45], [ground_y, ground_y], color=DARK, lw=1.5)
        longitudinal.text(2.45, -3.16, "Solo", ha="right")
        longitudinal.scatter([0], [0], s=35, color=DARK, zorder=4)
        longitudinal.annotate("Centro de\nprojeção", xy=(0, 0), xytext=(0.78, 0.1),
                              ha="left", va="center", fontsize=10,
                              arrowprops={"arrowstyle": "-", "color": DARK})
        half_angle_y = np.arctan(sensor_halflength / sensor_y)
        angles_y = np.linspace(-half_angle_y, half_angle_y, 100)
        longitudinal.plot(radius * np.sin(angles_y), -radius * np.cos(angles_y),
                          color=DARK, lw=1.2)
        longitudinal.text(0, -1.02, r"$\beta_y$", ha="center", va="center",
                          fontsize=12, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        for level in [sensor_y, 0, ground_y]:
            longitudinal.plot([-2.75, -0.5 if level == sensor_y else 0],
                              [level, level], color="0.7", ls=":", lw=0.8)
        dimension(longitudinal, (-2.7, 0), (-2.7, sensor_y), "$f_c$", (-2.97, 0.6))
        dimension(longitudinal, (-2.7, ground_y), (-2.7, 0), "$h$", (-2.97, -1.7))
        dimension(longitudinal, (-ground_halflength, -4.35), (ground_halflength, -4.35),
                  "$L$", (0, -4.62))

        for ax in [sampling, footprint]:
            ax.spines[["right", "top"]].set_visible(False)
            ax.set_xlim(0, 85)
            ax.set_xticks([0, 20, 40, 60, 80])
            ax.set_xlabel("Altura de voo $h$ (m)")
            ax.yaxis.set_major_formatter(FuncFormatter(comma))
            ax.grid(axis="y", color="0.91", lw=0.7)
            ax.set_axisbelow(True)
            for height in [40, 80]:
                ax.axvline(height, color="0.8", ls=":", lw=0.9)

        sampling.set_title("(c) Distância entre amostras no solo", loc="left", pad=12)
        sampling.plot(heights, gsd * 100, color=ORANGE, lw=2)
        sampling.set_ylabel("GSD (cm/pixel)")
        sampling.set_ylim(0, 3.35)
        sampling.set_yticks([0, 0.7, 1.4, 2.1, 2.8])
        for height in [40, 80]:
            value = height * 0.035
            sampling.plot(height, value, "o", color=ORANGE, ms=4)

        footprint.set_title("(d) Dimensões da região imageada", loc="left", pad=12)
        footprint.plot(heights, 3840 * gsd, color=BLUE, lw=2, label="Largura $w$")
        footprint.plot(heights, 2160 * gsd, color=GREEN, lw=2, ls="--", label="Comprimento $L$")
        footprint.set_ylabel("Dimensão (m)")
        footprint.set_ylim(0, 130)
        footprint.set_yticks([0, 30, 60, 90, 120])
        footprint.legend(frameon=False, loc="upper left", fontsize=9, handlelength=2)
        for pixels, color in [(3840, BLUE), (2160, GREEN)]:
            for height in [40, 80]:
                value = pixels * height * 0.00035
                footprint.plot(height, value, "o", color=color, ms=4)

        output = Path(__file__).resolve().parents[1] / "doc/figuras/fig_escala_pegada.pdf"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
