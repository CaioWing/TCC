"""Rotação dos deslocamentos da câmera para norte e leste."""

from pathlib import Path
from paleta_tcc import BLUE, ORANGE, GREEN

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def arrow(ax, start, end, color='0.2', lw=1.4, style='->'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': style, 'color': color, 'lw': lw})


def main():
    xc, yc = 0.7, 1.4
    orange = ORANGE
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.18, wspace=0.40)
        for ax, degrees, panel in zip(axes, [0, 45], ['a', 'b']):
            psi = np.deg2rad(degrees)
            front = np.array([np.sin(psi), np.cos(psi)])
            right = np.array([np.cos(psi), -np.sin(psi)])
            step = xc * front
            point = step + yc * right  # Plot coordinates: east, north.
            east, north = point
            ax.set_aspect('equal')
            ax.set_xlim(-0.4, 2.1)
            ax.set_ylim(-1.1, 1.2)
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'.replace('.', ',')))
            ax.spines[['right', 'top']].set_visible(False)
            ax.grid(color='0.90', lw=0.6)
            ax.set_axisbelow(True)
            ax.axhline(0, color='0.55', lw=0.8)
            ax.axvline(0, color='0.55', lw=0.8)
            ax.set_xlabel(r'$\Delta E$ (m; leste $\rightarrow$)', labelpad=10)
            ax.set_ylabel(r'$\Delta N$ (m; norte $\uparrow$)')
            ax.set_title(f'({panel}) Guinada de {degrees}°', pad=14)
            ax.scatter([0], [0], color='0.15', s=25, zorder=5)
            ax.text(0.07, -0.12, 'Câmera', ha='left', va='top', fontsize=9)

            # Camera components added head to tail; resultant is the object offset.
            arrow(ax, (0, 0), step, color=BLUE, lw=2.0)
            arrow(ax, step, point, color=GREEN, lw=2.0)
            arrow(ax, (0, 0), point, color=orange, lw=2.2)
            ax.scatter(*point, color=orange, s=35, zorder=5)
            ax.text(east + 0.07, north + 0.06, 'A', color=orange, weight='bold')
            if degrees == 0:
                ax.text(0.13, 0.54, 'Frente $x_c$',
                        ha='left', va='center', fontsize=10)
                ax.text(0.7, 0.87, 'Direita $y_c$', ha='center', fontsize=10)
            else:
                ax.text(0.40, 0.72, 'Frente $x_c$',
                        ha='center', fontsize=10)
                ax.text(1.16, 0.19, 'Direita $y_c$',
                        ha='center', va='center', rotation=-45, fontsize=10)
                angles = np.linspace(0, psi, 60)
                ax.plot(0.32 * np.sin(angles), 0.32 * np.cos(angles), color='0.2', lw=1)
                ax.text(0.09, 0.44, '$\\psi$', ha='center', fontsize=12)

            ax.plot([east, east], [0, north], color=orange, ls=':', lw=1.3)
            ax.plot([0, east], [north, north], color=orange, ls=':', lw=1)
        output = Path(__file__).resolve().parents[1] / 'doc/figuras/fig_guinada_deslocamentos.pdf'
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
        print(output)


if __name__ == '__main__':
    main()
