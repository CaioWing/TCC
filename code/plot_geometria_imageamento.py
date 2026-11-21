"""Geometria e área nova por segundo para o exemplo de imageamento do TCC."""

from pathlib import Path
from paleta_tcc import BLUE, ORANGE, GREEN

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def main():
    h, gsd, speed = 40, 0.014, 4
    length, width = 2160 * gsd, 3840 * gsd
    tile = 640 * gsd
    orange = ORANGE
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
        fig, (side, top) = plt.subplots(1, 2, figsize=(10, 4.4))
        fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.13, wspace=0.55)
        side.set_title('Vista lateral (direção do voo)', pad=25)
        side.set_xlim(-32, 32)
        side.set_ylim(-7, 44)
        side.axis('off')
        side.add_patch(Rectangle((-2.5, h-1), 5, 2, color='0.12'))
        side.plot([-length/2, 0, length/2], [0, h, 0], color='0.55', lw=1.4)
        side.plot([-length/2, length/2], [0, 0], color='0.12', lw=1.8)
        side.plot([length/2-speed, length/2], [0, 0], color=orange, lw=4)
        side.annotate('', xy=(9, h), xytext=(3, h),
                      arrowprops={'arrowstyle': '-|>', 'color': orange, 'lw': 2})
        side.text(10, h, '$V$', color=orange, va='center', fontsize=13)
        side.plot([-23, 0], [h, h], color='0.55', ls=':', lw=0.7)
        side.annotate('', xy=(-21, h), xytext=(-21, 0),
                      arrowprops={'arrowstyle': '<->', 'color': 'black'})
        side.text(-22.5, h/2, '$h$', ha='right', va='center')
        side.annotate('', xy=(length/2, -3), xytext=(-length/2, -3),
                      arrowprops={'arrowstyle': '<->', 'color': 'black'})
        side.text(0, -5.4, '$L$', ha='center', va='center')
        side.text(length/2, 4, 'faixa nova\nem 1 s', color=orange, fontsize=10)

        top.set_title('Vista superior (região imageada)', pad=25)
        top.set_aspect('equal')
        top.set_xlim(-2, length+5)
        top.set_ylim(-8, width+3)
        top.axis('off')
        top.add_patch(Rectangle((length-speed, 0), speed, width,
                                facecolor=orange, alpha=0.32, edgecolor='none'))
        top.add_patch(Rectangle((0, 0), length, width, fill=False, edgecolor='0.12', lw=1.8))
        for x in np.arange(tile, length, tile):
            top.plot([x, x], [0, width], color='0.55', lw=0.7)
        for y in np.arange(tile, width, tile):
            top.plot([0, length], [y, y], color='0.55', lw=0.7)
        top.annotate('', xy=(length+4.5, width/2), xytext=(length+1, width/2),
                     arrowprops={'arrowstyle': '-|>', 'color': orange, 'lw': 2})
        top.text(length/2, -4, 'Faixa nova', ha='center', color=orange, fontsize=10)
        output = Path(__file__).resolve().parents[1] / 'doc/figuras/fig_geometria_imageamento.pdf'
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
        print(output)


if __name__ == '__main__':
    main()
