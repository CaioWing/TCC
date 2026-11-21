"""Deslocamentos em pixels e metros, com escalas iguais nos eixos de cada painel."""
from pathlib import Path
from paleta_tcc import BLUE, ORANGE, GREEN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    orange, blue = ORANGE, BLUE
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
        fig, (im, ground) = plt.subplots(1, 2, figsize=(10, 3.8))
        fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.19, wspace=0.40)
        for ax in (im, ground):
            ax.set_aspect('equal')
            ax.spines[['right', 'top']].set_visible(False)
            ax.grid(color='0.90', lw=0.6)
            ax.set_axisbelow(True)
            ax.axhline(0, color='0.55', lw=0.8)
            ax.axvline(0, color='0.55', lw=0.8)
            ax.scatter([0], [0], s=20, color='0.15', zorder=4)
        im.set_title('(a) Deslocamentos na imagem', pad=14)
        im.set_xlim(-150, 150)
        im.set_ylim(110, -110)
        im.set_xticks([-150, -100, -50, 0, 50, 100, 150])
        im.set_yticks([-100, -50, 0, 50, 100])
        im.set_xlabel(r'$u-c_x$ (pixels; direita $\rightarrow$)', labelpad=10)
        im.set_ylabel(r'$v-c_y$ (pixels; baixo $\downarrow$)')
        im.text(-5, -8, 'Ponto principal', ha='right', fontsize=9)
        ground.set_title('(b) Deslocamentos no solo', pad=14)
        ground.set_xlim(-2.1, 2.1)
        ground.set_ylim(-1.54, 1.54)
        ground.set_xticks([-2, -1, 0, 1, 2])
        ground.set_yticks([-1.4, -0.7, 0, 0.7, 1.4])
        ground.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'.replace('.', ',')))
        ground.set_xlabel(r'$y_c$ (m; direita $\rightarrow$)', labelpad=10)
        ground.set_ylabel(r'$x_c$ (m; frente $\uparrow$)')
        ground.text(-0.07, 0.12, 'Projeção da câmera', ha='right', fontsize=9)
        for ax, scale in [(im, 1), (ground, 0.014)]:
            for name, du, dv, color, marker in [('A', 100, -50, orange, 'o'), ('B', -100, 50, blue, 's')]:
                x = du * scale
                y = dv if ax is im else -dv * scale
                ax.plot([x, x, 0], [0, y, y], color=color, ls=':', lw=1.2)
                ax.annotate('', xy=(x, y), xytext=(0, 0), arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.8})
                ax.scatter([x], [y], s=38, color=color, marker=marker, zorder=5)
                ax.annotate(name, (x, y), xytext=(6, 7), textcoords='offset points', color=color, weight='bold')
        output = Path(__file__).resolve().parents[1] / 'doc/figuras/fig_deslocamentos_imagem.pdf'
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
        print(output)


if __name__ == '__main__':
    main()
