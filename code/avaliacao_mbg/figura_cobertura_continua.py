"""Resultados pareados da seleção por cobertura e da cadência fixa nos seis voos."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from paleta_tcc import BLUE, ORANGE

DADOS = ROOT / "doc/dados/qualidade_memoria/cobertura_continua"


def diferencas_por_voo(ax, delta, rotulos, titulo, unidade, limites, ticks, decimais):
    y = np.arange(len(delta))
    ax.axvline(0, lw=.8, color=".45")
    ax.hlines(y, 0, delta, color=BLUE, lw=1.6)
    ax.scatter(delta, y, color=BLUE, s=16, zorder=3)
    for pos, valor in zip(y, delta):
        texto = f"{valor:+.{decimais}f}".replace(".", ",").replace("-", "−")
        ax.annotate(texto, (valor, pos), xytext=(-4 if valor < 0 else 4, 0),
                    textcoords="offset points", ha="right" if valor < 0 else "left",
                    va="center", fontsize=7.2)
    ax.set(yticks=y, yticklabels=rotulos, xlabel=unidade, xlim=limites,
           ylim=(len(delta) - .4, -.65), xticks=ticks)
    ax.set_title(titulo, loc="left", pad=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.18)


def figura():
    voos = pd.read_csv(DADOS / "por_voo.csv")
    trilhas = pd.read_csv(DADOS / "por_trilha.csv")
    assert not trilhas.duplicated(["video", "trilha", "classe", "metodo"]).any()
    geo = voos.query("metodo == 'cobertura'").set_index("video").sort_index()
    intervalo = voos.query("metodo == 'intervalo'").set_index("video").sort_index()
    pd.testing.assert_frame_equal(geo[["recortes", "referencias"]], intervalo[["recortes", "referencias"]])
    rotulos = [v.removeprefix("video").replace("_region", " · ") for v in geo.index]
    total = int(geo.referencias.sum())
    t_final = trilhas.atraso_s.max()

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.45, 2.43),
                                     gridspec_kw={"width_ratios": [1.04, .90, 1.87]})
        fig.subplots_adjust(left=.065, right=.994, top=.78, bottom=.23, wspace=.28)
        diferencas_por_voo(a, 100*(geo.recall-intervalo.recall), rotulos,
                           "(a) Recall geográfico", "Diferença (p.p.)", (-8.6, 2.4), [-6,-3,0], 1)
        diferencas_por_voo(b, geo.fp-intervalo.fp, [], "(b) Falsos alarmes",
                           "Diferença (contagem)", (-96, 6), [-60,-30,0], 0)
        # Os dois painéis usam as mesmas linhas e a mesma ordem de voos.
        b.tick_params(labelleft=False)
        for metodo, nome, cor, estilo in [("cobertura", "Memória por cobertura", BLUE, "-"),
                                          ("intervalo", "Um quadro a cada 30", ORANGE, "--")]:
            grupo = trilhas[trilhas.metodo == metodo]
            assert len(grupo) == total
            tempos, contagens = np.unique(grupo.atraso_s.dropna(), return_counts=True)
            x = np.r_[0, tempos, t_final]
            y = np.r_[0, 100*contagens.cumsum()/total, 100*contagens.sum()/total]
            c.step(x, y, where="post", label=nome, color=cor, ls=estilo, lw=1.5)
            ate_um_segundo = 100 * grupo.atraso_s.le(1).sum() / total
            c.scatter([1], [ate_um_segundo], s=20, facecolor="white",
                      edgecolor=cor, linewidth=1.1, zorder=4)
            deslocamento = (9,-10) if metodo == "cobertura" else (-7,10)
            c.annotate(f"{ate_um_segundo:.1f}%".replace(".", ","), (1, ate_um_segundo),
                       xytext=deslocamento, textcoords="offset points",
                       ha="left" if metodo == "cobertura" else "right",
                       fontsize=7.2, color=cor, weight="semibold",
                       bbox=dict(facecolor="white", edgecolor="none", pad=.5))
        c.axvline(1, lw=.65, color=".5", ls=":", zorder=1)
        c.set_title("(c) Primeiro alerta visual", loc="left", pad=7)
        c.set_xscale("symlog", linthresh=2., base=2)
        c.set(xlim=(0,t_final*1.03), ylim=(0,80), xlabel="Tempo desde a primeira aparição (s)",
              ylabel="Trilhas confirmadas (%)", yticks=[0,20,40,60,80])
        c.set_xticks([0,1,2,4,8], ["0","1","2","4","8"])
        c.grid(axis="y", alpha=.2)
        c.legend(loc="lower right", frameon=False, fontsize=7.2)
        for ax in (a,b,c):
            ax.tick_params(axis="both", labelsize=7.5)
            ax.set_axisbelow(True)
        return fig


if __name__ == "__main__":
    grafico = figura()
    grafico.savefig(ROOT / "doc/figuras/fig_cobertura_continua.pdf", bbox_inches="tight", pad_inches=.05)
    plt.close(grafico)
