"""Resultados pareados da memória por quadrantes e do intervalo de mesma cota nos seis voos."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from paleta_tcc import BLUE, GRAY, ORANGE

DADOS = ROOT / "doc/dados/qualidade_memoria/quadrantes"


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
    pareados = pd.read_csv(DADOS / "por_trilha.csv").query("metodo in ['quadrantes', 'intervalo_pareado']")
    integral = pd.read_csv(DADOS.parent / "altitude_integral/por_trilha.csv").query("metodo == 'integral' and altura_m == 40")
    trilhas = pd.concat([pareados, integral], ignore_index=True)
    assert set(trilhas.metodo) == {"quadrantes", "intervalo_pareado", "integral"}
    assert not trilhas.duplicated(["video", "trilha", "classe", "metodo"]).any()
    geo = voos.query("metodo == 'quadrantes'").set_index("video").sort_index()
    intervalo = voos.query("metodo == 'intervalo_pareado'").set_index("video").sort_index()
    pd.testing.assert_frame_equal(geo[["recortes", "referencias"]], intervalo[["recortes", "referencias"]])
    rotulos = [v.removeprefix("video").replace("_region", " · ") for v in geo.index]
    total = int(geo.referencias.sum())
    t_final = trilhas.atraso_s.max()
    quadros = trilhas.pivot(index=["video", "classe", "trilha"], columns="metodo",
                           values="primeiro_alerta_quadro")
    assert len(quadros) == total
    comuns = quadros[["quadrantes", "intervalo_pareado"]].dropna()
    diferenca = comuns.quadrantes - comuns.intervalo_pareado
    contagens = [diferenca.lt(0).sum(), diferenca.eq(0).sum(), diferenca.gt(0).sum()]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.25, 4.05), layout="constrained")
        diferencas_por_voo(a, 100*(geo.recall-intervalo.recall), rotulos,
                           "(a) Recall geográfico", "Diferença (p.p.)", (-.4, 5.7), [0,2,4], 1)
        diferencas_por_voo(b, geo.fp-intervalo.fp, [], "(b) Falsos alarmes",
                           "Diferença (contagem)", (-207, 8), [-150,-100,-50,0], 0)
        # Os dois painéis usam as mesmas linhas e a mesma ordem de voos.
        b.tick_params(labelleft=False)
        for metodo, nome, cor, estilo in [("integral", "Integral", ".24", "-."),
                                          ("intervalo_pareado", "Intervalo", ORANGE, "--"),
                                          ("quadrantes", "Quadrantes", BLUE, "-")]:
            grupo = trilhas[trilhas.metodo == metodo]
            assert len(grupo) == total
            tempos, alertas_por_tempo = np.unique(grupo.atraso_s.dropna(), return_counts=True)
            x = np.r_[0, tempos, t_final]
            y = np.r_[0, 100*alertas_por_tempo.cumsum()/total, 100*alertas_por_tempo.sum()/total]
            n_um_segundo = grupo.atraso_s.le(1).sum()
            ate_um_segundo = 100 * n_um_segundo / total
            percentual = f"{ate_um_segundo:.1f}".replace(".", ",")
            quantidade = f"{n_um_segundo:,}".replace(",", ".")
            rotulo = f"{nome} · {percentual}% ({quantidade})"
            c.step(x, y, where="post", label=rotulo, color=cor, ls=estilo, lw=1.5)
            c.scatter([1], [ate_um_segundo], s=14, facecolor="white",
                      edgecolor=cor, linewidth=1.1, zorder=4)
        c.axvline(1, lw=.65, color=".5", ls=":", zorder=1)
        c.set_title("(c) Primeira detecção correta no vídeo", loc="left", pad=7)
        c.set_xscale("symlog", linthresh=2., base=2)
        c.set(xlim=(0,t_final*1.03), ylim=(0,80), xlabel="Tempo desde a primeira aparição (s)",
              ylabel="Trilhas confirmadas (%)", yticks=[0,20,40,60,80])
        c.set_xticks([0,1,2,4,8,16], ["0","1","2","4","8","16"])
        c.grid(axis="y", alpha=.2)
        c.legend(title="Em até 1 s", title_fontsize=7.5, loc="lower right", frameon=False, fontsize=7.2)
        d.set_title("(d) Qual estratégia detectou primeiro?", loc="left", pad=7)
        barras = d.barh([2, 1, 0], contagens, color=[BLUE, GRAY, ORANGE], height=.48)
        d.bar_label(barras, labels=[str(n) for n in contagens], padding=4, fontsize=8)
        d.set(yticks=[2, 1, 0], yticklabels=["Quadrantes", "Mesmo quadro", "Intervalo"],
              xlabel="Trilhas confirmadas por ambos", xlim=(0, 1.16 * max(contagens)))
        d.tick_params(axis="y", length=0)
        d.grid(axis="x", alpha=.18)
        for ax in (a,b,c,d):
            ax.tick_params(axis="both", labelsize=7.5)
            ax.set_axisbelow(True)
        return fig


def figura_alertas_por_movimento():
    dados = pd.read_csv(DADOS / "alertas_por_movimento.csv")
    totais = dados.loc[dados.video.eq("todos")].set_index("regime")
    colunas = ["ambas", "somente_quadrantes", "somente_intervalo", "nenhuma"]
    rotulos = ["Ambas as estratégias", "Só quadrantes",
               "Só inferência por intervalo", "Nenhuma das duas"]
    grupos = [("maior_variacao", "Maior variação"), ("menor_variacao", "Menor variação")]
    y = np.arange(len(colunas))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, eixos = plt.subplots(1, 2, sharey=True, figsize=(7.45, 2.40), layout="constrained")
        for ax, (regime, titulo) in zip(eixos, grupos):
            contagens = totais.loc[regime, colunas].astype(int)
            n = int(totais.loc[regime, "trilhas"])
            assert contagens.sum() == n
            percentuais = 100 * contagens / n
            barras = ax.barh(y, percentuais, height=.46, color=["#4e555b", BLUE, ORANGE, GRAY])
            for barra, contagem, percentual in zip(barras, contagens, percentuais):
                porcentagem = f"{percentual:.1f}".replace(".", ",")
                quantidade = f"{contagem:,}".replace(",", ".")
                ax.annotate(f"{porcentagem}% ({quantidade})",
                            (percentual, barra.get_y() + barra.get_height()/2),
                            xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.2)
            tamanho = f"{n:,}".replace(",", ".")
            ax.set_title(f"{titulo} ({tamanho} trilhas)", fontsize=9.8, loc="left")
            ax.set(xlim=(0, 100), xticks=[0, 25, 50, 75, 100], yticks=y, yticklabels=rotulos,
                   ylim=(len(y)-.5, -.5))
            ax.tick_params(axis="y", length=0, pad=5)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color=".90", lw=.55)
        fig.supxlabel("Trilhas do grupo (%)", fontsize=9)
        return fig


if __name__ == "__main__":
    grafico = figura()
    grafico.savefig(ROOT / "doc/figuras/fig_resultados_quadrantes.pdf", bbox_inches="tight", pad_inches=.05)
    plt.close(grafico)
    grafico = figura_alertas_por_movimento()
    grafico.savefig(ROOT / "doc/figuras/fig_alertas_movimento_quadrantes.pdf", bbox_inches="tight", pad_inches=.05)
    plt.close(grafico)
