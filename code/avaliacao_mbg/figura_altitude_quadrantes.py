"""Figuras de escala da memória por quadrantes e de seu controle temporal."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

from dados import ROOT, OUT
from paleta_tcc import BLUE, ORANGE
from resumir_altitude_integral import figura, resultados_por_classe

DADOS = OUT / "quadrantes"
ALTITUDE = DADOS / "altitude"
ALTURAS = [40, 50, 60, 70, 80]
METODOS = ("quadrantes", "intervalo_pareado")
NOMES = {"quadrantes": "Quadrantes", "intervalo_pareado": "Intervalo"}
CLASSES = {"bucket": "Balde", "tire": "Pneu", "potted_plant": "Vaso de planta",
           "storm_drain": "Bueiro", "watertank": "Caixa d’água", "pool": "Piscina"}


def paineis_por_classe(dados):
    fig = plt.figure(figsize=(7.35, 4.85))
    grade = fig.add_gridspec(2, 3, left=.085, right=.988, bottom=.09, top=.855,
                             wspace=.29, hspace=.60)
    lados_a_40 = dados.query("metodo == 'quadrantes' and altura_m == 40").set_index("classe").lado_medio_px
    ordem_classes = list(CLASSES)
    for indice, (classe, titulo) in enumerate(CLASSES.items()):
        recorte = dados[dados.classe == classe]
        n, = recorte.referencias.unique()
        sub = grade[indice].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=.15)
        recall = fig.add_subplot(sub[0])
        tamanho = fig.add_subplot(sub[1], sharex=recall)
        integral = recorte[recorte.metodo == "integral"]
        assert integral.altura_m.tolist() == [40]
        percentual_integral = f"{integral.recall_pct.item():.1f}".replace(".", ",")
        recall.set_title(f"{titulo} ({n})", loc="left", fontsize=10, fontweight="semibold", pad=15)
        recall.text(0, 1.035, f"Integral a 40 m ({percentual_integral}%)", transform=recall.transAxes,
                    fontsize=8, color="#474747", va="bottom")

        seletivos = recorte[recorte.metodo.isin(METODOS)]
        ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=True).tick_values(
            seletivos.recall_pct.min(), seletivos.recall_pct.max())
        ticks = ticks[(ticks >= 0) & (ticks <= 100)]
        margem = .12 * (ticks[1] - ticks[0])
        recall.set(yticks=ticks, ylim=(ticks[0] - margem, ticks[-1] + margem), xlim=(36, 84))
        # O círculo cabe no quadrado aberto quando os dois métodos têm o mesmo recall.
        for metodo, cor, marca, linha, ms in [("intervalo_pareado", ORANGE, "s", "--", 5.8),
                                               ("quadrantes", BLUE, "o", "-", 3.8)]:
            serie = seletivos[seletivos.metodo == metodo].sort_values("altura_m")
            recall.plot(serie.altura_m, serie.recall_pct, color=cor, marker=marca, ls=linha,
                        lw=1.4, ms=ms, mfc="white" if metodo == "intervalo_pareado" else cor,
                        mew=1, clip_on=False)
        recall.tick_params(axis="x", bottom=False, labelbottom=False)
        recall.tick_params(axis="y", length=0, labelsize=8, pad=2)
        recall.grid(axis="y", color=".83", lw=.55)
        recall.spines[["left", "bottom"]].set_visible(False)
        base = lados_a_40[classe]
        lados = [base * 40 / h for h in ALTURAS]
        barras = tamanho.bar(ALTURAS, lados, width=4.8, color="#afb7bc")
        tamanho.bar_label(barras, labels=[f"{v:.0f}" for v in lados], padding=.6,
                          fontsize=8, color="#333333")
        inicio_linha = (indice // 3) * 3
        maior_lado_na_linha = lados_a_40[ordem_classes[inicio_linha:inicio_linha + 3]].max()
        tamanho.set(ylim=(0, maior_lado_na_linha * 1.48), yticks=[], xticks=ALTURAS)
        tamanho.spines["left"].set_visible(False)
        tamanho.spines["bottom"].set_color(".65")
        tamanho.tick_params(axis="x", labelsize=7.5, length=2, pad=2)
        if indice % 3 == 0:
            recall.set_ylabel("Recall (%)", fontsize=8)
            tamanho.set_ylabel("Lado (px)", fontsize=8, labelpad=3)
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.4, ms=3.8, label="Quadrantes"),
               Line2D([], [], color=ORANGE, marker="s", mfc="white", ls="--", lw=1.4,
                      ms=5.8, label="Intervalo pareado")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.005),
               ncol=2, frameon=False, fontsize=9)
    fig.supxlabel("Altura equivalente (m)", y=.014, fontsize=8.7)
    return fig


def salvar_figura(fig, nome):
    fig.savefig(ROOT / "doc/figuras" / nome, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main(somente_originais):
    if somente_originais:
        pares = pd.read_csv(DADOS / "por_trilha.csv").loc[lambda df: df.metodo.isin(METODOS)]
        integral = pd.read_csv(OUT / "altitude_integral/por_trilha.csv").query("metodo == 'integral' and altura_m == 40")
        trilhas = pd.concat([pares.assign(altura_m=40), integral], ignore_index=True)
        alturas_medidas = [40]
    else:
        trilhas = pd.read_csv(ALTITUDE / "por_trilha.csv")
        agregado = pd.read_csv(ALTITUDE / "resumo_seis_voos.csv")
        assert agregado.loc[agregado.metodo.isin(METODOS), "recortes"].eq(8796).all()
        alturas_medidas = ALTURAS

    esperado = {(m, h) for m in METODOS for h in alturas_medidas} | {("integral", 40)}
    assert set(zip(trilhas.metodo, trilhas.altura_m)) == esperado
    assert trilhas.groupby(["metodo", "altura_m"]).size().eq(1924).all()
    assert not trilhas.duplicated(["video", "classe", "trilha", "metodo", "altura_m"]).any()
    classes = resultados_por_classe(trilhas)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        grafico = paineis_por_classe(classes)
        salvar_figura(grafico, "fig_escala_classes_quadrantes.pdf")
        if not somente_originais:
            classes.to_csv(ALTITUDE / "tamanho_recall_por_classe.csv", index=False, float_format="%.6f")
            salvar_figura(figura(agregado, METODOS, NOMES), "fig_escala_quadrantes.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--somente-originais", action="store_true")
    main(ap.parse_args().somente_originais)
