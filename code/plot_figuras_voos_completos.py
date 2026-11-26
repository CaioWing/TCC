"""Figuras físicas e de processamento dos seis voos completos."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paleta_tcc import BLUE, ORANGE
from plot_verificacao_telemetria import observed_errors, paineis_fisicos
from simular_potencia_missao import (BASE_KG, FIG, JETSON, KIT_KG, OUT, TEST_MASSES,
                                    eficiencia_ajustada, mechanical)


def decimal(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")


def fisica(telemetria):
    fig = plt.figure(figsize=(7.65, 3.7))
    esquerda = fig.add_gridspec(2, 1, left=.082, right=.670, bottom=.17, top=.84,
                                hspace=.58)
    energia = fig.add_subplot(esquerda[0])
    duracao = fig.add_subplot(esquerda[1], sharex=energia)
    inclinacao = fig.add_axes([.77, .24, .215, .55])
    paineis_fisicos(telemetria, energia, duracao)
    handles, legendas = energia.get_legend_handles_labels()
    energia.get_legend().remove()
    fig.legend(handles, legendas, loc="upper left", bbox_to_anchor=(.058, .995),
               ncol=2, frameon=False, fontsize=8.3, handletextpad=.2)
    validos = telemetria.dropna(subset=["tilt", "cruise_speed"])
    campanhas = validos.groupby(validos.region.eq("A"))[["tilt", "cruise_speed"]].median()
    # Equilíbrio em cruzeiro: tan(theta) = D/(mg), com fe = 0,02 m².
    tangente_modelo = .5 * 1.225 * .02 * campanhas.cruise_speed**2 / (BASE_KG * 9.81)
    observada = np.degrees(np.arctan(campanhas.tilt))
    prevista = np.degrees(np.arctan(tangente_modelo))
    pos = np.arange(len(campanhas))
    inclinacao.vlines(pos, prevista, observada, color=".65", lw=1.4)
    inclinacao.scatter(pos, observada, color=BLUE, s=21, label="Telemetria", zorder=3)
    inclinacao.scatter(pos, prevista, facecolor="white", edgecolor=".22", marker="s",
                       s=24, label="Modelo", zorder=3)
    for x, med, mod in zip(pos, observada, prevista):
        for valor in (med, mod):
            inclinacao.annotate(decimal(valor, 1), (x, valor), xytext=(5, 0),
                                textcoords="offset points", va="center", fontsize=8)
    inclinacao.set_title("(c) Inclinação em cruzeiro", loc="left", fontsize=9, pad=15)
    inclinacao.set(xticks=pos, xticklabels=["4,1", "7,9"], xlim=(-.4, 1.65), ylim=(0, 13.4),
                   yticks=[0, 3, 6, 9, 12], xlabel="Velocidade (m/s)", ylabel="Inclinação (°)")
    inclinacao.legend(frameon=False, fontsize=7.4, loc="upper left", handlelength=1)
    inclinacao.grid(axis="y", color=".89", lw=.6)
    return fig


def jetson():
    base = OUT / "qualidade_memoria"
    fila = pd.read_csv(base / "quadrantes/fila_jetson_por_voo.csv").query("cenario == 'media'")
    totais = fila.groupby("metodo")[["recortes", "duracao_s"]].sum()
    np.testing.assert_allclose(totais.loc["quadrantes"], totais.loc["intervalo_pareado"])
    total, duracao = totais.loc["quadrantes"]
    integral = pd.read_csv(base / "altitude_integral/resumo_seis_voos.csv")
    total_integral = integral.query("metodo == 'integral' and altura_m == 40").recortes.item()
    bancada = pd.read_csv(JETSON / "summary.csv").query("config == 'trt_fp16' and mode == 'e2e'").iloc[0]
    exemplo = pd.read_csv(base / "quadrantes/fila_jetson_inicio_voo09.csv")

    fig = plt.figure(figsize=(7.45, 2.1))
    a = fig.add_axes([.157, .245, .245, .56])
    b = fig.add_axes([.571, .245, .414, .56])
    fig.text(.005, .98, "(a) Demanda e vazão de bancada", va="top", size=9.3)
    fig.text(.464, .98, "(b) Ocupação do detector no início do voo 09", va="top", size=9.3)
    a.set_xscale("log")
    for y, valor, cor in [(2, total / duracao, BLUE), (1, bancada.fps, ".3"),
                          (0, total_integral / duracao, BLUE)]:
        a.scatter(valor, y, color=cor, s=25, zorder=3)
        a.annotate(decimal(valor), (valor, y), xytext=(0, 5), textcoords="offset points",
                   ha="center", va="bottom", fontsize=7.8,
                   bbox={"facecolor":"white", "edgecolor":"none", "pad":.25})
    a.axvline(bancada.fps, lw=.8, ls=":", color=".4", zorder=0)
    a.set_yticks([2, 1, 0], ["Quadrantes e intervalo", "Bancada FP16", "Integral"], fontsize=7.6)
    a.set(xlim=(3, 1500), ylim=(-.45, 2.7), xlabel="Recortes/s (escala logarítmica)")
    a.set_xticks([5, 10, 100, 1000], ["5", "10", "100", "1.000"])
    a.tick_params(axis="x", which="minor", bottom=False)
    a.tick_params(axis="y", length=0)
    a.grid(axis="x", alpha=.15)
    for y, (metodo, cor) in enumerate([("intervalo_pareado", ORANGE), ("quadrantes", BLUE)]):
        lotes = exemplo[exemplo.metodo == metodo]
        duracoes = (lotes.fim_ms - lotes.inicio_ms) / 1000
        b.barh(y, 6, height=.43, color="#ededed")
        b.barh(np.full(len(lotes), y), duracoes, left=lotes.inicio_ms / 1000, height=.43,
               color=cor, linewidth=0)
    b.set(xlim=(0, 6), xticks=range(7), xlabel="Tempo desde o início do voo (s)", ylim=(-.52, 1.57))
    b.set_yticks([0, 1], ["Intervalo", "Quadrantes"], fontsize=7.6)
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.grid(axis="x", alpha=.15)
    return fig


def massa():
    desempenho = pd.read_csv(OUT / "curva_potencia_por_voo.csv")
    eta = eficiencia_ajustada(desempenho)
    velocidade = np.linspace(0, 14, 281)
    potencia_base = mechanical(velocidade, BASE_KG)/eta
    potencia_kit = mechanical(velocidade, BASE_KG+KIT_KG)/eta
    fig, a = plt.subplots(figsize=(4.95, 2.22), layout="constrained")
    a.fill_between(velocidade, mechanical(velocidade, BASE_KG+min(TEST_MASSES))/eta,
                   mechanical(velocidade, BASE_KG+max(TEST_MASSES))/eta,
                   color=BLUE,alpha=.16,label="Kit de 100 a 250 g")
    a.plot(velocidade,potencia_kit,c=BLUE,label="Kit de 174 g",lw=1.55)
    a.plot(velocidade,potencia_base,c=".3",ls="--",label="Sem kit",lw=1.2)
    a.set(xlim=(0,14),xticks=range(0,15,2),ylabel="Potência elétrica (W)",xlabel="Velocidade (m/s)")
    a.legend(frameon=False,fontsize=8,loc="upper right")
    a.grid(axis="y",alpha=.18)
    return fig


def main():
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9.3,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.axisbelow":True,
                         "pdf.fonttype":42}):
        for nome, figura in [("fig_jetson_quadrantes_intervalo",jetson()),
                             ("fig_massa_kit",massa()),
                             ("fig_telemetria_fisica",fisica(observed_errors()))]:
            figura.savefig(FIG / f"{nome}.pdf",bbox_inches="tight",pad_inches=.055)
            plt.close(figura)


if __name__ == "__main__":
    main()
