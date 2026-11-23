"""Figuras físicas e de processamento para a comparação entre cobertura e intervalo."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paleta_tcc import BLUE
from plot_demanda_jetson import painel_fila
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
    inclinacao = fig.add_axes([.78, .35, .21, .49])
    paineis_fisicos(telemetria, energia, duracao, inclinacao)
    handles, legendas = energia.get_legend_handles_labels()
    energia.get_legend().remove()
    fig.legend(handles, legendas, loc="upper left", bbox_to_anchor=(.058, .995),
               ncol=2, frameon=False, fontsize=8.3, handletextpad=.2)
    inclinacao.set_title("(c) Inclinação entre\nas campanhas", loc="left", fontsize=9)
    inclinacao.set_xticks([0,1,2], ["Observada", "Com V", "Com V²"], fontsize=7.4)
    inclinacao.set_ylabel("Razão entre campanhas")
    return fig


def jetson(taxa, bancada):
    fig = plt.figure(figsize=(7.5, 2.03))
    a = fig.add_axes([.14, .26, .232, .46])
    b = fig.add_axes([.565, .26, .425, .46])
    fig.text(.008, .98, "(a) Demanda e vazão de bancada", va="top", size=9.3)
    fig.text(.407, .98, "(b) Fila em um bloco regular de 60 quadros", va="top", size=9.3)
    fig.text(.990, .82, "Processamento em cor · folga em cinza", ha="right", fontsize=7.4)
    valores = [taxa, bancada.fps]
    barras = a.barh([1, 0], valores, color=[BLUE, ".35"], height=.51)
    a.bar_label(barras, labels=[decimal(x) for x in valores], label_type="center",
                color="white", fontsize=7.8, weight="semibold")
    a.set_yticks([1, 0], ["Por estratégia", "Bancada FP16"])
    a.set(xlim=(0, 50), ylim=(-.6, 1.6), xticks=[0, 15, 30, 45], xlabel="Recortes por segundo")
    a.tick_params(axis="y", length=0)
    a.grid(axis="x", alpha=.15)
    painel_fila(b, OUT/"qualidade_memoria")
    b.set_yticklabels(["Cobertura\n16 recortes", "Intervalo\n24 recortes"], fontsize=7.5)
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
    voos = pd.read_csv(OUT / "qualidade_memoria/cobertura_continua/por_voo.csv")
    totais = voos.groupby("metodo")[["recortes","duracao_s"]].sum()
    assert totais.loc["cobertura"].equals(totais.loc["intervalo"])
    taxa = totais.loc["cobertura","recortes"]/totais.loc["cobertura","duracao_s"]
    bancada = pd.read_csv(JETSON / "summary.csv").query("config == 'trt_fp16' and mode == 'e2e'").iloc[0]
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9.3,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.axisbelow":True,
                         "pdf.fonttype":42}):
        for nome, figura in [("fig_jetson_cobertura_intervalo",jetson(taxa,bancada)),
                             ("fig_massa_kit",massa()),
                             ("fig_telemetria_fisica",fisica(observed_errors()))]:
            figura.savefig(FIG / f"{nome}.pdf",bbox_inches="tight",pad_inches=.055)
            plt.close(figura)


if __name__ == "__main__":
    main()
