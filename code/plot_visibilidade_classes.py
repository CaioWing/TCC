"""Compara a duração das trilhas com L/V nos mesmos voos, por classe e campanha."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paleta_tcc import BLUE, ORANGE
from plot_verificacao_telemetria import DATA, ROOT, flight_geometry, visible_tracks


NOMES = {
    "watertank": "Caixa-d’água", "potted_plant": "Vaso de planta",
    "storm_drain": "Bueiro", "pool": "Piscina", "tire": "Pneu", "bucket": "Balde",
    "plastic_bag": "Saco plástico", "bottle": "Garrafa", "small_trash_bin": "Lixeira pequena",
    "puddle": "Poça", "large_trash_bin": "Lixeira grande", "dumpster": "Caçamba",
}
PAINEIS = {"A": "(a) Região A · 7,9 m/s", "B–M": "(b) Regiões B a M · 4,1 m/s"}
RESUMO = ROOT / "doc/dados/visibilidade_por_classe.csv"
FIGURA = ROOT / "doc/figuras/fig_visibilidade_por_classe.pdf"


def ler_trilhas():
    voos = pd.read_csv(ROOT / "doc/dados/balanco_telemetria.csv")
    linhas = []
    for voo in voos.itertuples():
        pasta = DATA / "raw" / f"flight-mbg-v2-{voo.voo:02d}-region{voo.regiao}"
        log, = (pasta / "log").glob("*.csv")
        anotacoes, = (pasta / "ann").glob("*.xml")
        sinais = pd.read_csv(log, usecols=["remainPowerPercent", "speed(mps)"])
        _, comprimento, velocidade, fps = flight_geometry(pasta, sinais)
        campanha = "A" if voo.regiao == "A" else "B–M"
        for trilha, classe, duracao in visible_tracks(anotacoes, fps):
            linhas.append((campanha, voo.voo, trilha, classe, duracao, comprimento / velocidade))
    return pd.DataFrame(linhas, columns=["campanha", "voo", "trilha", "classe", "anotado", "previsto"])


def resumir(trilhas):
    linhas = []
    for campanha in PAINEIS:
        for classe in NOMES:
            dados = trilhas[(trilhas.campanha == campanha) & (trilhas.classe == classe)]
            linha = dict(campanha=campanha, classe=classe, trilhas=len(dados), voos=dados.voo.nunique())
            for medida in ("anotado", "previsto"):
                for percentil in (10, 50, 90):
                    linha[f"{medida}_p{percentil}_s"] = dados[medida].quantile(percentil / 100)
            linhas.append(linha)
    return pd.DataFrame(linhas)


def desenhar(resumo):
    fig, eixos = plt.subplots(1, 2, figsize=(7.2, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.175, right=.94, bottom=.20, top=.91, wspace=.28)
    estilos = [("anotado", BLUE, "o", "-", -.15), ("previsto", ORANGE, "D", "--", .15)]
    for eixo, (campanha, titulo) in zip(eixos, PAINEIS.items(), strict=True):
        painel = resumo[resumo.campanha == campanha].set_index("classe").loc[list(NOMES)]
        for i, linha in enumerate(painel.itertuples()):
            eixo.text(1.025, i, f"{linha.trilhas:,}".replace(",", "."), transform=eixo.get_yaxis_transform(),
                      va="center", ha="left", fontsize=7.6)
            if not linha.trilhas:
                continue
            for medida, cor, marca, estilo, dy in estilos:
                p10, p50, p90 = (getattr(linha, f"{medida}_p{p}_s") for p in (10, 50, 90))
                eixo.plot([p10, p90], [i+dy, i+dy], color=cor, ls=estilo, lw=1.25)
                eixo.plot(p50, i+dy, marker=marca, ms=4.2, color=cor,
                          mfc=cor if medida == "anotado" else "white", mew=1.15)
        eixo.set_title(titulo, loc="left", fontsize=10.3)
        eixo.text(1.025, 1.015, "n", fontstyle="italic", transform=eixo.transAxes, fontsize=8)
        eixo.set_axisbelow(True)
        eixo.grid(axis="x", color=".87", lw=.6)
        for y in np.arange(.5, len(NOMES)-1):
            eixo.axhline(y, lw=.45, color=".92", zorder=0)
        eixo.tick_params(axis="y", length=0)
    maximo = resumo[["anotado_p90_s", "previsto_p90_s"]].max().max()
    eixos[0].set(xlim=(0, np.ceil(maximo)), ylim=(len(NOMES)-.5, -.5),
                 yticks=np.arange(len(NOMES)), yticklabels=list(NOMES.values()))
    eixos[0].xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    fig.supxlabel("Tempo de visibilidade (s)", x=.56, y=.105, fontsize=9.5)
    legendas = ["Duração nas anotações", "Previsão do voo (L/V)"]
    chaves = [Line2D([], [], color=cor, marker=marca, ls=estilo, lw=1.3, ms=4.2,
                     mfc=cor if medida == "anotado" else "white", label=legenda)
              for (medida, cor, marca, estilo, _), legenda in zip(estilos, legendas, strict=True)]
    fig.legend(handles=chaves, loc="lower center", bbox_to_anchor=(.56, .005), ncol=2, frameon=False)
    fig.savefig(FIGURA, bbox_inches="tight", pad_inches=.04, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    trilhas = ler_trilhas()
    resumo = resumir(trilhas)
    if trilhas.duplicated(["voo", "trilha"]).any() or resumo.trilhas.sum() != len(trilhas):
        raise ValueError("Identificadores de trilhas duplicados ou classe sem correspondência.")
    resumo.to_csv(RESUMO, index=False, float_format="%.6f")
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False}):
        desenhar(resumo)
