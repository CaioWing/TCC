"""Gera as figuras de seleção e sensibilidade a partir dos CSV do cruzeiro."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
import numpy as np
import pandas as pd

from paleta_tcc import BLUE, ORANGE


ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "doc/dados"
FIGURAS = ROOT / "doc/figuras"
CRITERIOS = {
    "horizontal_ms": "Desvio horizontal\nmáximo (m/s)",
    "duracao_s": "Duração da janela (s)",
    "lacuna_s": "Lacuna máxima (s)",
    "vertical_ms": "Velocidade vertical\nem módulo (m/s)",
    "campanha_ms": "Tolerância na velocidade\nde cruzeiro (m/s)",
    "altitude_m": "Tolerância na altitude (m)",
}


def decimal(x, _=None):
    return f"{x:g}".replace(".", ",")


def painel_voos(eixos, voos):
    for eixo in eixos:
        eixo.set_xlim(voos[0]-.55, voos[-1]+.55)
        eixo.set_xticks(voos)
        eixo.yaxis.set_major_formatter(FuncFormatter(decimal))
        eixo.tick_params(axis="x", labelsize=8, length=3, pad=3)
        eixo.grid(axis="y", color=".91", lw=.55)
        eixo.axvline(2.5, color=".72", lw=.7, ls=":", zorder=0)
    eixos[-1].set_xlabel("Número do voo (1 e 2 na região A)")


def caixas(eixo, janelas, coluna, voos):
    """Quartis e bigodes de Tukey nas unidades de origem. Cada extremo é desenhado."""
    valores = [grupo[coluna].to_numpy() for _, grupo in janelas.groupby("voo", sort=True)]
    estatisticas = boxplot_stats(valores, whis=1.5, autorange=False)
    eixo.bxp(estatisticas, positions=voos, widths=.56, showfliers=False, manage_ticks=False,
             patch_artist=True, boxprops={"facecolor": "#e0e9ef", "edgecolor": BLUE, "lw": .82},
             whiskerprops={"color": BLUE, "lw": .75}, capprops={"color": BLUE, "lw": .75},
             medianprops={"color": ".12", "lw": 1.1})
    # Deslocar apenas em x torna visíveis valores coincidentes em sensores quantizados.
    rng = np.random.default_rng(26)
    for voo, resumo in zip(voos, estatisticas):
        pontos = resumo["fliers"]
        if len(pontos):
            dx = rng.uniform(-.17, .17, len(pontos)) if len(pontos) > 1 else 0
            eixo.scatter(voo+dx, pontos, marker="o", s=4.8, facecolor=BLUE, edgecolor="none",
                         alpha=.62, zorder=2.5)


def limite_horizontal(eixo, limite):
    eixo.axhline(limite, color=ORANGE, lw=1.05, ls="--", zorder=1.6)


def escala_mista(eixo, linear_ate, marcas, maximo):
    # A caixa é calculada antes da transformação do eixo; a origem é representável.
    eixo.set_yscale("symlog", linthresh=linear_ate, linscale=.75)
    eixo.set_yticks(marcas)
    eixo.yaxis.set_minor_locator(NullLocator())
    eixo.set_ylim(-.06*linear_ate, maximo*1.18)


def salvar(fig, nome):
    fig.savefig(FIGURAS / nome, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def desenhar_selecao(janelas, referencia, voos):
    fig, (velocidade_media, horizontal, vertical) = plt.subplots(3, 1, sharex=True, figsize=(7.2, 5.35))
    fig.subplots_adjust(left=.102, right=.983, top=.945, bottom=.097, hspace=.51)
    caixas(velocidade_media, janelas, "velocidade_ms", voos)
    tolerancia = referencia.campanha_ms
    for esquerda, direita, nominal in ((.45, 2.5, 7.9), (2.5, 26.55, 4.1)):
        velocidade_media.fill_between([esquerda, direita], nominal-tolerancia, nominal+tolerancia,
                             color=ORANGE, alpha=.16, lw=0, zorder=0)
    velocidade_media.set_title("(a) Média do módulo da velocidade horizontal nas janelas de 5 s", loc="left")
    velocidade_media.set_ylabel("Módulo (m/s)")
    velocidade_media.set_yticks([0, 2, 4, 6, 8])
    velocidade_media.set_ylim(-.25, max(8.5, janelas.velocidade_ms.max())+.35)
    velocidade_media.text(.99, .94, f"Faixas de 7,9 ± {decimal(tolerancia)} m/s (A) e 4,1 ± "
                 f"{decimal(tolerancia)} m/s (B–M)", ha="right", va="top", fontsize=8,
                 transform=velocidade_media.transAxes)
    caixas(horizontal, janelas, "desvio_horizontal_ms", voos)
    limite_horizontal(horizontal, referencia.horizontal_ms)
    horizontal.set_title(f"(b) Desvio máximo do vetor de velocidade horizontal "
                         f"(limite de {decimal(referencia.horizontal_ms)} m/s)", loc="left")
    horizontal.set_ylabel("Desvio (m/s)")
    escala_mista(horizontal, .3, [0, .15, .3, .5, 1, 2, 6], janelas.desvio_horizontal_ms.max())
    caixas(vertical, janelas, "vertical_max_ms", voos)
    limite_horizontal(vertical, referencia.vertical_ms)
    vertical.set_title(f"(c) Maior módulo da velocidade vertical (limite de {decimal(referencia.vertical_ms)} m/s)",
                       loc="left")
    vertical.set_ylabel("Módulo (m/s)")
    vertical.set_yticks([0, .1, .2, .3])
    vertical.set_ylim(-.015, max(janelas.vertical_max_ms.max(), referencia.vertical_ms)*1.15)
    painel_voos((velocidade_media, horizontal, vertical), voos)
    salvar(fig, "fig_selecao_cruzeiro.pdf")


def desenhar_altitude_lacuna(janelas, referencia, voos):
    fig, (alt, lacuna) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 4.0))
    fig.subplots_adjust(left=.102, right=.983, top=.927, bottom=.13, hspace=.48)
    caixas(alt, janelas, "desvio_altitude_m", voos)
    limite_horizontal(alt, referencia.altitude_m)
    alt.set_title(f"(a) Desvio máximo da altitude em relação a 40 m "
                  f"(limite de {decimal(referencia.altitude_m)} m)", loc="left")
    alt.set_ylabel("Desvio (m)")
    escala_mista(alt, .2, [0, .1, .2, .5, 1, 2], max(janelas.desvio_altitude_m.max(), referencia.altitude_m))
    caixas(lacuna, janelas, "lacuna_s", voos)
    limite_horizontal(lacuna, referencia.lacuna_s)
    lacuna.set_title(f"(b) Maior intervalo entre registros consecutivos "
                     f"(limite de {decimal(referencia.lacuna_s)} s)", loc="left")
    lacuna.set_ylabel("Intervalo (s)")
    escala_mista(lacuna, .4, [0, .2, .4, 1, 3, 10], max(janelas.lacuna_s.max(), referencia.lacuna_s))
    painel_voos((alt, lacuna), voos)
    salvar(fig, "fig_altitude_lacuna_cruzeiro.pdf")


def desenhar_potencia(janelas, referencia, alternativas, voos):
    aceitas = janelas.loc[janelas.aceita.eq(1)]
    central = aceitas.groupby("voo").potencia_W.median().loc[voos]
    fig, (potencia, diferenca) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 4.3),
                                             gridspec_kw={"height_ratios": [1.14, 1]})
    fig.subplots_adjust(left=.102, right=.983, top=.934, bottom=.205, hspace=.50)
    caixas(potencia, aceitas, "potencia_W", voos)
    potencia.set_title(f"(a) Potência nas janelas aceitas com limite horizontal de "
                       f"{decimal(referencia.horizontal_ms)} m/s", loc="left")
    potencia.set_ylabel("Potência média (W)")
    potencia.yaxis.set_major_locator(MaxNLocator(nbins=5))
    diferenca.set_title("(b) Deslocamento da mediana ao alterar o limite horizontal", loc="left")
    diferenca.set_ylabel("Diferença (W)")
    diferenca.axhline(0, color=".30", lw=.85, zorder=1)
    for (limite, linha), dx, cor, marca in zip(alternativas.iterrows(),
                                               [-.13, .13], [BLUE, ORANGE], ["o", "^"]):
        selecao = janelas.loc[janelas.demais_criterios.eq(1) & janelas.desvio_horizontal_ms.le(limite)]
        if len(selecao) != linha.n_janelas:
            raise ValueError(f"Contagem divergente para o limite horizontal de {limite} m/s")
        mediana = selecao.groupby("voo").potencia_W.median().loc[voos]
        delta = mediana-central
        diferenca.vlines(voos+dx, np.zeros(len(voos)), delta, color=cor, lw=.8)
        diferenca.scatter(voos+dx, delta, marker=marca, s=17, color=cor, linewidth=0,
                          label=f"Limite de {decimal(limite)} m/s", zorder=3)
    diferenca.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    diferenca.margins(y=.12)
    painel_voos((potencia, diferenca), voos)
    rotulos, nomes = diferenca.get_legend_handles_labels()
    fig.legend(rotulos, nomes, loc="lower center", bbox_to_anchor=(.54, .012), ncol=2,
               fontsize=8.4, frameon=False, handlelength=1, columnspacing=2)
    salvar(fig, "fig_potencia_limites_cruzeiro.pdf")


def desenhar_sensibilidade(ensaios):
    base = ensaios.loc[ensaios.criterio == "referencia"].iloc[0]
    fig, (criterios, tempo, erro) = plt.subplots(
        1, 3, figsize=(7.35, 4.0), sharey=True, gridspec_kw={"width_ratios": [1.7, 1.33, 1.33]})
    fig.subplots_adjust(left=.015, right=.985, bottom=.105, top=.86, wspace=.10)
    criterios.set_axis_off()
    criterios.text(0, 1.04, "Critério", transform=criterios.transAxes, fontweight="bold")
    criterios.text(.97, 1.04, "De → para", ha="right", transform=criterios.transAxes, fontweight="bold")

    for eixo, nome, referencia, unidade, marcas in (
        (tempo, "(a) Tempo selecionado", base.tempo_min, "min", [-10, -5, 0, 5]),
        (erro, "(b) Erro absoluto médio", base.mae_comparacao_W, "W", [0, .5, 1]),
    ):
        eixo.set_title(nome, pad=17, fontweight="bold", fontsize=9)
        ref = f"{referencia:.2f}".replace(".", ",")
        eixo.text(.5, 1.015, f"Referência de {ref} {unidade}", ha="center", va="bottom",
                  transform=eixo.transAxes, fontsize=8, color=".30")
        eixo.axvline(0, color=".40", lw=.9)
        eixo.set_xticks(marcas)
        eixo.xaxis.set_major_formatter(FuncFormatter(decimal))
        eixo.spines["left"].set_visible(False)
        eixo.set_yticks([])
        eixo.set_xlabel(f"Variação ({unidade})")
    tempo.set_xlim(-13.9, 7.6)
    erro.set_xlim(-.34, 1.34)

    for i, (campo, nome) in enumerate(CRITERIOS.items()):
        y = 2.6 * i
        criterios.text(0, y+.47, nome, va="center", transform=criterios.get_yaxis_transform())
        alternativas = ensaios.loc[ensaios.criterio == campo].sort_values("valor")
        for j, alternativa in enumerate(alternativas.itertuples()):
            linha = y + .94*j
            mudanca = f"{decimal(base[campo])} → {decimal(alternativa.valor)}"
            criterios.text(.97, linha, mudanca, ha="right", va="center",
                          transform=criterios.get_yaxis_transform())
            for eixo, medida, casas, cor in ((tempo, "tempo_min", 1, BLUE), (erro, "mae_comparacao_W", 2, ORANGE)):
                delta = getattr(alternativa, medida) - base[medida]
                eixo.barh(linha, delta, height=.48, color=cor, zorder=2)
                valor = f"{delta:+.{casas}f}".replace(".", ",").replace("-", "−")
                if round(delta, casas) == 0:
                    valor = "0" if delta == 0 else "≈0"
                lado = -1 if delta < 0 else 1
                eixo.annotate(valor, (delta, linha), xytext=(3*lado, 0), textcoords="offset points",
                              ha="right" if lado < 0 else "left", va="center", fontsize=8)
        if i < len(CRITERIOS)-1:
            for eixo in (criterios, tempo, erro):
                eixo.axhline(y+1.77, color=".88", lw=.55, zorder=0)
    tempo.set_ylim(y+1.5, -.55)
    salvar(fig, "fig_sensibilidade_cruzeiro.pdf")


if __name__ == "__main__":
    janelas = pd.read_csv(DADOS / "janelas_cruzeiro_por_voo.csv")
    ensaios = pd.read_csv(DADOS / "sensibilidade_cruzeiro.csv")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.6,
                         "axes.titlesize": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "axes.axisbelow": True}):
        voos = np.arange(1, 27)
        if not np.array_equal(np.sort(janelas.voo.unique()), voos):
            raise ValueError("A base deve conter os voos 1 a 26.")
        referencia = ensaios.loc[ensaios.criterio == "referencia"].iloc[0]
        if janelas.aceita.sum() != referencia.n_janelas:
            raise ValueError("As janelas aceitas divergem do cenário de referência.")
        alternativas = ensaios.loc[ensaios.criterio == "horizontal_ms"].set_index("horizontal_ms").sort_index()
        desenhar_selecao(janelas, referencia, voos)
        desenhar_altitude_lacuna(janelas, referencia, voos)
        desenhar_potencia(janelas, referencia, alternativas, voos)
        desenhar_sensibilidade(ensaios)
