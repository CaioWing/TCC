"""Recall com poucas vistas e distância ao integral em orçamentos maiores."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dados import OUT, ROOT
from exemplo_passagem_pareada import desenhar_exemplo
from paleta_tcc import BLUE, ORANGE


def comparar(ax, x, amostra, referencia=0.0):
    """A fase muda os quadros temporais selecionados sob a mesma cota de recortes."""
    tempo = amostra["media"]
    ax.errorbar(
        x, tempo - referencia, yerr=[tempo - amostra["menor"], amostra["maior"] - tempo],
        color=ORANGE, linestyle="--", linewidth=1.25, marker="s", markersize=3.2,
        elinewidth=.75, capsize=2, capthick=.75, label="Temporal por trecho",
    )
    ax.plot(x, amostra["geografico"] - referencia, color=BLUE, linewidth=1.6,
            marker="o", markersize=3.8, label="Geográfica por passagem", zorder=4)


def desenhar():
    resumo = pd.read_csv(OUT / "passagens/curva/resumo.csv")
    integral = resumo[resumo.estrategia == "integral"].iloc[0]
    geografia = resumo[resumo.estrategia == "geografica"].set_index("k").sort_index()
    fases = resumo[resumo.estrategia == "temporal"].groupby("k")
    if not fases.size().eq(4).all() or not fases.recortes.min().equals(geografia.recortes):
        raise ValueError("A comparação requer quatro fases com o orçamento da seleção geográfica.")
    if not fases.recortes.max().equals(geografia.recortes):
        raise ValueError("As quatro fases devem usar o mesmo total de recortes.")
    serie = 100 * fases.recall.agg(media="mean", menor="min", maior="max")
    serie["geografico"] = 100 * geografia.recall
    serie["custo"] = 100 * geografia.recortes / integral.recortes
    # O CSV arquiva o total de vistas. A linha de uma vista pertence ao controle de entrada.
    serie = serie.loc[2:].copy()
    serie.index = serie.index - 2
    integral_recall = 100 * integral.recall
    cor_referencia = "#44494e"

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.4, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True}):
        fig = plt.figure(figsize=(7.35, 5.05))
        grade_curvas = fig.add_gridspec(1, 2, left=.077, right=.986, bottom=.55,
                                       top=.951, wspace=.31)
        inicio, custo = (fig.add_subplot(c) for c in grade_curvas)
        for ax in (inicio, custo):
            ax.grid(axis="y", color=".89", linewidth=.6)
            ax.tick_params(labelsize=7.6, length=3, pad=2)

        primeiras = serie.loc[:10]
        comparar(inicio, primeiras.index, primeiras)
        inicio.set(xlim=(-.35, 10.35), ylim=(20, 76.7), xticks=range(11),
                   yticks=[25, 40, 55, 70], xlabel=r"Quadros intermediários por passagem ($k_{\mathrm{int}}$)",
                   ylabel="Recall por recipiente (%)")
        inicio.set_title("(a) Entrada e saída + 0 a 10 intermediários", loc="left", fontsize=8.7)
        linha = inicio.axhline(integral_recall, color=cor_referencia, linewidth=.9, ls=":")
        valor_integral = f"{integral_recall:.1f}%".replace(".", ",")
        inicio.annotate(f"Integral · {valor_integral}", (1, integral_recall),
                        xytext=(1, 2.4), textcoords="offset points", fontsize=7.7, color=cor_referencia)
        if geografia.loc[4, "vp"] == geografia.loc[5, "vp"]:
            valor = primeiras.loc[2, "geografico"]
            inicio.annotate(f"{valor:.1f}%".replace(".", ","), (2.5, valor),
                            xytext=(2, -16), textcoords="offset points", ha="center", color=BLUE,
                            fontsize=7.5, arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=.65))

        restantes = serie.loc[10:]
        comparar(custo, restantes["custo"], restantes, integral_recall)
        custo.axhline(0, color=cor_referencia, linewidth=linha.get_linewidth(), ls=":")
        custo.scatter(100, 0, marker="D", color=cor_referencia, s=22, zorder=5)
        custo.annotate("Integral", (100, 0), xytext=(-1, 6), textcoords="offset points",
                       ha="right", fontsize=7.5, color=cor_referencia)
        custo.set(xlim=(0, 104), ylim=(-15, 2.5), xticks=range(0, 101, 20),
                  yticks=[-15, -10, -5, 0], xlabel="Recortes usados (% do integral)",
                  ylabel="Recall em relação ao integral (p.p.)")
        custo.set_title("(b) De 10 intermediários à passagem completa", loc="left", fontsize=8.7)
        k_melhor = restantes.geografico.idxmax()
        diferenca = restantes.at[k_melhor, "geografico"] - integral_recall
        rotulo_melhor = (f"$k_{{\\mathrm{{int}}}}={k_melhor}$\n{diferenca:.1f} p.p."
                         .replace(".", ",", 1).replace("-", "−"))
        ultima_meta = restantes.index.max()
        for k, texto, pos, ha in [(62, r"$k_{\mathrm{int}}=62$", (6, -18), "left"),
                                  (k_melhor, rotulo_melhor, (-4, -17), "right"),
                                  (ultima_meta, "Todos os quadros", (1, -12), "center")]:
            ponto = restantes.loc[k]
            custo.annotate(texto, (ponto.custo, ponto.geografico - integral_recall), xytext=pos,
                           textcoords="offset points", ha=ha, fontsize=7.5, color=BLUE,
                           va="top", arrowprops=dict(arrowstyle="-", color=BLUE, lw=.6, shrinkB=2.5))

        handles, _ = inicio.get_legend_handles_labels()
        ordem = {h.get_label(): h for h in handles}
        fig.legend([ordem["Geográfica por passagem"], ordem["Temporal por trecho"]],
                   ["Geográfica por passagem", "Temporal por trecho (média e extremos das quatro fases)"],
                   loc="lower center", ncol=2, frameon=False, columnspacing=1.8,
                   fontsize=7.7, bbox_to_anchor=(.5, .405), handlelength=2.1)
        imagens = fig.add_gridspec(1, 2, left=.008, right=.994, top=.353,
                                   bottom=.01, wspace=.045)
        desenhar_exemplo([fig.add_subplot(c) for c in imagens])
        fig.savefig(ROOT / "doc/figuras/fig_curva_recall_passagens.pdf",
                    bbox_inches="tight", pad_inches=.05, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    desenhar()
