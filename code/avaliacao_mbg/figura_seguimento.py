"""Recall geográfico e temporal com orçamento igual em cada par de estratégias."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dados import ROOT
from paleta_tcc import BLUE, ORANGE
from seguimento import METAS, RESULTADOS


def desenhar():
    resumo = pd.read_csv(RESULTADOS / "resumo.csv")
    fig, eixos = plt.subplots(1, 2, figsize=(6.65, 3.35), sharey=True)
    for ax, regra, titulo in zip(
        eixos,
        ("natural", "fixo_bruto"),
        ("(a) Demanda própria de cada meta", "(b) Mesmo orçamento em todas as metas"),
    ):
        parte = resumo[resumo.orcamento == regra]
        mem = parte[parte.metodo == "memoria"].set_index("k").loc[METAS]
        tempo = (
            parte[parte.metodo == "temporal"]
            .groupby("k")
            .recall.agg(["min", "mean", "max"])
            .loc[METAS]
            * 100
        )
        ax.fill_between(
            METAS,
            tempo["min"].to_numpy(),
            tempo["max"].to_numpy(),
            facecolor=ORANGE,
            alpha=0.18,
            linewidth=0,
            label="Extremos das quatro fases",
        )
        ax.plot(
            METAS,
            tempo["mean"],
            color=ORANGE,
            ls="--",
            marker="s",
            ms=4,
            label="Temporal (média das fases)",
        )
        ax.plot(METAS, 100 * mem.recall, color=BLUE, marker="o", ms=4, label="Memória geográfica")
        ax.set_title(titulo, fontsize=9, loc="left")
        if regra == "natural":
            ax.set_xticks(
                METAS, [f"{k}\n({n:,})".replace(",", ".") for k, n in zip(METAS, mem.recortes)]
            )
            ax.set_xlabel("Faixas por célula\n(total de recortes)", fontsize=8)
        else:
            ax.set_xticks(METAS)
            ax.set_xlabel(
                f"Faixas por célula\n({mem.recortes.iloc[0]} recortes em cada meta)", fontsize=8
            )
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.24)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
    topo = min(
        100,
        10
        * np.ceil(
            (100 * resumo[resumo.orcamento.isin(["natural", "fixo_bruto"])].recall.max() + 3) / 10
        ),
    )
    eixos[0].set_ylim(0, topo)
    eixos[0].set_ylabel("Referências reconhecidas (%)", fontsize=8)
    h, l = eixos[0].get_legend_handles_labels()
    fig.legend(
        [h[2], h[1], h[0]],
        [l[2], l[1], l[0]],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.9, bottom=0.30, wspace=0.14)
    fig.savefig(
        ROOT / "doc/figuras/fig_seguimento_multivistas.pdf", bbox_inches="tight", pad_inches=0.05
    )
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "lines.linewidth": 1.5}):
        desenhar()
