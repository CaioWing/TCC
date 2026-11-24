"""Compara as vistas elegíveis das mesmas trilhas no trecho e na aparição completa."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paleta_tcc import BLUE, GRAY


ROOT = Path(__file__).resolve().parents[1]
ESTRATEGIAS = {
    "todos": "Processamento integral",
    "imediata": "Entrada da célula",
    "cada30": "Um quadro a cada 30",
    "pareada": "Intervalo por voo",
    "adiada": "Saída da célula",
}


def main():
    trilhas = pd.read_csv(ROOT / "doc/dados/qualidade_memoria/oportunidades_trilhas_completas.csv")
    posicoes = np.arange(len(ESTRATEGIAS))
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, eixo = plt.subplots(figsize=(6.9, 2.9))
        fig.subplots_adjust(left=.26, right=.94, bottom=.19, top=.86)
        for sufixo, nome, cor, deslocamento in [
            ("", "Trecho de 4 s", BLUE, -.17),
            ("_voo", "Aparição completa", GRAY, .17),
        ]:
            contagens = [trilhas[chave + sufixo].sum() for chave in ESTRATEGIAS]
            barras = eixo.barh(posicoes + deslocamento, contagens, height=.28,
                               color=cor, label=nome)
            eixo.bar_label(barras, padding=3, fontsize=8)
        eixo.set_yticks(posicoes, ESTRATEGIAS.values())
        eixo.tick_params(axis="y", length=0)
        eixo.invert_yaxis()
        eixo.set_xlim(0, len(trilhas) * 1.09)
        eixo.set_xlabel(f"Trilhas com ao menos uma vista elegível (de {len(trilhas)})")
        eixo.grid(axis="x", color=".9", lw=.6)
        eixo.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
                    frameon=False, borderaxespad=0)
        fig.savefig(ROOT / "doc/figuras/fig_oportunidades_teste.pdf", bbox_inches="tight", pad_inches=.07)
        plt.close(fig)


if __name__ == "__main__":
    main()
