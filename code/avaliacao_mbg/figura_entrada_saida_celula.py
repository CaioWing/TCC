"""Compara entrada e saída na mesma célula usada na figura de duas faixas."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figura_saida_faixas import desenhar as desenhar_comparacao


def desenhar():
    desenhar_comparacao(por_faixa=False)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        desenhar()
