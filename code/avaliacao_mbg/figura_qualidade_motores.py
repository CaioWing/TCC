"""Compara a detecção das mesmas referências por estratégia e ambiente de execução."""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from paleta_tcc import BLUE, GREEN, ORANGE, PURPLE


AMBIENTES = [
    ("(a) TensorRT", [("trt_fp32", "FP32"), ("trt_fp16", "FP16"), ("trt_int8", "INT8")]),
    ("(b) ONNX Runtime", [
        ("ort_trt_fp32", "TensorRT FP32"), ("ort_trt_fp16", "TensorRT FP16"),
        ("ort_cuda_fp32", "CUDA FP32"), ("ort_cpu_fp32", "CPU FP32"),
    ]),
    ("(c) TorchScript", [("torchscript_fp32", "FP32"), ("torchscript_fp16", "FP16")]),
]
ESTRATEGIAS = [
    ("todos", "Processamento integral", "#44494e", "o"),
    ("cada30", "Um quadro a cada 30", PURPLE, "s"),
    ("pareada", "Intervalo por voo", ORANGE, "D"),
    ("imediata", "Entrada da célula", GREEN, "^"),
    ("adiada", "Saída da célula", BLUE, "v"),
]


def main():
    origem = ROOT / "doc/dados/qualidade_memoria/qualidade_teste.csv"
    with origem.open(newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    (referencias,) = {int(linha["referencias"]) for linha in linhas}
    acertos = {(linha["motor"], linha["estrategia"]): int(linha["vp"]) for linha in linhas}
    combinacoes = {(motor, regra) for _, grupo in AMBIENTES for motor, _ in grupo
                   for regra, *_ in ESTRATEGIAS}
    if acertos.keys() != combinacoes or len(linhas) != len(combinacoes):
        raise ValueError("A figura exige um único resultado por configuração e estratégia.")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, eixos = plt.subplots(3, 1, figsize=(6.6, 3.75), sharex=True,
                              height_ratios=[len(grupo) for _, grupo in AMBIENTES])
    for ax, (titulo, grupo) in zip(eixos, AMBIENTES):
        ax.set_title(titulo, loc="left", fontsize=8.6, pad=3)
        ax.set_yticks(range(len(grupo)), [rotulo for _, rotulo in grupo])
        ax.set_ylim(len(grupo) - 0.48, -0.48)
        ax.set_xlim(0, referencias)
        ax.grid(axis="x", color="#d6dadd", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(length=0, labelsize=7.3)
        for chave, nome, cor, marca in ESTRATEGIAS:
            valores = [acertos[(motor, chave)] for motor, _ in grupo]
            ax.scatter(valores, range(len(grupo)), color=cor, marker=marca,
                       s=18, linewidths=0.4, edgecolors="#33383d", label=nome, zorder=3)
            for y, valor in enumerate(valores):
                ax.annotate(str(valor), (valor, y), xytext=(3.6, 0), textcoords="offset points",
                            va="center", fontsize=6.7)
    eixos[-1].set_xticks([*range(0, referencias, 20), referencias])
    eixos[-1].set_xlabel(f"Recipientes reconhecidos entre {referencias} referências", fontsize=8)
    itens, nomes = eixos[0].get_legend_handles_labels()
    fig.legend(itens, nomes, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.00),
               frameon=False, fontsize=7.3, columnspacing=1.5, handletextpad=0.2)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.11, top=0.81, hspace=0.56)
    fig.savefig(ROOT / "doc/figuras/fig_qualidade_motores.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
