"""Vazão, energia e qualidade do YOLO11n nos ensaios da Jetson Orin Nano."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

from paleta_tcc import BLUE, GREEN, ORANGE


ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "data/yolo11n-jetson24-20261116"
GRUPOS = [
    ("TensorRT nativo", [("trt_fp32", "FP32"), ("trt_fp16", "FP16"), ("trt_int8", "INT8")]),
    ("ONNX Runtime", [("ort_trt_fp32", "TensorRT · FP32"), ("ort_trt_fp16", "TensorRT · FP16"),
                     ("ort_cuda_fp32", "CUDA · FP32"), ("ort_cpu_fp32", "CPU · FP32")]),
    ("PyTorch (TorchScript)", [("torchscript_fp32", "CUDA · FP32"),
                              ("torchscript_fp16", "CUDA · FP16")]),
]


def decimal(valor):
    return f"{valor:.1f}".replace(".", ",")


def comparar_motores(dados):
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 4.16))
    fig.subplots_adjust(left=.255, right=.98, top=.92, bottom=.20, wspace=.15)
    metodos, posicoes, linhas, secoes = [], [], [], []
    pos = 0.
    for grupo, configs in GRUPOS:
        secoes.append((pos, grupo))
        for ident, linha in configs:
            pos += 1
            metodos.append(ident)
            posicoes.append(pos)
            linhas.append(linha)
        pos += 1.55

    for ax, (titulo, campo, ticks, limites) in zip(axes, [
        ("(a) Vazão", "fps", [5, 10, 20, 50, 100, 200], (4.8, 265)),
        ("(b) Energia total", "E_total_mJ_per_inf", [50, 100, 200, 500, 1000], (45, 1850)),
    ], strict=True):
        tensor = dados.loc[[(ident, "infer") for ident in metodos], campo].to_numpy()
        imagem = dados.loc[[(ident, "e2e") for ident in metodos], campo].to_numpy()
        ax.hlines(posicoes, np.minimum(tensor, imagem), np.maximum(tensor, imagem),
                  lw=1.15, color="#adb1b4", zorder=2)
        ax.scatter(tensor, posicoes, marker="D", s=25, edgecolors=BLUE, facecolors="white",
                   linewidths=1.15, zorder=3)
        ax.scatter(imagem, posicoes, marker="o", s=22, color=ORANGE, zorder=4)
        for y, img, bruto in zip(posicoes, imagem, tensor):
            lado = 1 if img >= bruto else -1
            ax.annotate(decimal(img), (img, y), xytext=(4.5 * lado, .3),
                        textcoords="offset points", ha="left" if lado > 0 else "right",
                        va="center", fontsize=6.9, color="#4e361f")
        ax.set_xscale("log")
        ax.set_xlim(*limites)
        ax.set_ylim(posicoes[-1] + .6, -.35)
        ax.set_xticks(ticks, [str(t).replace("1000", "1.000") for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="x", color=".90", lw=.65)
        ax.set_title(titulo, loc="left", fontsize=10)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=2.5)
    axes[0].set_xlabel("Inferências/s")
    axes[1].set_xlabel("mJ por inferência")
    for y, nome in zip(posicoes, linhas):
        axes[0].text(-.04, y, nome, ha="right", va="center",
                     transform=axes[0].get_yaxis_transform(), fontsize=8)
    for y, nome in secoes:
        axes[0].text(-.04, y, nome, ha="right", va="center", weight="semibold",
                     transform=axes[0].get_yaxis_transform(), fontsize=8.5)
    handles = [Line2D([], [], marker="o", ls="", color=ORANGE, ms=5.8, label="Imagem até as detecções"),
               Line2D([], [], marker="D", ls="", mec=BLUE, mfc="white", mew=1.1, ms=5.8,
                      label="Tensor até a saída bruta")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               handletextpad=.45, columnspacing=2., bbox_to_anchor=(.55, .005))
    return fig


def comparar_precisoes(dados):
    fig, eixos = plt.subplots(1, 2, figsize=(6.1, 2.15))
    fig.subplots_adjust(left=.10, right=.985, top=.78, bottom=.14, wspace=.36)
    identificadores = ["trt_fp32", "trt_fp16", "trt_int8"]
    medidas = dados.loc[[(ident, "e2e") for ident in identificadores]]
    for ax, valores, titulo, unidade in zip(
            eixos, [100 * medidas.mAP50_95, medidas.E_dyn_mJ_per_inf],
            ["(a) Detecção no COCO", "(b) Energia acima do repouso"],
            ["mAP (%)", "mJ por inferência"], strict=True):
        bars = ax.bar(["FP32", "FP16", "INT8"], valores, color=[BLUE, ORANGE, GREEN], width=.54)
        ax.bar_label(bars, labels=[decimal(v) for v in valores], padding=2.7, fontsize=8.5)
        ax.set_ylim(0, max(valores) * 1.23)
        ax.set_title(titulo, fontsize=9.4, loc="left", pad=6)
        ax.set_ylabel(unidade)
        ax.grid(axis="y", color=".90", lw=.65)
        ax.tick_params(axis="x", length=0)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


if __name__ == "__main__":
    dados = pd.read_csv(DADOS / "summary.csv").set_index(["config", "mode"])
    if not dados.index.is_unique:
        raise ValueError("Cada combinação de motor, precisão e modo requer uma única medição.")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.4, "pdf.fonttype": 42,
                         "axes.axisbelow": True}):
        for nome, fig in (("fig_bancada_vazao_energia", comparar_motores(dados)),
                          ("fig_bancada_precisao_coco", comparar_precisoes(dados))):
            fig.savefig(ROOT / "doc/figuras" / f"{nome}.pdf", bbox_inches="tight", pad_inches=.045)
            plt.close(fig)
