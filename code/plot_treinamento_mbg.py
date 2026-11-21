"""Reproduz as curvas de treinamento e validação a partir do histórico da execução."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import yaml

from paleta_tcc import BLUE, ORANGE


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/mbg_y11n_640_v1"


def main():
    data = pd.read_csv(RUN / "results.csv")
    config = yaml.safe_load((RUN / "args.yaml").read_text(encoding="utf-8"))
    first_without_mosaic = config["epochs"] - config["close_mosaic"] + 1
    epoch = data["epoch"]
    map_strict = data["metrics/mAP50-95(B)"]
    best = map_strict.idxmax()

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, (loss, metrics) = plt.subplots(2, 1, sharex=True, figsize=(8.3, 4.75))
        fig.subplots_adjust(left=.095, right=.78, bottom=.12, top=.925, hspace=.50)
        for ax in (loss, metrics):
            ax.axvspan(first_without_mosaic - 0.5, epoch.max() + 0.5,
                       color=".93", linewidth=0, zorder=0)
            ax.set(xlim=(1, epoch.max()+.5), xticks=[1, 5, 10, 15, 17, 21, 25, 27])
            ax.grid(axis="y", color="0.88", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))

        for ax, values, label, color, linestyle in [
            (loss, data["train/box_loss"], "Treino", BLUE, "-"),
            (loss, data["val/box_loss"], "Validação", ORANGE, "--"),
            (metrics, 100 * data["metrics/mAP50(B)"], "mAP50", BLUE, "-"),
            (metrics, 100 * map_strict, "mAP50–95", ORANGE, "--"),
        ]:
            ax.plot(epoch, values, color=color, ls=linestyle, lw=1.8)
            ax.annotate(label, (epoch.iloc[-1], values.iloc[-1]), xytext=(11, 0),
                        textcoords="offset points", va="center", color="0.14", clip_on=False)
        loss.set_title("(a) Localização das caixas", loc="left")
        loss.set(ylabel="Perda de localização", ylim=(.76, 1.24), yticks=[.8, .9, 1, 1.1, 1.2])
        loss.text((first_without_mosaic+epoch.max())/2, 1.20, "Sem mosaico",
                  ha="center", va="center", fontsize=9)

        metrics.set_title("(b) Detecção nas 12 classes de validação", loc="left")
        metrics.set(xlabel="Época", ylabel="mAP (%)", ylim=(8, 35), yticks=[10, 15, 20, 25, 30, 35])
        peak = 100 * map_strict[best]
        metrics.scatter(epoch[best], peak, s=35, facecolors="white", edgecolors=ORANGE,
                        lw=1.6, zorder=3)
        metrics.annotate(f"Pesos selecionados · época {epoch[best]}", (epoch[best], peak),
                         xytext=(epoch[best]-2.4, 12.3), ha="center", fontsize=9,
                         arrowprops={"arrowstyle":"-", "color":".35", "lw":.8})
        fig.savefig(ROOT / "doc/figuras/fig_treino_curvas.pdf", bbox_inches="tight", pad_inches=.07)
        plt.close(fig)


if __name__ == "__main__":
    main()
