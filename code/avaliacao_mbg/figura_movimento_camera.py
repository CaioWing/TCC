"""Confronta o movimento aparente de uma piscina e a direção anotada nos seis voos."""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from dados import CACHE, ROOT, folder
from figura_exemplo_temporal import fotos_do_video
from memoria_sim import T
from metricas import ground, labels
from movimento_camera import pontos_medicao
from paleta_tcc import BLUE, ORANGE
from passagens import DESTINO


def main():
    video = "video23_regionL"
    quadros = [3649, 3679]
    with np.load(CACHE / f"{video}.npz") as dados:
        pos = np.searchsorted(dados["frames"], quadros)
        assert np.array_equal(dados["frames"][pos], quadros)
        poses, tamanho, gsd = dados["pose"][pos], dados["size"], float(dados["gsd"])
        grade = dados["tiles"]
    caixas = np.stack([next(b[:4] for b in objetos if int(b[4]) == 267)
                       for objetos in labels(video, {"frames": quadros})])
    centros = (caixas[:, :2] + caixas[:, 2:]) / 2
    e, n = (ground(caixas[:1], poses[0], tamanho, gsd)[0] - poses[1, :2]) / gsd
    c, s = np.cos(poses[1, 2]), np.sin(poses[1, 2])
    esperado = tamanho / 2 + [c*e - s*n, -s*e - c*n]
    lado = 420
    x, y = np.clip(np.rint(centros.mean(axis=0) - lado/2).astype(int), 0, tamanho-lado)
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.45), width_ratios=[1.4, 1, 1.05])
    fig.subplots_adjust(left=.012, right=.978, bottom=.27, top=.83, wspace=.20)
    w, h = tamanho
    ax = axs[0]
    foto, = fotos_do_video([quadros[0]], video)
    ax.imshow(foto, extent=(0, w, h, 0))
    assert len(grade) == 24
    for origem in grade:
        ax.add_patch(Rectangle(origem, T, T, fill=False, ec="#ffb538", lw=.55,
                               path_effects=[patheffects.withStroke(linewidth=1.1, foreground="#253340")]))
    pontos = tamanho / 2 + pontos_medicao(tamanho)
    ax.scatter(*pontos.T, marker="+", s=30, color="white", linewidth=1,
               path_effects=[patheffects.withStroke(linewidth=2.1, foreground="#20292e")])
    ax.add_patch(Rectangle((x, y), lado, lado, fill=False, ec="white", lw=1.25,
                           path_effects=[patheffects.withStroke(linewidth=2.6, foreground="#20292e")]))
    ax.set(xlim=(0, w), ylim=(h, 0))
    ax.set_axis_off()
    ax.set_title("(a) Quadro 3649 · 24 recortes", loc="left", fontsize=8.0)

    # Mantém a resolução do pequeno detalhe, onde se medem os centros em pixels.
    arquivo, = (folder(video) / "avi").glob("*.avi")
    leitor = cv2.VideoCapture(str(arquivo))
    try:
        leitor.set(cv2.CAP_PROP_POS_FRAMES, quadros[1])
        ok, bgr = leitor.read()
        if not ok or round(leitor.get(cv2.CAP_PROP_POS_FRAMES)) != quadros[1]+1:
            raise IOError(f"Quadro {quadros[1]} indisponível")
        detalhe = cv2.cvtColor(bgr[y:y+lado, x:x+lado], cv2.COLOR_BGR2RGB)
    finally:
        leitor.release()
    ax = axs[1]
    ax.imshow(detalhe, extent=(x, x+lado, y+lado, y))
    u, v, u2, v2 = caixas[1]
    ax.add_patch(Rectangle((u, v), u2-u, v2-v, fill=False, ec="white", lw=.9))
    ax.annotate("", xy=centros[1], xytext=centros[0], arrowprops=dict(arrowstyle="->", color="white", lw=1.15))
    ax.scatter(*centros[0], marker="o", facecolors="none", edgecolors="white", linewidth=1.25, s=28, zorder=4)
    ax.scatter(*centros[1], color=BLUE, ec="white", lw=.8, s=28, zorder=4)
    ax.scatter(*esperado, marker="D", color=ORANGE, ec="white", s=27, zorder=4)
    ax.set_title("(b) Quadro 3679 · detalhe", loc="left", fontsize=8.)
    ax.set_axis_off()
    ax = axs[2]
    trilhas = pd.read_csv(DESTINO / "camera/direcao_trilhas.csv")
    grupos = trilhas.groupby("video", sort=True)
    for linha, (_, g) in enumerate(grupos):
        angulos = np.sort(g.angulo_observado.to_numpy())
        ax.scatter(angulos, np.full(len(g), linha), marker="|", s=22, lw=.6, color="#9daebb")
        ax.scatter(np.median(angulos), linha, marker="o", s=16, color=BLUE, zorder=3)
    ax.set_yticks(range(len(grupos)), [v[5:7] for v in grupos.groups])
    ax.invert_yaxis()
    ax.set_title("(c) Direção observada por voo", loc="left", fontsize=8.4)
    ax.set_xlabel("Ângulo com o eixo mais próximo (°)", fontsize=7.4)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(-.5, 20)
    ax.grid(axis="x", alpha=.22)
    ax.spines[["right","top","left"]].set_visible(False)
    ax.tick_params(labelsize=7, length=2)
    handles = [Line2D([], [], ls="", marker="o", mfc="none", mec="#444", label="Centro no quadro 3649"),
               Line2D([], [], ls="", marker="o", color=BLUE, label="Centro no quadro 3679"),
               Line2D([], [], ls="", marker="D", color=ORANGE, label="Previsão pela telemetria")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025), ncol=3,
               frameon=False, fontsize=7, handletextpad=.1, columnspacing=.5)
    fig.savefig(ROOT / "doc/figuras/fig_movimento_camera.pdf", bbox_inches="tight", pad_inches=.045)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family":"DejaVu Sans","pdf.fonttype":42}):
        main()
