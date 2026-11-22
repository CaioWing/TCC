"""Reproduz a seleção de um recorte do voo 17 e a localiza no quadro original."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd

from memoria_sim import T, load_pose, simulate, tiles_grid
from paleta_tcc import BLUE, GREEN, ORANGE


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mbg_v2"
GSD = .014  # m/pixel, ajuste do voo 17


def project(east, north, yaw, k, pixels, width, height):
    u, v = pixels.T
    front = (height/2 - v) * GSD
    right = (u - width/2) * GSD
    c, s = np.cos(yaw[k]), np.sin(yaw[k])
    return np.column_stack((east[k] + s*front + c*right, north[k] + c*front - s*right))


def main():
    folder = DATA / "raw/flight-mbg-v2-17-regionI"
    log, = (folder / "log").glob("*.csv")
    video, = (folder / "avi").glob("*.avi")
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=avg_frame_rate,nb_frames,width,height", "-of", "json", str(video)
    ]))["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    n, w, h = int(stream["nb_frames"]), stream["width"], stream["height"]
    east, north, yaw = load_pose(log, n, fps)
    window = round(fps)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    example = []

    def select_example(k, x, y, u, v):
        # Primeiro acionamento após 1/4 do voo, com saída pela borda e longe dos cantos.
        if example or k < n//4 or k + window >= n:
            return
        middle = np.flatnonzero((u > .22*w) & (u < .78*w))
        if not len(middle):
            return
        distance = np.hypot(east[k+window]-east[k], north[k+window]-north[k])
        if 3.9 < distance < 4.3 and np.ptp(yaw[k:k+window+1]) < np.deg2rad(1):
            idx = middle[len(middle)//2]
            example.append((k, x, y, np.array([[u[idx], v[idx]]])))

    saved = pd.read_csv(DATA / "splits/memoria_sim_26voos.csv").set_index("vid").loc[17]
    ntiles = len(tiles_grid(w, h))
    matched_step = round(ntiles * n / saved.adiada_recortes)
    result = simulate(log, n, fps, w, h, GSD, skip_steps=(30, matched_step), on_deferred_tile=select_example)
    for policy in ("imediata", "adiada"):
        if result[policy]["recortes"] != saved[f"{policy}_recortes"]:
            raise ValueError("A contagem de recortes diverge da simulação registrada.")
    (k, x, y, point), = example
    sample = project(east, north, yaw, k, point, w, h)[0]
    next_pose = project(east, north, yaw, k+1, corners, w, h)
    if cv2.pointPolygonTest(next_pose.astype("float32"), tuple(sample), False) >= 0:
        raise ValueError("A célula destacada também aparece no quadro seguinte.")
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != k+1:
            raise ValueError(f"Não foi possível obter o quadro {k} do voo 17.")
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (960, 540), interpolation=cv2.INTER_AREA)
    finally:
        cap.release()

    def footprint(j):
        return project(east, north, yaw, j, corners, w, h)

    start = np.array([east[k], north[k]])
    delta = np.array([east[k+window], north[k+window]]) - start
    distance = np.linalg.norm(delta)
    forward = delta / distance
    # Eixos locais ortogonais à direção do avanço e ao longo dela.
    rotation = np.column_stack(([forward[1], -forward[0]], forward))
    before, after = [(footprint(j)-start) @ rotation for j in [k, k+window]]
    shared, _ = cv2.intersectConvexConvex(before.astype("float32"), after.astype("float32"))
    new_pct = 100 * (1 - shared / ((w*GSD) * (h*GSD)))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(8.6, 5.65))
        grid = fig.add_gridspec(2, 3, left=.04, right=.97, top=.90, bottom=.12,
                                width_ratios=[.13, .40, .47], height_ratios=[1.05, 1],
                                hspace=.41, wspace=.15)
        overlay = fig.add_subplot(grid[0, :2])
        camera = fig.add_subplot(grid[0, 2])
        lower = grid[1, 1:].subgridspec(1, 2, width_ratios=[5, 1], wspace=.10)
        counts_ax = fig.add_subplot(lower[0])
        coverage_ax = fig.add_subplot(lower[1], sharey=counts_ax)
        coverage_ax.set_title("Área coberta", fontsize=10, pad=8)
        coverage_ax.axis("off")

        for ax, title in [(overlay, "(a) Terreno novo após um segundo"), (camera, "(b) Última vista da célula")]:
            bounds = ax.get_position(original=True)
            fig.text(bounds.x0, bounds.y1 + .014, title, fontsize=11, va="bottom")
        overlay.add_patch(Polygon(after, facecolor="#d6e4d3", edgecolor=GREEN, lw=1.2))
        overlay.add_patch(Polygon(before, facecolor="#e0e9ef", edgecolor=BLUE, lw=1.2))
        overlay.add_patch(Polygon(after, fill=False, edgecolor=GREEN, lw=1.15, ls="--"))
        tile_px = np.array([[x, y], [x+T, y], [x+T, y+T], [x, y+T]])
        tile_ground = (project(east, north, yaw, k, tile_px, w, h)-start) @ rotation
        cell_ground = (sample-start) @ rotation
        overlay.add_patch(Polygon(tile_ground, fill=False, edgecolor=ORANGE, lw=2))
        overlay.plot(*cell_ground, "o", color=ORANGE, mec="white", ms=6, zorder=3)
        # Centro da borda de entrada nos dois instantes. Entre as bordas está a faixa nova.
        leading_edges = [poly[np.argsort(poly[:, 1])[-2:]].mean(axis=0) for poly in (before, after)]
        new_band = np.mean(leading_edges, axis=0)
        overlay.annotate(f"Terreno novo ({new_pct:.1f}%)".replace(".", ","), new_band,
                         xytext=(0, 25), ha="center", va="center", fontsize=9.2,
                         arrowprops={"arrowstyle": "-", "lw": 1, "color": GREEN})
        overlay.text(0, 2, "Terreno visível\nnos dois instantes", ha="center", va="center", fontsize=9.3)
        overlay.annotate("Recorte selecionado", tile_ground.mean(axis=0), xytext=(0, -22),
                         ha="center", va="center", fontsize=9.2,
                         arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1})
        overlay.set_aspect("equal")
        overlay.set(xlim=(-32, 32), ylim=(-25, 28))
        overlay.axis("off")

        camera.imshow(gray, extent=(0, w, h, 0), cmap="gray", vmin=-28, vmax=275)
        for u0, v0 in tiles_grid(w, h):
            camera.add_patch(Rectangle((u0, v0), T, T, fill=False, lw=.45, edgecolor="white", alpha=.55))
        camera.add_patch(Rectangle((x, y), T, T, fill=False, lw=2.2, edgecolor=ORANGE))
        camera.plot(*point[0], marker="o", mec="white", mfc=ORANGE, ms=6, clip_on=False)
        camera.annotate("Célula que sai de vista", point[0], xytext=(16, 36), textcoords="offset points",
                        ha="left", va="center", fontsize=9,
                        bbox={"facecolor":"white", "edgecolor":"none", "pad":2},
                        arrowprops={"arrowstyle":"-", "lw":1, "color":"white"})
        camera.set(xlim=(-20, w+20), ylim=(h+20, -20))
        camera.axis("off")

        counts_ax.set_title("(c) Recortes selecionados no voo completo", loc="left", pad=8)
        interval = f"Intervalo ajustado por voo\n(a cada {matched_step} quadros)"
        counts = {
            "Processamento integral": ntiles * n,
            "Seleção na entrada": result["imediata"]["recortes"],
            "A cada 30 quadros": len(range(0, n, 30)) * ntiles,
            interval: len(range(0, n, matched_step)) * ntiles,
            "Seleção na saída": result["adiada"]["recortes"],
        }
        coverage = {
            "Processamento integral": 1.0,  # A união de todos os quadros define a área imageada.
            "Seleção na entrada": 1 - result["imediata"]["frac_nao_analisada"],
            "A cada 30 quadros": 1 - result["skip_30"]["frac_nao_vista"],
            interval: 1 - result[f"skip_{matched_step}"]["frac_nao_vista"],
            "Seleção na saída": 1 - result["adiada"]["frac_nao_analisada"],
        }
        for row, (name, count) in enumerate(counts.items()):
            color = {"Seleção na entrada": BLUE, "Seleção na saída": ORANGE}.get(name, ".44")
            counts_ax.scatter(count, row, color=color, s=38, zorder=3)
            label = f"{count:,}".replace(",", ".")
            percent = f"{100*coverage[name]:.2f}".removesuffix(".00").replace(".", ",")
            coverage_ax.text(.5, row, f"{percent}%", ha="center", va="center",
                             transform=coverage_ax.get_yaxis_transform(), fontsize=9.6)
            counts_ax.annotate(label, (count, row), xytext=(6, 0), textcoords="offset points",
                               va="center", fontsize=9.6)
        counts_ax.set_xscale("log")
        counts_ax.set(xlim=(1000, 500000), ylim=(len(counts)-.55, -.5), yticks=range(len(counts)),
                      yticklabels=list(counts), xticks=[1000, 3000, 10000, 30000, 100000, 300000],
                      xticklabels=["1 mil", "3 mil", "10 mil", "30 mil", "100 mil", "300 mil"],
                      xlabel="Total de recortes (escala logarítmica)")
        counts_ax.minorticks_off()
        counts_ax.tick_params(axis="y", length=0)
        counts_ax.grid(axis="x", color=".88", lw=.6)
        fig.savefig(ROOT / "doc/figuras/fig_memoria_voo17.pdf", dpi=260, bbox_inches="tight", pad_inches=.08)
        plt.close(fig)
    print(f"Quadro {k}, t={k/fps:.2f} s, avanço={distance:.2f} m, área nova={new_pct:.2f}%")
    print("Totais:", counts)
    print("Cobertura por estratégia:", coverage)


if __name__ == "__main__":
    main()
