"""Compara entrada e saída da célula na malha geográfica e no voo 17 completo."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from avaliacao_mbg.figuras_malha import BORDA, ENVIO, VISIVEL, amostra_de_celulas, contagem, legenda, painel
from memoria_sim import T, load_pose, simulate, tiles_grid
from paleta_tcc import BLUE, GREEN, ORANGE


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mbg_v2"
GSD = .014  # m/pixel, ajuste do voo 17
QUADRO = 2573  # Evento da figura: cruzeiro retilíneo aos 85,85 s.


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
    k = QUADRO
    grid_tiles = np.array(tiles_grid(w, h))
    selected = {"imediata": [], "adiada": []}
    def record_tile(policy, frame, x, y):
        if frame == k:
            selected[policy].append((x, y))

    # A entrada atende todas as células novas. Reconstituir a união das pegadas
    # anteriores também considera células já vistas numa passagem anterior.
    seen = None
    raster = entry_due = None
    def record_cells(frame, current, following, shape):
        nonlocal seen, raster, entry_due
        if frame > k:
            return
        if seen is None:
            seen = np.zeros(shape, bool)
        j0, j1, i0, i1, inside, _, _ = current
        region = seen[j0:j1, i0:i1]
        if frame == k:
            raster, entry_due = current, inside & ~region
        region |= inside

    exit_requests = []
    def record_requests(frame, x, y, u, v):
        if frame == k:
            exit_requests.append(np.column_stack((u, v)))

    saved = pd.read_csv(DATA / "splits/memoria_sim_26voos.csv").set_index("vid").loc[17]
    ntiles = len(grid_tiles)
    matched_step = round(ntiles * n / saved.adiada_recortes)
    result = simulate(log, n, fps, w, h, GSD, skip_steps=(30, matched_step),
                      on_selected=record_tile, on_deferred_frame=record_cells, on_deferred_tile=record_requests)
    for policy in ("imediata", "adiada"):
        if result[policy]["recortes"] != saved[f"{policy}_recortes"]:
            raise ValueError("A contagem de recortes diverge da simulação registrada.")
    j0, _, i0, _, inside, u, v = raster
    requests = np.unique(np.vstack(exit_requests), axis=0)
    exit_due = np.zeros_like(inside)
    for px, py in requests:
        exit_due |= inside & (u == px) & (v == py)
    assert exit_due.sum() == len(requests)
    middle = requests[(requests[:, 0] > .22*w) & (requests[:, 0] < .78*w)]
    point = middle[len(middle)//2]
    (j,), (i,) = np.nonzero(inside & (u == point[0]) & (v == point[1]))
    cell = j+j0, i+i0
    samples = {}
    for policy, due in [("imediata", entry_due), ("adiada", exit_due)]:
        tile_set = set(selected[policy])
        mask = [tuple(tile) in tile_set for tile in grid_tiles]
        samples[policy] = amostra_de_celulas(raster, due, grid_tiles, mask, cell)
    assert not samples["imediata"][-1] and samples["adiada"][-1]
    sample = project(east, north, yaw, k, point[None], w, h)[0]
    next_pose = project(east, north, yaw, k+1, corners, w, h)
    if cv2.pointPolygonTest(next_pose.astype("float32"), tuple(sample), False) >= 0:
        raise ValueError("A célula destacada também aparece no quadro seguinte.")
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, frame = cap.read()
        if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != k+1:
            raise ValueError(f"Não foi possível obter o quadro {k} do voo 17.")
        resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
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
        fig = plt.figure(figsize=(8.6, 5.05))
        grid = fig.add_gridspec(2, 3, left=.015, right=.985, top=.835, bottom=.115,
                                width_ratios=[.90, 1.22, 1.22], height_ratios=[1., 1.1],
                                hspace=.35, wspace=.09)
        overlay = fig.add_subplot(grid[0, 0])
        cameras = [fig.add_subplot(grid[0, col]) for col in (1, 2)]
        lower = fig.add_gridspec(1, 2, left=.235, right=.985, bottom=.115, top=.447,
                                 width_ratios=[5, 1], wspace=.10)
        counts_ax = fig.add_subplot(lower[0])
        coverage_ax = fig.add_subplot(lower[1], sharey=counts_ax)
        coverage_ax.set_title("Área coberta", fontsize=10, pad=8)
        coverage_ax.axis("off")

        legenda(fig, "Célula")
        for ax, title in [(overlay, "(a) Terreno em 1 s"), (cameras[0], f"(b) Decisão no quadro {k}")]:
            bounds = ax.get_position(original=True)
            fig.text(bounds.x0, bounds.y1 + .065, title, fontsize=10.5, va="bottom")
        overlay.add_patch(Polygon(after, facecolor="#d6e4d3", edgecolor=GREEN, lw=1.2))
        overlay.add_patch(Polygon(before, facecolor="#e0e9ef", edgecolor=BLUE, lw=1.2))
        overlay.add_patch(Polygon(after, fill=False, edgecolor=GREEN, lw=1.15, ls="--"))
        cell_ground = (sample-start) @ rotation
        for x, y in selected["adiada"]:
            tile_px = np.array([[x, y], [x+T, y], [x+T, y+T], [x, y+T]])
            tile_ground = (project(east, north, yaw, k, tile_px, w, h)-start) @ rotation
            overlay.add_patch(Polygon(tile_ground, fill=False, edgecolor=ORANGE, lw=1.4))
        outros, ativos, _, _ = samples["adiada"]
        for pixels, forma, cor, area in [(outros, "o", VISIVEL, 9), (ativos, "D", ENVIO, 20)]:
            xy = (project(east, north, yaw, k, pixels, w, h)-start) @ rotation
            overlay.scatter(*xy.T, marker=forma, c=cor, edgecolors=BORDA, linewidth=.5, s=area)
        overlay.annotate("C", cell_ground, xytext=(8, -9), textcoords="offset points",
                         weight="bold", fontsize=7.5, color="white", ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=.15", fc=BORDA, ec="white", lw=.65),
                         arrowprops=dict(arrowstyle="-", lw=.8, color=BORDA, shrinkB=2))
        # Centro da borda de entrada nos dois instantes. Entre as bordas está a faixa nova.
        leading_edges = [poly[np.argsort(poly[:, 1])[-2:]].mean(axis=0) for poly in (before, after)]
        new_band = np.mean(leading_edges, axis=0)
        overlay.annotate(f"{new_pct:.1f}% de área nova".replace(".", ","), new_band,
                         xytext=(0, 27), ha="center", va="center", fontsize=9.2,
                         arrowprops={"arrowstyle": "-", "lw": 1, "color": GREEN})
        overlay.set_aspect("equal")
        overlay.set(xlim=(-29, 29), ylim=(-23, 29))
        overlay.axis("off")

        for camera, (nome, policy) in zip(cameras, [("Entrada", "imediata"), ("Saída", "adiada")]):
            outros, ativos, pixel, _ = samples[policy]
            painel(camera, image, (w, h), selected[policy], outros, ativos, pixel, "C")
            camera.set_title(f"{nome} da célula · {contagem(len(selected[policy]))}", loc="left", fontsize=9)

        counts_ax.set_title("(c) Recortes selecionados no voo completo", loc="left", pad=8)
        interval = f"Intervalo por voo\n(a cada {matched_step} quadros)"
        counts = {
            "Processamento integral": ntiles * n,
            "Entrada da célula": result["imediata"]["recortes"],
            "Um quadro a cada 30": len(range(0, n, 30)) * ntiles,
            interval: len(range(0, n, matched_step)) * ntiles,
            "Saída da célula": result["adiada"]["recortes"],
        }
        coverage = {
            "Processamento integral": 1.0,  # A união de todos os quadros define a área imageada.
            "Entrada da célula": 1 - result["imediata"]["frac_nao_analisada"],
            "Um quadro a cada 30": 1 - result["skip_30"]["frac_nao_vista"],
            interval: 1 - result[f"skip_{matched_step}"]["frac_nao_vista"],
            "Saída da célula": 1 - result["adiada"]["frac_nao_analisada"],
        }
        for row, (name, count) in enumerate(counts.items()):
            color = {"Entrada da célula": BLUE, "Saída da célula": ORANGE}.get(name, ".44")
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
    print("Recortes do quadro:", selected)
    print("Células que pedem envio:", int(entry_due.sum()), "(entrada);", int(exit_due.sum()), "(saída).")
    print("Totais:", counts)
    print("Cobertura por estratégia:", coverage)


if __name__ == "__main__":
    main()
