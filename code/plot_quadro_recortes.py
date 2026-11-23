"""Mostra um quadro real da MBG-V2 e um recorte da grade usada pelo detector."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from PIL import Image

from build_yolo_tiles import T, grid
from paleta_tcc import ORANGE


ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "data/mbg_v2/raw/flight-mbg-v2-15-regionH/frames/frame_05700.png"
OUTPUT = ROOT / "doc/figuras/fig_quadro_recortes.pdf"


def main():
    with Image.open(FRAME) as source:
        frame = source.convert("RGB")
    width, height = frame.size
    origins = grid(width, height)
    selected = (2560, 507)  # Quinta coluna, segunda fileira do quadro 5700.
    assert selected in origins
    x, y = selected
    crop = frame.crop((x, y, x + T, y + T))

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        fig, (overview, detail) = plt.subplots(
            1, 2, figsize=(7.3, 2.85), gridspec_kw={"width_ratios": [width / height, 1]})
        fig.subplots_adjust(left=.008, right=.992, bottom=.025, top=.87, wspace=.055)
        # Apenas a miniatura do quadro é reduzida para impressão. A grade usa os pixels originais.
        preview = frame.resize((1536, 864), Image.Resampling.LANCZOS)
        overview.imshow(preview, extent=(0, width, height, 0), interpolation="none")
        for left, top in origins:
            edge = Rectangle((left, top), T, T, fill=False, edgecolor="white", linewidth=.55)
            edge.set_path_effects([patheffects.withStroke(linewidth=1.25, foreground=".20")])
            overview.add_patch(edge)
        selected_edge = Rectangle(selected, T, T, fill=False, edgecolor=ORANGE, linewidth=2.4)
        selected_edge.set_path_effects([patheffects.withStroke(linewidth=3.8, foreground="white")])
        overview.add_patch(selected_edge)
        overview.set(xlim=(0, width), ylim=(height, 0))

        detail.imshow(crop, interpolation="none")
        for axis, title in ((overview, "(a) Quadro original e grade de recortes"),
                            (detail, "(b) Recorte de 640 × 640 pixels")):
            axis.set_title(title, loc="left", fontsize=10, pad=5)
            axis.set_axis_off()
        fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=.04)
        plt.close(fig)


if __name__ == "__main__":
    main()
