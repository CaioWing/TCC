"""Prepara trechos pareados sem consultar as caixas das regiões de teste."""
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from memoria_sim import load_pose, simulate, tiles_grid

ROOT = CODE.parent
DATA = ROOT / "data/mbg_v2"
CACHE = ROOT / "tmp/avaliacao_mbg"
OUT = ROOT / "doc/dados/qualidade_memoria"
SPLITS = json.loads((DATA / "splits/splits.json").read_text(encoding="utf-8"))
GSD = json.loads((ROOT / "doc/dados/ajuste_gsd_voo03/metadados.json").read_text(encoding="utf-8"))["gsd_cm"] / 100
STRATEGIES = ["todos", "imediata", "cada30", "pareada", "adiada"]
WEIGHTS = ROOT / "runs/mbg_y11n_640_v1/weights/best.pt"


def folder(video):
    flight, region = video.removeprefix("video").split("_region")
    return DATA / "raw" / f"flight-mbg-v2-{flight}-region{region}"


def prepare(video):
    video_dir = folder(video)
    source, = (video_dir / "avi").glob("*.avi")
    log, = (video_dir / "log").glob("*.csv")
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=avg_frame_rate,nb_frames,width,height", "-of", "json", str(source)
    ]))["streams"][0]
    fps = float(Fraction(stream["avg_frame_rate"]))
    n, width, height = (int(stream[k]) for k in ["nb_frames", "width", "height"])
    tiles = tiles_grid(width, height)
    index = {tile: i for i, tile in enumerate(tiles)}
    length = round(4 * fps)
    starts = [round(f * n / 3) - length // 2 for f in (1, 2)]
    frames = np.concatenate([np.arange(start, start + length) for start in starts])
    frame_index = {frame: i for i, frame in enumerate(frames)}
    masks = np.zeros((len(frames), len(tiles), len(STRATEGIES)), dtype=bool)

    def record(policy, frame, x, y):
        if frame in frame_index:
            masks[frame_index[frame], index[x, y], 1 if policy == "imediata" else 4] = True

    result = simulate(log, n, fps, width, height, GSD, on_selected=record)
    step = round(len(tiles) * n / result["adiada"]["recortes"])
    masks[:, :, 0] = True
    masks[frames % 30 == 0, :, 2] = True
    masks[frames % step == 0, :, 3] = True
    pose = np.column_stack(load_pose(log, n, fps))[frames]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{video}.npz", frames=frames, masks=masks, pose=pose,
                        starts=starts, fps=fps, step=step, tiles=tiles, gsd=GSD,
                        size=[width, height], total_frames=n)
    print(video, "frames", starts, "step", step, "tiles", masks.sum(axis=(0, 1)).tolist(), flush=True)


if __name__ == "__main__":
    for split in ("val", "test"):
        for name in SPLITS["voos"][split]:
            prepare(name)
