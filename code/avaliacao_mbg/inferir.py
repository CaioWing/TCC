"""Executa o detector uma vez por recorte; as cinco seleções reutilizam a mesma saída."""
import argparse
import time

import cv2
import numpy as np
import torch
from filelock import FileLock

from dados import CACHE, SPLITS, folder
from motores import ENGINES, blob, postprocess, runner, setup


def infer_video(video, engine, run, *, cache=CACHE, sparse=False, leitura_continua=False):
    dst = cache / engine / f"{video}.npz"
    if dst.exists():
        return
    dst.parent.mkdir(exist_ok=True)
    with np.load(cache / f"{video}.npz") as info:
        frames, tiles = info["frames"], info["tiles"]
        requested = info["requested"] if sparse else None
    source, = (folder(video) / "avi").glob("*.avi")
    cam = cv2.VideoCapture(str(source))
    codec = int(cam.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
    sequencial = leitura_continua and codec != b"MJPG"
    predictions, offsets = [], [0]
    t = time.monotonic()
    try:
        for i, frame in enumerate(frames):
            if sequencial:
                anterior = frames[i-1] if i else -1
                for _ in range(anterior + 1, frame):
                    if not cam.grab():
                        raise IOError(f"Vídeo interrompido antes do quadro {frame}: {video}")
            elif i == 0 or frame != frames[i-1]+1:
                cam.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cam.read()
            if not ok or round(cam.get(cv2.CAP_PROP_POS_FRAMES)) != frame+1:
                raise RuntimeError(f"Falha ao decodificar {video} quadro {frame}")
            for j, (x, y) in enumerate(tiles):
                if sparse and not requested[i, j]:
                    offsets.append(offsets[-1])
                    continue
                pred = postprocess(run(blob(image[y:y+640, x:x+640])))
                predictions.append(pred)
                offsets.append(offsets[-1]+len(pred))
    finally:
        cam.release()
    tmp = dst.with_suffix(".partial.npz")
    extra = {"computed": requested} if sparse else {}
    detections = np.concatenate(predictions) if predictions else np.empty((0, 6), dtype=np.float32)
    np.savez_compressed(tmp, det=detections, offsets=offsets, frames=frames, **extra)
    tmp.replace(dst)
    print(engine, video, len(predictions), "recortes", round(time.monotonic()-t, 1), "s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=ENGINES)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    args = ap.parse_args()
    setup()
    with FileLock(str(CACHE / f"inferir_{args.engine}_{args.split}.lock")), torch.inference_mode():
        completed = all((CACHE / args.engine / f"{video}.npz").is_file()
                        for video in SPLITS["voos"][args.split])
        if completed:
            raise SystemExit(0)
        run = runner(args.engine, verificar_provedor=args.engine.startswith("ort_"))
        for video in SPLITS["voos"][args.split]:
            infer_video(video, args.engine, run)
