"""Exporta os pesos escolhidos na validação, com entrada fixa e NMS externo comum."""
import csv
import hashlib
import json
import os
import shutil

from dados import CACHE, OUT, ROOT, WEIGHTS

os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
from ultralytics import YOLO


def registrar_origem():
    files = [WEIGHTS, CACHE / "detector.pt", CACHE / "detector.torchscript", CACHE / "detector.onnx"]
    digests = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if len(set(list(digests.values())[:2])) != 1:
        raise ValueError("O arquivo exportado difere do checkpoint escolhido.")
    with (WEIGHTS.parents[1] / "results.csv").open(encoding="utf-8", newline="") as f:
        best = max(csv.DictReader(f), key=lambda row: float(row["metrics/mAP50-95(B)"]))
    origin = dict(epoca=int(best["epoch"]), map50_95_validacao_12_classes=float(best["metrics/mAP50-95(B)"]),
                  sha256=digests, entrada=[1,3,640,640], opset_onnx=17)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "origem_modelo.json").write_text(json.dumps(origin, indent=2), encoding="utf-8")


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / "detector.pt"
    shutil.copyfile(WEIGHTS, local)
    for fmt in ("torchscript", "onnx"):
        options = {"opset": 17, "simplify": False} if fmt == "onnx" else {}
        YOLO(local).export(format=fmt, imgsz=640, batch=1, device="cpu", nms=False, **options)
    registrar_origem()
