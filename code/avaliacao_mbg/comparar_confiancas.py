"""Compara os maiores escores de caixas corretas nas mesmas trilhas, em TensorRT."""
import json

import pandas as pd

from dados import OUT, SPLITS
from metricas import CLASSES, frame_metrics, load_clips, selected


def comparar():
    conf = json.loads((OUT / "protocolo.json").read_text(encoding="utf-8"))["conf"]
    rows = []
    for engine in ("trt_fp16","trt_int8"):
        for video in SPLITS["voos"]["test"]:
            for clip in load_clips(video,engine):
                scores = {}
                frame_metrics(clip, selected(clip,"todos"), conf, track_scores=scores)
                for cls, ident in clip["centers"]:
                    rows.append(dict(motor=engine, video=video, trecho=clip["clip"], classe=CLASSES[cls],
                                     trilha=ident, maior_escore=scores.get((cls,ident))))
    paired = pd.DataFrame(rows).pivot(index=["video","trecho","classe","trilha"],columns="motor",
                                      values="maior_escore").reset_index()
    paired.to_csv(OUT / "confianca_pareada_tensorrt.csv", index=False, float_format="%.8f")
    for engine in ("trt_fp16","trt_int8"):
        print(engine,"caixa correta:",paired[engine].count(),"confiança suficiente:",(paired[engine]>=conf).sum())
    below = paired[(paired.trt_fp16>=conf) & ~(paired.trt_int8>=conf)]
    print("Apenas FP16 acima do limiar:",len(below),"com caixa correspondente abaixo do limiar em INT8:",
          below.trt_int8.notna().sum())
    print("Apenas INT8 acima do limiar:",((paired.trt_int8>=conf) & ~(paired.trt_fp16>=conf)).sum())


if __name__=="__main__":
    comparar()
