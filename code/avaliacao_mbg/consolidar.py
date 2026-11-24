"""Consolida as configurações concluídas e verifica os denominadores pareados."""

import numpy as np
import pandas as pd

from dados import CACHE, OUT, SPLITS, STRATEGIES
from metricas import CLASSES, labels


MOTORS = {
    "trt_fp32": "TensorRT FP32",
    "trt_fp16": "TensorRT FP16",
    "trt_int8": "TensorRT INT8",
    "ort_trt_fp32": "ORT--TensorRT FP32",
    "ort_trt_fp16": "ORT--TensorRT FP16",
    "ort_cuda_fp32": "ORT--CUDA FP32",
    "ort_cpu_fp32": "ORT--CPU FP32",
    "torchscript_fp32": "TorchScript FP32",
    "torchscript_fp16": "TorchScript FP16",
}
SELECTION = {
    "todos": "Processamento integral",
    "imediata": "Entrada da célula",
    "cada30": "Um quadro a cada 30",
    "pareada": "Intervalo por voo",
    "adiada": "Saída da célula",
}


def porcentagem(fracao):
    return f"{100 * fracao:.1f}".replace(".", ",")


def tabelas_tcc(overall):
    rows = overall.set_index(["motor", "estrategia"])
    primary = []
    for strategy, label in SELECTION.items():
        row = rows.loc[("trt_fp16", strategy), :]
        counts = [f"{int(row[key]):,}".replace(",", ".") for key in ("recortes", "vp", "fp", "fn")]
        rates = [porcentagem(row[key]) for key in ("precisao", "recall", "f1", "map50_95")]
        primary.append(" & ".join([label, *counts, *rates]) + r" \\")
    (OUT / "tabela_estrategias.tex").write_text(
        "\\newcommand{\\linhasEstrategiasMBG}{%\n" + "\n".join(primary) + "\n}\n", encoding="utf-8")

    engines = []
    for key, label in MOTORS.items():
        group = rows.loc[key]
        values = [str(int(group.loc[s, "vp"])) for s in STRATEGIES]
        diff = group.loc["adiada", "recall"] - group.loc["pareada", "recall"]
        contrast = f"{100 * diff:+.1f}".replace(".", "{,}")
        engines.append(" & ".join([label, *values, "$" + contrast + "$"]) + r" \\")
    (OUT / "tabela_motores.tex").write_text(
        "\\newcommand{\\linhasMotoresMBG}{%\n" + "\n".join(engines) + "\n}\n", encoding="utf-8")


def oportunidades():
    """Conta trilhas com pelo menos 40% de uma caixa em algum recorte selecionado."""
    rows = []
    for video in SPLITS["voos"]["test"]:
        with np.load(CACHE / f"{video}.npz") as info:
            boxes = labels(video, info)
            origem = info["tiles"]
            masks = info["masks"]
        for trecho, frames in enumerate(np.array_split(np.arange(len(boxes)), 2), 1):
            tracks = {}
            for i in frames:
                for x0, y0, x1, y1, track, cls in boxes[i]:
                    area = (x1 - x0) * (y1 - y0)
                    wh = np.minimum(origem + 640, [x1, y1]) - np.maximum(origem, [x0, y0])
                    elegivel = np.maximum(wh, 0).prod(axis=1) >= .4 * area
                    key = (int(cls), int(track))
                    if key not in tracks:
                        tracks[key] = np.zeros(len(STRATEGIES) + 1, dtype=bool)
                    tracks[key][:-1] |= masks[i, elegivel].any(axis=0)
                    tracks[key][-1] |= i == frames[0] or i == frames[-1]
            for (cls, track), flags in tracks.items():
                rows.append(dict(video=video, trecho=trecho, classe=CLASSES[cls], trilha=track,
                                 **dict(zip(STRATEGIES, map(int, flags[:-1]))), toca_borda=int(flags[-1])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "oportunidades_trechos.csv", index=False)
    return df


def consolidate():
    parts, classes, segments = [], [], []
    for engine in MOTORS:
        parts.append(pd.read_csv(OUT/f"resumo_{engine}.csv"))
        classes.append(pd.read_csv(OUT/f"classes_{engine}.csv"))
        segments.append(pd.read_csv(OUT/f"trechos_{engine}.csv"))
    overall = pd.concat(parts,ignore_index=True)
    strata = pd.concat(segments,ignore_index=True)
    assert set(overall.set_index(["motor", "estrategia"]).index) == {(m,s) for m in MOTORS for s in STRATEGIES}
    assert len(overall)==len(MOTORS)*len(STRATEGIES) and overall.referencias.nunique()==1
    assert (overall.vp+overall.fn==overall.referencias).all()
    assert (overall.groupby("estrategia").recortes.nunique()==1).all()
    assert (strata.groupby(["motor","estrategia"]).size()==12).all()
    assert len(set(strata.video))==6
    for motor, group in overall.groupby("motor"):
        paired = pd.read_csv(OUT / f"pareamento_{motor}.csv")
        assert not paired.duplicated(["video","trecho","classe","trilha"]).any()
        assert len(paired) == group.referencias.iloc[0]
        for row in group.itertuples():
            detected = paired[row.estrategia].astype(bool)
            dense = paired.todos.astype(bool)
            assert detected.sum() == row.vp
            assert (detected & ~dense).sum() == row.ganhos_vs_todos
            assert (dense & ~detected).sum() == row.perdas_vs_todos
    for name, frame in [("qualidade_teste",overall),("qualidade_por_classe",pd.concat(classes))]:
        frame.to_csv(OUT/f"{name}.csv",index=False,float_format="%.8f")
    strata["regiao"]=strata.video.str[-1]
    regional = strata.groupby(["motor","regiao","estrategia"],as_index=False)[
        ["vp","fp","fn","recortes","referencias"]].sum()
    regional["recall"] = regional.vp/regional.referencias
    baseline = regional.loc[regional.estrategia=="todos",["motor","regiao","recall"]].rename(
        columns={"recall":"recall_todos"})
    regional = regional.merge(baseline,on=["motor","regiao"],validate="many_to_one")
    regional["delta_recall_pp"]=100*(regional.recall-regional.recall_todos)
    regional.to_csv(OUT/"qualidade_por_regiao.csv",index=False,float_format="%.8f")
    manifest = []
    for name in SPLITS["voos"]["test"]:
        with np.load(CACHE/f"{name}.npz") as data:
            for win,blocks in enumerate(np.array_split(np.arange(len(data["frames"])),2),1):
                d=dict(video=name,trecho=win,primeiro_quadro=data["frames"][blocks[0]],
                       ultimo_quadro=data["frames"][blocks[-1]],fps=float(data["fps"]),k=int(data["step"]))
                d.update(dict(zip(STRATEGIES,data["masks"][blocks].sum(axis=(0,1)))))
                manifest.append(d)
    pd.DataFrame(manifest).to_csv(OUT/"trechos_avaliados.csv",index=False)
    opportunities = oportunidades()
    assert len(opportunities) == overall.referencias.iloc[0]
    tabelas_tcc(overall)
    print(overall.to_string(index=False))


if __name__=="__main__":
    consolidate()
