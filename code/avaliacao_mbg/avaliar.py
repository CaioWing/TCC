"""Fixa associação em F/I e mede os 12 trechos pareados de E/H/L."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dados import CACHE, GSD, OUT, SPLITS, STRATEGIES, WEIGHTS
from metricas import CLASSES, ap_class, focus_counts, frame_metrics, load_clips, projected, selected


def ratios(counts):
    tp,fp,fn = map(int,counts)
    return dict(vp=tp,fp=fp,fn=fn,referencias=tp+fn,
                precisao=tp/(tp+fp) if tp+fp else np.nan,
                recall=tp/(tp+fn) if tp+fn else np.nan,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan)


def atomic_csv(rows, name):
    frame = pd.DataFrame(rows)
    tmp = OUT / (name+".partial")
    frame.to_csv(tmp,index=False,float_format="%.8f")
    tmp.replace(OUT/name)


def calibrate():
    clips = [clip for video in SPLITS["voos"]["val"] for clip in load_clips(video,"torchscript_fp32")]
    # Tolerância geográfica a partir da dispersão dos centros anotados na validação.
    dmax = max(.5, np.ceil(np.percentile(np.concatenate([c["scatter"] for c in clips]),95)*4)/4)
    candidates = [(conf, radius) for conf in (.15,.25,.35,.50) for radius in (.5,1.,1.5,2.)]
    tally = {p:np.zeros((len(STRATEGIES),3),dtype=int) for p in candidates}
    for clip in clips:
        for s,strategy in enumerate(STRATEGIES):
            projections = projected(clip, selected(clip,strategy))
            for conf,radius in candidates:
                tally[conf,radius][s] += focus_counts(clip,projections,conf,radius,dmax).sum(axis=0)
    val = [dict(conf=conf, raio_m=radius, f1_medio=np.mean([ratios(c)["f1"] for c in counts]))
           for (conf,radius),counts in tally.items()]
    winner = max(val,key=lambda x:(x["f1_medio"],-x["raio_m"],x["conf"]))
    atomic_csv(val,"calibracao_validacao.csv")
    protocol = dict(**winner, dmax_m=dmax, gsd_m=GSD, janela_s=4, centros_fracao=[1/3,2/3],
                    iou_nms=.7, conf_ap=.001, margem_descritiva_recall_pp=5,
                    pesos_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
                    classes=CLASSES, validacao=SPLITS["regioes"]["val"], teste=SPLITS["regioes"]["test"])
    (OUT/"protocolo.json").write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding="utf-8")
    print(protocol,flush=True)


def evaluate(engine):
    protocol = json.loads((OUT/"protocolo.json").read_text(encoding="utf-8"))
    confidence, radius, dmax = (protocol[k] for k in ("conf","raio_m","dmax_m"))
    rows, details, class_rows, hits = [], [], [], []
    aps = {s:[[] for _ in CLASSES] for s in STRATEGIES}
    for video in SPLITS["voos"]["test"]:
        for clip in load_clips(video,engine):
            matched_by_strategy = {}
            for s,strategy in enumerate(STRATEGIES):
                det = selected(clip,strategy)
                projected_det = projected(clip,det)
                counts, matched_by_strategy[strategy] = focus_counts(
                    clip,projected_det,confidence,radius,dmax,return_matches=True)
                ap = frame_metrics(clip,det,confidence)
                inputs = int(clip["masks"][:,:,s].sum())
                for cls,name in enumerate(CLASSES):
                    aps[strategy][cls].append(ap[cls])
                    details.append(dict(motor=engine,estrategia=strategy,video=video,trecho=clip["clip"],
                                        quadro_inicial=clip["frames"][0],quadro_final=clip["frames"][-1],
                                        classe=name,**ratios(counts[cls]),trilhas_iou=ap[cls][3]))
                rows.append(dict(motor=engine,estrategia=strategy,video=video,trecho=clip["clip"],
                                 recortes=inputs,**ratios(counts.sum(axis=0)),
                                 trilhas_iou=sum(a[3] for a in ap)))
            for key in sorted(clip["centers"]):
                hits.append(dict(video=video,trecho=clip["clip"],classe=CLASSES[key[0]],trilha=key[1],
                                 **{s:int(key in ids) for s,ids in matched_by_strategy.items()}))
    atomic_csv(details,f"trechos_classes_{engine}.csv")
    atomic_csv(rows,f"trechos_{engine}.csv")
    atomic_csv(hits,f"pareamento_{engine}.csv")
    df = pd.DataFrame(rows)
    per_class = pd.DataFrame(details)
    summary=[]
    for strategy in STRATEGIES:
        values = np.vstack([ap_class(rs) for rs in aps[strategy]])
        group=df[df.estrategia==strategy]
        s=ratios(group[["vp","fp","fn"]].sum())
        summary.append(dict(motor=engine,estrategia=strategy,recortes=group.recortes.sum(),
                            **s,recall_trilha_iou=group.trilhas_iou.sum()/s["referencias"],
                            map50=np.nanmean(values[:,0]),map50_95=np.nanmean(values)))
        for cls,name in enumerate(CLASSES):
            g=per_class[(per_class.estrategia==strategy)&(per_class.classe==name)]
            class_rows.append(dict(motor=engine,estrategia=strategy,classe=name,
                                   **ratios(g[["vp","fp","fn"]].sum()),
                                   map50=values[cls,0],map50_95=values[cls].mean()))
    baseline = summary[0]
    for r in summary:
        for m in ("recall","precisao","f1","map50","map50_95"):
            r["delta_"+m+"_pp"]=100*(r[m]-baseline[m])
        strategy=r["estrategia"]
        r["ganhos_vs_todos"]=sum(h[strategy] and not h["todos"] for h in hits)
        r["perdas_vs_todos"]=sum(h["todos"] and not h[strategy] for h in hits)
    atomic_csv(class_rows,f"classes_{engine}.csv")
    atomic_csv(summary,f"resumo_{engine}.csv")
    print(pd.DataFrame(summary)[["motor","estrategia","recortes","vp","fp","fn","f1","recall","map50_95"]].to_string(index=False),flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("etapa",choices=["calibrar","teste"])
    ap.add_argument("--motor",default="torchscript_fp32")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.etapa=="calibrar":
        calibrate()
    else:
        evaluate(args.motor)
