"""Compara as mesmas previsões das nove configurações sob as novas seleções geográficas."""
import argparse
import json

import numpy as np
import pandas as pd

from avaliar import ratios
from consolidar import MOTORS
from dados import OUT, SPLITS
from metricas import CLASSES, ap_class, focus_counts, frame_metrics, load_clips, projected, selected_tiles
from multivistas import DIR, METAS, TEMP


def score(engine):
    protocol = json.loads((OUT / "protocolo.json").read_text(encoding="utf-8"))
    conf, radius, dmax = (protocol[key] for key in ("conf","raio_m","dmax_m"))
    methods = [(kind,k) for k in METAS for kind in ("geo","tempo")]
    aps = {key:[[] for _ in CLASSES] for key in methods}
    rows, paired = [], []
    for video in SPLITS["voos"]["test"]:
        with np.load(TEMP / f"{video}.npz") as data:
            masks = dict(data)
        for clip in load_clips(video,engine):
            indices = np.searchsorted(masks["frames"],clip["frames"])
            assert np.array_equal(masks["frames"][indices],clip["frames"])
            hits = {key:dict(video=video,trecho=clip["clip"],classe=CLASSES[key[0]],trilha=key[1])
                    for key in clip["centers"]}
            for kind,k in methods:
                mask = masks[f"{kind}{k}"][indices]
                predictions = selected_tiles(clip,mask)
                counts,matched = focus_counts(clip,projected(clip,predictions),conf,radius,dmax,return_matches=True)
                ap = frame_metrics(clip,predictions,conf)
                for c, item in enumerate(ap):
                    aps[kind,k][c].append(item)
                for key in hits:
                    hits[key][f"{kind}{k}"] = int(key in matched)
                rows.append(dict(motor=engine,metodo=kind,k=k,video=video,trecho=clip["clip"],
                                 recortes=int(mask.sum()),trilhas_iou=sum(item[3] for item in ap),
                                 **ratios(counts.sum(axis=0))))
            paired.extend(hits.values())
    segment = pd.DataFrame(rows)
    hit = pd.DataFrame(paired)
    summary = []
    original = pd.read_csv(OUT / f"resumo_{engine}.csv").set_index("estrategia")
    for (kind,k),group in segment.groupby(["metodo","k"],sort=False):
        rates = ratios(group[["vp","fp","fn"]].sum())
        ap = np.stack([ap_class(rs) for rs in aps[kind,k]])
        detected = hit[f"{kind}{k}"].astype(bool)
        one = hit.geo1.astype(bool)
        assert rates["referencias"] == len(hit) and detected.sum() == rates["vp"]
        row = dict(motor=engine,metodo=kind,k=k,recortes=group.recortes.sum(),**rates,
                   recall_trilha_iou=group.trilhas_iou.sum()/len(hit),map50=ap[:,0].mean(),map50_95=ap.mean(),
                   ganhos_sobre_geo1=(detected & ~one).sum(),perdas_sobre_geo1=(~detected & one).sum(),
                   delta_recall_integral_pp=100*(rates["recall"]-original.loc["todos","recall"]))
        if k == 1:
            ref = original.loc["adiada" if kind=="geo" else "pareada"]
            for metric in ("vp","fp","fn","recortes","f1","map50","map50_95"):
                assert np.isclose(row[metric],ref[metric],atol=1e-7), (engine,metric)
        summary.append(row)
    for name,df in (("resumo",pd.DataFrame(summary)),("trechos",segment),("trilhas",hit)):
        df.to_csv(TEMP / f"{name}_{engine}.csv",index=False,float_format="%.8f")
    print(engine,[(r["k"],r["vp"]) for r in summary if r["metodo"]=="geo"],flush=True)


def consolidate():
    flights = pd.concat([pd.read_csv(TEMP / f"{v}.csv") for v in SPLITS["voos"]["test"]],ignore_index=True)
    flights.to_csv(DIR / "custo_e_cobertura_por_voo.csv",index=False,float_format="%.8f")
    for source,name in (("resumo","qualidade"),("trechos","qualidade_por_trecho"),("trilhas","pareamento")):
        frames = [pd.read_csv(TEMP / f"{source}_{m}.csv").assign(motor=m) for m in MOTORS]
        combined = pd.concat(frames,ignore_index=True)
        if source=="resumo":
            assert len(combined)==2*len(METAS)*len(MOTORS)
            assert (combined.referencias==137).all()
            assert (combined.groupby(["metodo","k"]).recortes.nunique()==1).all()
        combined.to_csv(DIR / f"{name}.csv",index=False,float_format="%.8f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("motores",nargs="+",choices=[*MOTORS,"consolidar"])
    for motor in parser.parse_args().motores:
        if motor=="consolidar":
            consolidate()
        else:
            score(motor)
