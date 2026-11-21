"""Sensibilidade em FP16 à fase temporal, igualando o orçamento nos trechos de teste."""
import json

import numpy as np
import pandas as pd

from avaliar import ratios
from dados import OUT, SPLITS
from metricas import focus_counts, load_clips, projected, selected_tiles
from multivistas import DIR, METAS, TEMP


def comparar():
    protocol = json.loads((OUT / "protocolo.json").read_text(encoding="utf-8"))
    params = [protocol[key] for key in ("conf","raio_m","dmax_m")]
    quotas = []
    for video in SPLITS["voos"]["test"]:
        with np.load(TEMP / f"{video}.npz") as masks:
            for trecho,frames in enumerate(np.array_split(np.arange(len(masks["frames"])),2),1):
                for k in METAS:
                    geo = masks[f"geo{k}"][frames]
                    quotas.append(dict(video=video,trecho=trecho,k=k,quadros_ideais=geo.sum()/geo.shape[1]))
    df = pd.DataFrame(quotas)
    # Quadros inteiros. Distribui o arredondamento pelos maiores restos dos doze trechos.
    df["quadros_controle"] = np.floor(df.quadros_ideais).astype(int)
    for _,part in df.groupby("k"):
        extra = round(part.quadros_ideais.sum())-part.quadros_controle.sum()
        priority = (part.quadros_ideais-part.quadros_controle).sort_values(ascending=False,kind="stable")
        df.loc[priority.index[:extra],"quadros_controle"] += 1
    df.to_csv(DIR / "orcamento_temporal_por_trecho.csv",index=False)
    rows = []
    for video in SPLITS["voos"]["test"]:
        for clip in load_clips(video,"trt_fp16"):
            part = df[(df.video==video)&(df.trecho==clip["clip"])]
            duration,ntiles = clip["masks"].shape[:2]
            for row in part.itertuples():
                n = row.quadros_controle
                for phase in (0.,.25,.5,.75):
                    mask = np.zeros((duration,ntiles),dtype=bool)
                    if n:
                        indices = np.floor((np.arange(n)+phase)*duration/n).astype(int)
                        mask[indices] = True
                    assert mask.sum() == n*ntiles
                    counts = focus_counts(clip,projected(clip,selected_tiles(clip,mask)),*params)
                    rows.append(dict(k=row.k,fase=phase,video=video,trecho=clip["clip"],recortes=int(mask.sum()),
                                     **ratios(counts.sum(axis=0))))
    segment = pd.DataFrame(rows)
    summary = [dict(k=k,fase=fase,recortes=g.recortes.sum(),**ratios(g[["vp","fp","fn"]].sum()))
               for (k,fase),g in segment.groupby(["k","fase"])]
    pd.DataFrame(summary).to_csv(DIR / "controle_fases_fp16.csv",index=False,float_format="%.8f")


if __name__ == "__main__":
    comparar()
