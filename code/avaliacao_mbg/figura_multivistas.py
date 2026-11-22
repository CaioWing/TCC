"""Figura de qualidade e demanda em TensorRT FP16, a partir dos CSV consolidados."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dados import OUT, ROOT
from multivistas import BENCHMARK, DIR, METAS
from paleta_tcc import BLUE, GREEN, ORANGE


def desenhar():
    q = pd.read_csv(DIR / "qualidade.csv").query("motor=='trt_fp16'")
    costs = pd.read_csv(DIR / "custo_e_cobertura_por_voo.csv")
    bench = pd.read_csv(BENCHMARK).set_index(["config","mode"]).loc[("trt_fp16","e2e")]
    full = pd.read_csv(OUT / "resumo_trt_fp16.csv").set_index("estrategia").loc["todos"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
                         "axes.spines.right":False,"axes.spines.top":False,"axes.axisbelow":True})
    fig,(quality,capacity) = plt.subplots(1,2,figsize=(6.25,3.0))
    for ax in (quality,capacity):
        ax.set_xticks(METAS)
        ax.set_xlim(.7,5.3)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y",alpha=.21)
        ax.set_xlabel("Meta de vistas por célula")
    quality.set_title("(a) Detecção nos trechos de teste",fontsize=8.5)
    quality.set_ylabel("Recipientes reconhecidos (de 137)")
    quality.set_ylim(0,121)
    quality.set_yticks([0,25,50,75,100])
    quality.axhline(full.vp,color="#55595c",ls="--",lw=.8)
    quality.text(.8,full.vp+2.4,f"Processamento integral ({int(full.vp)})",fontsize=7,color="#424649")
    for method,label,color,marker,shift in (("geo","Seleção na saída",BLUE,"o",-10),
                                            ("tempo","Intervalo ajustado por voo",ORANGE,"s",4)):
        g = q[q.metodo==method].sort_values("k")
        quality.plot(g.k,g.vp,marker=marker,color=color,label=label,lw=1.45,ms=4)
        for x,y in zip(g.k,g.vp):
            quality.annotate(str(y),(x,y),xytext=(0,shift),textcoords="offset points",
                             ha="center",color=color,fontsize=6.8)
    quality.legend(loc="lower right",frameon=False,fontsize=6.7)
    capacity.set_title("(b) Demanda da seleção na saída",fontsize=8.5)
    capacity.set_ylabel("Recortes por segundo")
    capacity.set_ylim(0,52)
    capacity.set_yticks([0,10,20,30,40,50])
    capacity.axhline(bench.fps,color="#55595c",ls="--",lw=.8)
    limit = f"{bench.fps:.2f}".replace(".",",")
    capacity.text(.8,bench.fps+1.1,f"Jetson FP16 em bancada ({limit})",fontsize=7,color="#424649")
    groups = costs.groupby("k")
    avg = groups.recortes_geo.sum() / costs.assign(segundos=costs.quadros/costs.fps).groupby("k").segundos.sum()
    interval = np.stack([avg-groups.taxa_geo.min(),groups.taxa_geo.max()-avg])
    capacity.errorbar(avg.index,avg,yerr=interval,color=GREEN,fmt="o-",capsize=3,lw=1.5,ms=4)
    for x,y in avg.items():
        capacity.annotate(f"{y:.1f}".replace(".",","),(x,groups.taxa_geo.min().loc[x]),
                          xytext=(0,-10),textcoords="offset points",ha="center",color="#455e48",fontsize=7)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(ROOT / "doc/figuras/fig_multivistas.pdf",bbox_inches="tight",pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    desenhar()
