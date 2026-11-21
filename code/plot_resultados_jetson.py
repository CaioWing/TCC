from pathlib import Path
import csv,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from paleta_tcc import BLUE,ORANGE,GREEN,PURPLE,RED,GRAY
with (ROOT/'data/yolo11n-jetson24-20261116/summary.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
def get(c,m='e2e'): return next(r for r in rows if r['config']==c and r['mode']==m)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
def save(fig,name):
 fig.savefig(ROOT/f'doc/figuras/{name}.pdf');fig.savefig(ROOT/f'tmp/jetson-review/{name}.png',dpi=220);plt.close(fig)
cfg=[('trt_fp32','TensorRT FP32',BLUE),('trt_fp16','TensorRT FP16',ORANGE),('trt_int8','TensorRT INT8',GREEN),('ort_trt_fp32','ORT TensorRT FP32',PURPLE),('ort_trt_fp16','ORT TensorRT FP16',RED),('ort_cuda_fp32','ORT CUDA FP32',GRAY),('ort_cpu_fp32','ORT CPU FP32',GRAY),('torchscript_fp32','TorchScript FP32',GRAY),('torchscript_fp16','TorchScript FP16',GRAY)]
fig,ax=plt.subplots(figsize=(5.9,3.5))
off={1:(-10,-12),2:(-10,7),3:(-8,8),4:(5,7),5:(7,-10),6:(5,7),7:(4,7),8:(-12,6),9:(6,-10)}
for i,(c,label,color) in enumerate(cfg,1):
 r=get(c);x=float(r['lat_mean_ms']);y=float(r['E_total_mJ_per_inf'])
 ax.scatter(x,y,s=47,color=color,edgecolor='#343b40',linewidth=.6,zorder=3);ax.annotate(str(i),(x,y),xytext=off[i],textcoords='offset points',fontsize=7,weight='bold')
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(17,160);ax.set_ylim(105,1300)
for axis,ticks in [(ax.xaxis,[20,30,40,60,100,150]),(ax.yaxis,[120,150,200,300,500,1000])]:
 axis.set_major_locator(FixedLocator(ticks));axis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'));axis.set_minor_locator(NullLocator())
ax.set_xlabel('Latência média e2e (ms/quadro)');ax.set_ylabel('Energia total (mJ/inferência)');ax.grid(alpha=.2)
ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=col,markeredgecolor='#343b40',markersize=5,label=f'{i} {label}') for i,(_,label,col) in enumerate(cfg,1)],bbox_to_anchor=(1.01,1),loc='upper left',frameon=False,fontsize=6.7,handletextpad=.3,borderaxespad=0)
fig.tight_layout();save(fig,'fig_jetson_configuracoes')
ids=['trt_fp32','trt_fp16','trt_int8'];labels=['FP32','FP16','INT8'];colors=[BLUE,ORANGE,GREEN]
fig,axes=plt.subplots(1,2,figsize=(5.9,2.8))
for ax,vals in zip(axes,[[100*float(get(c)['mAP50_95']) for c in ids],[float(get(c)['E_total_mJ_per_inf']) for c in ids]]):
 bars=ax.bar(labels,vals,color=colors,width=.6);ax.set_ylim(0,max(vals)*1.22);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.025,f'{v:.1f}'.replace('.',','),ha='center',fontsize=7)
axes[0].set_title('(a) Qualidade da detecção',fontsize=8.5)
axes[1].set_title('(b) Custo por inferência',fontsize=8.5)
axes[0].set_ylabel('mAP50–95 do benchmark (%)');axes[1].set_ylabel('Energia total e2e (mJ/inferência)');fig.tight_layout(w_pad=2);save(fig,'fig_jetson_precisao')
with (ROOT/'data/mbg_v2/splits/memoria_sim_26voos.csv').open(encoding='utf-8',newline='') as f: flights=list(csv.DictReader(f))
bm=[r for r in flights if r['region'] != 'A']; a=[r for r in flights if r['region']=='A']
def stats(group,key):
 v=np.array([float(r[key]) for r in group]);return float(v.min()),float(np.median(v)),float(v.max())
raw=[stats(bm,'adiada_recortes_s'),stats(a,'adiada_recortes_s'),(24,24,24),stats(bm,'imediata_recortes_s'),(719,719,719)]
print('Demanda mínima, mediana, máxima (passadas/s):',raw)
cap=float(get('trt_fp16')['fps']); print('Capacidade FP16:',cap)
fig,ax=plt.subplots(figsize=(5.9,3.0));y=np.arange(5)
labels=['Memória adiada (B–M)','Memória adiada (A)','Amostragem: 1 quadro/s (B–M)','Memória imediata (B–M)','Todos os quadros (B–M)']
for yi,(lo,med,hi),color in zip(y,raw,[BLUE,BLUE,ORANGE,GREEN,RED]):
 ax.plot([lo/cap,hi/cap],[yi,yi],color=color,lw=2.5,zorder=2);ax.scatter(med/cap,yi,color=color,edgecolor='#343b40',s=35,linewidth=.5,zorder=3)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(.09,25);ax.axvline(1,color='#343b40',lw=1,ls='--');ax.set_xlabel('Demanda / capacidade FP16 (vezes)')
ax.xaxis.set_major_locator(FixedLocator([.1,.2,.5,1,2,5,10,20]));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'.replace('.',',')));ax.xaxis.set_minor_locator(NullLocator());ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'fig_jetson_demanda')
