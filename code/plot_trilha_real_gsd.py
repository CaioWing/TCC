"""Exemplo real da projeção de uma trilha do voo 03, sem novo ajuste de GSD.

Seleciona, entre caixas-d'água em trechos com pouca variação de guinada,
a trilha cuja dispersão é mais próxima da mediana do voo. A seleção não
minimiza a dispersão individual. Os dados e o critério ficam registrados.
"""
from paleta_tcc import BLUE, ORANGE, GREEN
import csv
import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from plot_ajuste_gsd import ROOT, load_tracks, np

OUT = ROOT / 'doc/dados/trilha_real_gsd_voo03'
COLORS = [ORANGE, BLUE, GREEN]


def main():
    tracks, metadata, details = load_tracks(with_details=True)
    calibration = json.loads((ROOT/'doc/dados/ajuste_gsd_voo03/metadados.json').read_text(encoding='utf-8'))
    assert metadata['source_sha256'] == calibration['source_sha256'], 'Fontes do ajuste mudaram'
    gsd = calibration['gsd_cm']/100
    dispersions = []
    for aircraft, offsets in tracks:
        projected = aircraft + gsd*offsets
        dispersions.append(np.median(np.linalg.norm(projected-np.median(projected, axis=0), axis=1)))
    target = np.median(dispersions)
    candidates = []
    for i, ((aircraft, offsets), d) in enumerate(zip(tracks, details)):
        sample = [0, len(aircraft)//2, len(aircraft)-1]
        duration = d['times'][-1]-d['times'][0]
        travel = np.linalg.norm(aircraft[-1]-aircraft[0])
        yaw_range = np.rad2deg(np.ptp(d['yaw']))
        if (d['label']=='watertank' and travel>=15 and 3<=duration<=12
                and yaw_range<=3 and not d['occluded'][sample].any()):
            # A mediana de um número par de trilhas pode deixar duas igualmente próximas.
            candidates.append((round(abs(dispersions[i]-target), 9), d['track_id'], i))
    _, _, selected = min(candidates)
    aircraft, offsets = tracks[selected]
    d = details[selected]
    origin = aircraft[0].copy()
    aircraft = aircraft-origin
    projected = aircraft+gsd*offsets
    sample = [0, len(aircraft)//2, len(aircraft)-1]
    uv = d['uv']
    metadata.update(track_id=d['track_id'], label=d['label'],
                    gsd_cm=100*gsd, track_samples=len(aircraft),
                    selected_frames=d['frames'][sample].tolist(),
                    selection='watertank; net aircraft displacement >=15m; duration 3..12s; yaw range <=3 degrees; three samples unoccluded; dispersion closest to flight median (distance rounded to 1e-9 m); tie by track id',
                    eligible_tracks=len(candidates),
                    plot_origin='aircraft position at first sampled frame of selected track',
                    plot_origin_flight_ne_m=origin.tolist(),
                    track_dispersion_m=dispersions[selected],
                    duration_s=float(d['times'][-1]-d['times'][0]),
                    aircraft_net_displacement_m=float(np.linalg.norm(aircraft[-1])),
                    image_net_displacement_px=float(np.linalg.norm(uv[-1]-uv[0])),
                    projected_net_displacement_m=float(np.linalg.norm(projected[-1]-projected[0])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'metadados.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    with (OUT/'posicoes.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['frame','time_video_s','time_relative_s','u_px','v_px','yaw_deg',
                         'aircraft_north_m','aircraft_east_m','offset_north_m','offset_east_m',
                         'projected_north_m','projected_east_m','highlight'])
        for i in range(len(aircraft)):
            writer.writerow([int(d['frames'][i]),d['times'][i],d['times'][i]-d['times'][0],
                             *uv[i],np.rad2deg(d['yaw'][i]),*aircraft[i],*(gsd*offsets[i]),
                             *projected[i],sample.index(i)+1 if i in sample else ''])
    fmt=FuncFormatter(lambda value,_: f'{value:g}'.replace('.',','))
    zoom_x = (np.floor(projected[:,1].min()-.6), np.ceil(projected[:,1].max()+.6))
    zoom_y = (np.floor(2*(projected[:,0].min()-.5))/2,
              np.ceil(2*(projected[:,0].max()+.5))/2)
    with plt.rc_context({'font.size':11, 'pdf.fonttype':42}):
        fig=plt.figure(figsize=(10.2,6.4))
        grid=fig.add_gridspec(2,2, width_ratios=[1,1.1], left=.09,right=.98,
                             bottom=.1,top=.94,hspace=.6,wspace=.38)
        image_ax=fig.add_subplot(grid[0,0])
        map_ax=fig.add_subplot(grid[:,1])
        detail_ax=fig.add_subplot(grid[1,0])
        for ax in [image_ax,map_ax,detail_ax]:
            ax.set_aspect('equal', adjustable='box')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(color='.91',lw=.6)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
        image_ax.plot(*uv.T, color='.65', lw=1.1)
        image_ax.add_patch(Rectangle((0,0), metadata['width'],metadata['height'],
                                    fill=False,edgecolor='.75',lw=.7))
        image_ax.set(xlim=(0,3840),ylim=(2160,0),xlabel='Posição horizontal (pixels)',
                     ylabel='Posição vertical (pixels)',title='(a) Objeto na imagem')
        image_ax.set_xticks([0,1000,2000,3000,3840])
        image_ax.set_yticks([0,1080,2160])
        for j,i in enumerate(sample):
            image_ax.scatter(*uv[i],s=40,color=COLORS[j],zorder=5,clip_on=False)
            image_ax.annotate(str(j+1),uv[i],xytext=(10,[6,0,-6][j]),textcoords='offset points',
                              color=COLORS[j],va='center',weight='bold')
        map_ax.plot(aircraft[:,1],aircraft[:,0],color='.45',lw=1.5)
        map_ax.scatter(projected[:,1],projected[:,0],s=6,color='.6',zorder=3)
        map_ax.set(xlim=(-34,5),ylim=(-13,23),xlabel='Leste (m)',ylabel='Norte (m)',
                   title='(b) Aeronave e projeções no terreno')
        map_ax.xaxis.set_major_locator(MultipleLocator(10))
        map_ax.yaxis.set_major_locator(MultipleLocator(10))
        for j,i in enumerate(sample):
            a,p=aircraft[i,::-1],projected[i,::-1]
            map_ax.annotate('',xy=p,xytext=a,
                            arrowprops={'arrowstyle':'->','color':COLORS[j],'lw':1,'linestyle':'--'})
            map_ax.scatter(*a,marker='s',s=40,color=COLORS[j],zorder=5)
            map_ax.scatter(*p,marker='o',s=30,color=COLORS[j],zorder=5)
            map_ax.annotate(str(j+1),a,xytext=(4,-15),textcoords='offset points',
                            color=COLORS[j],weight='bold')
        map_ax.add_patch(Rectangle((zoom_x[0],zoom_y[0]),zoom_x[1]-zoom_x[0],
                                  zoom_y[1]-zoom_y[0],fill=False,edgecolor='.4',ls=':',lw=1))
        map_ax.text(np.mean(zoom_x),zoom_y[1]+1.2,'Detalhe em (c)',ha='center',color='.4',fontsize=10)
        legend=[Line2D([],[],marker='s',color='.3',lw=0,label='Aeronave'),
                Line2D([],[],marker='o',color='.3',lw=0,label='Projeção do objeto')]
        map_ax.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,-.15),
                      frameon=False,fontsize=10,ncol=1)
        detail_ax.plot(projected[:,1],projected[:,0],color='.7',lw=.8,marker='.',ms=3)
        detail_ax.set(xlim=zoom_x,ylim=zoom_y,xlabel='Leste (m)',ylabel='Norte (m)',
                      title='(c) Detalhe das projeções')
        detail_ax.xaxis.set_major_locator(MultipleLocator(1))
        detail_ax.yaxis.set_major_locator(MultipleLocator(1))
        for j,i in enumerate(sample):
            p=projected[i,::-1]
            detail_ax.scatter(*p,s=40,color=COLORS[j],zorder=5)
            detail_ax.annotate(str(j+1),p,xytext=(6,8),textcoords='offset points',
                               color=COLORS[j],weight='bold')
        fig.savefig(ROOT/'doc/figuras/fig_trilha_real_gsd.pdf',bbox_inches='tight')
        plt.close(fig)
    print(json.dumps({k:metadata[k] for k in ['track_id','track_samples','selected_frames',
          'track_dispersion_m','duration_s','aircraft_net_displacement_m',
          'image_net_displacement_px','projected_net_displacement_m']},indent=2),flush=True)


if __name__=='__main__':
    main()
