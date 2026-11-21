"""Calcula a curva de ajuste do GSD do voo 03 e gera a figura do TCC.

Usa centros das caixas visíveis em quadros múltiplos de 5, trilhas com
pelo menos 5 amostras e a mediana entre as dispersões medianas das trilhas.
Não usa as divisões de treinamento/teste: esta é a geometria calibrada por
anotações, para ilustrar o procedimento, e não uma avaliação independente.
"""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from paleta_tcc import BLUE, ORANGE, GREEN
from fractions import Fraction
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLIGHT = ROOT / 'data/mbg_v2/raw/flight-mbg-v2-03-regionB'
OUT = ROOT / 'doc/dados/ajuste_gsd_voo03'


def load_tracks(with_details=False, flight=FLIGHT):
    """Interpola a telemetria nos tempos dos quadros, sem defasagem adicional."""
    flight = Path(flight).absolute()  # Preserva o caminho do projeto quando os dados estão em uma junção.
    video = next((flight / 'avi').glob('*.avi'))
    stream = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,width,height,nb_frames',
        '-of', 'json', str(video)], text=True))['streams'][0]
    fps = float(Fraction(stream['avg_frame_rate']))
    width, height = int(stream['width']), int(stream['height'])
    with next((flight / 'log').glob('*.csv')).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    columns = ['time(millisecond)', 'latitude', 'longitude', 'gimbalYawRaw']
    telemetry = np.array([[float(row[c]) for c in columns] for row in rows])
    assert np.isfinite(telemetry).all(), 'Telemetria não finita'
    # Ordenação estável; registros repetidos no mesmo instante usam a primeira linha.
    telemetry = telemetry[np.argsort(telemetry[:, 0], kind='stable')]
    _, unique = np.unique(telemetry[:, 0], return_index=True)
    telemetry = telemetry[unique]
    times = (telemetry[:, 0] - telemetry[0, 0]) / 1000
    lat, lon = np.deg2rad(telemetry[:, 1:3]).T
    lat0, lon0 = lat[0], lon[0]
    a, e2 = 6378137., 6.6943799901413165e-3
    rn = a / np.sqrt(1 - e2*np.sin(lat0)**2)
    rm = a*(1-e2) / (1-e2*np.sin(lat0)**2)**1.5
    north, east = rm*(lat-lat0), rn*np.cos(lat0)*(lon-lon0)
    # O CSV DJI expressa a guinada bruta do gimbal em décimos de grau.
    yaw = np.unwrap(np.deg2rad(telemetry[:, 3]/10))
    tracks, details = [], []
    for _, elem in ET.iterparse(next((flight / 'ann').glob('*.xml')), events=['end']):
        if elem.tag != 'track':
            continue
        points = []
        for box in elem.findall('box'):
            frame = int(box.get('frame'))
            if frame % 5 or box.get('outside') == '1':
                continue
            t = frame/fps
            if not times[0] <= t <= times[-1]:
                continue
            u = (float(box.get('xtl')) + float(box.get('xbr')))/2
            v = (float(box.get('ytl')) + float(box.get('ybr')))/2
            points.append((t, u, v, frame, int(box.get('occluded', '0'))))
        if len(points) >= 5:
            t, u, v, frames, occluded = np.array(points).T
            psi = np.interp(t, times, yaw)
            front, right = -(v-height/2), u-width/2
            pixel_offsets = np.column_stack([
                front*np.cos(psi) - right*np.sin(psi),
                front*np.sin(psi) + right*np.cos(psi)])
            aircraft = np.column_stack([np.interp(t, times, north), np.interp(t, times, east)])
            tracks.append((aircraft, pixel_offsets))
            if with_details:
                details.append(dict(track_id=int(elem.get('id')), label=elem.get('label'),
                                    frames=frames.astype(int), times=t,
                                    uv=np.column_stack([u, v]), yaw=psi,
                                    occluded=occluded.astype(bool)))
        elem.clear()
    metadata = dict(video=video.stem, fps=fps, width=width, height=height,
                    tracks=len(tracks), boxes=sum(len(x[0]) for x in tracks),
                    frame_step=5, minimum_track_samples=5,
                    alignment='frame/fps relative to first telemetry timestamp; no extra offset',
                    yaw='gimbalYawRaw / 10, degrees, unwrapped before interpolation',
                    position='WGS84 local north/east; origin at first telemetry position; H_e=0',
                    interpolation='linear; first sample retained at duplicate timestamps',
                    center='image center; bounding box center; coordinatewise track medians',
                    origin_latitude_deg=float(telemetry[0, 1]),
                    origin_longitude_deg=float(telemetry[0, 2]),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder, ext in [('ann', '*.xml'), ('log', '*.csv')]
                                   for p in (flight/folder).glob(ext)})
    if with_details:
        return tracks, metadata, details
    return tracks, metadata


def dispersion(tracks, gsd_cm):
    scales = np.asarray(gsd_cm)/100
    per_track = []
    for aircraft, pixels in tracks:
        projected = aircraft[None, :, :] + scales[:, None, None]*pixels[None, :, :]
        center = np.median(projected, axis=1, keepdims=True)
        distances = np.linalg.norm(projected-center, axis=2)
        per_track.append(np.median(distances, axis=1))
    return np.median(per_track, axis=0)


def fit_gsd(tracks):
    # Primeiro mínimo na ordem crescente em caso de empate numérico exato.
    coarse = np.arange(80, 221, 5)/100
    coarse_disp = dispersion(tracks, coarse)
    coarse_best = int(round(100*coarse[np.argmin(coarse_disp)]))
    fine = np.arange(max(80, coarse_best-5), min(220, coarse_best+5)+1)/100
    fine_disp = dispersion(tracks, fine)
    best = int(np.argmin(fine_disp))
    return coarse, coarse_disp, fine, fine_disp, best


def main():
    tracks, metadata = load_tracks()
    coarse, coarse_disp, fine, fine_disp, best = fit_gsd(tracks)
    metadata.update(gsd_cm=float(fine[best]), dispersion_m=float(fine_disp[best]),
                    coarse_step_cm=.05, fine_step_cm=.01,
                    refinement='within 0.05 cm/pixel of coarse minimum, clipped to search domain',
                    tie_break='smallest GSD among exact ties',
                    zero_gsd_dispersion_m=float(dispersion(tracks, [0])[0]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'metadados.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    with (OUT/'curva.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'gsd_cm_pixel', 'dispersion_m'])
        for stage, x, y in [('coarse', coarse, coarse_disp), ('fine', fine, fine_disp)]:
            writer.writerows((stage, f'{a:.2f}', f'{b:.8f}') for a, b in zip(x, y))
    with plt.rc_context({'font.size':11, 'pdf.fonttype':42}):
        fig, ax = plt.subplots(figsize=(8.5,4.3))
        fig.subplots_adjust(left=.11, right=.97, bottom=.17, top=.95)
        ax.plot(coarse, coarse_disp, color=BLUE, lw=1.6, marker='o', ms=3,
                label='Busca inicial')
        ax.plot(fine, fine_disp, color=ORANGE, lw=2, marker='o', ms=3,
                label='Refinamento')
        ax.scatter(fine[best], fine_disp[best], s=65, color=ORANGE,
                   edgecolors='white', linewidths=.8, zorder=5)
        ax.vlines(fine[best], 0, fine_disp[best], color=ORANGE, lw=1, ls='--')
        ax.annotate('Mínimo', (fine[best], fine_disp[best]), xytext=(-45,45),
                    textcoords='offset points', color=ORANGE,
                    arrowprops={'arrowstyle':'-', 'color':ORANGE})
        ax.set(xlabel='GSD testado (cm/pixel)', ylabel='Dispersão mediana das trilhas (m)',
               xlim=(.8,2.2), ylim=(0,None))
        ax.set_xticks(np.arange(.8,2.21,.2))
        formatter = FuncFormatter(lambda x, _: f'{x:.1f}'.replace('.',','))
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(axis='y', color='.9', lw=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False)
        fig.savefig(ROOT/'doc/figuras/fig_ajuste_gsd.pdf', bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
