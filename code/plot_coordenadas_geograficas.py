"""Globo esquemático e composição de posições no plano local norte-leste."""

from pathlib import Path
from paleta_tcc import BLUE, ORANGE, GREEN

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def sphere(latitude, longitude):
    return np.array([np.cos(latitude) * np.cos(longitude),
                     np.cos(latitude) * np.sin(longitude), np.sin(latitude)])


def main():
    orange = ORANGE
    phi, lam = np.deg2rad([30, 40])
    origin = sphere(phi, lam)
    east = np.array([-np.sin(lam), np.cos(lam), 0])
    north = np.array([-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)])
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(11, 4.7))
        globe = fig.add_subplot(121, projection='3d')
        local = fig.add_subplot(122)
        fig.subplots_adjust(left=0.01, right=0.98, top=0.88, bottom=0.12, wspace=0.23)
        fig.text(0.25, 0.95, '(a) Origem geográfica e direções locais', ha='center')
        fig.text(0.77, 0.95, '(b) Posições no plano local', ha='center')
        globe.set_proj_type('ortho')
        globe.view_init(elev=25, azim=-35)
        globe.set_box_aspect((1, 1, 1), zoom=1.28)
        globe.set_axis_off()
        globe.set(xlim=(-1.05, 1.2), ylim=(-1.05, 1.2), zlim=(-1.05, 1.25))
        longitude = np.linspace(0, 2*np.pi, 241)
        latitude = np.linspace(-np.pi/2, np.pi/2, 121)
        for lat in np.deg2rad([-60, -30, 0, 30, 60]):
            curve = np.array([
                np.cos(lat)*np.cos(longitude), np.cos(lat)*np.sin(longitude),
                np.full_like(longitude, np.sin(lat))])
            globe.plot(*curve, color='0.85' if lat else '0.35', lw=0.6 if lat else 1.1)
        for lon in np.deg2rad(np.arange(0, 360, 30)):
            curve = np.array([np.cos(latitude)*np.cos(lon), np.cos(latitude)*np.sin(lon), np.sin(latitude)])
            globe.plot(*curve, color='0.85', lw=0.6)
        # Equator reference, local meridian and the projection of O on the equator.
        p_equator = sphere(0, lam)
        for endpoint in [np.array([1, 0, 0]), p_equator, origin]:
            globe.plot(*np.column_stack([np.zeros(3), endpoint]), color='0.3', lw=1)
        arc_lam = np.linspace(0, lam, 80)
        globe.plot(0.42*np.cos(arc_lam), 0.42*np.sin(arc_lam), np.zeros_like(arc_lam),
                   color=orange, lw=1.8)
        arc_phi = np.linspace(0, phi, 80)
        globe.plot(0.62*np.cos(arc_phi)*np.cos(lam), 0.62*np.cos(arc_phi)*np.sin(lam),
                   0.62*np.sin(arc_phi), color=orange, lw=1.8)
        globe.text(*(0.48*sphere(-0.13, lam/2)), '$\\lambda_0$', color=orange, fontsize=13)
        globe.text(*(0.79*sphere(phi/2, lam)), '$\\phi_0$', color=orange, fontsize=13)
        meridian = np.array([np.cos(latitude)*np.cos(lam), np.cos(latitude)*np.sin(lam), np.sin(latitude)])
        globe.plot(*meridian, color='0.45', lw=1)
        globe.plot([0, 0], [0, 0], [-1.1, 1.18], color='0.35', lw=1)
        globe.text(0, 0, 1.23, 'Polo norte', ha='center', fontsize=9)
        globe.text(1.12, -0.1, -0.09, 'Meridiano de\nreferência', fontsize=9)
        globe.text(-0.75, -0.8, -0.05, 'Equador', fontsize=9)
        # Small tangent patch with the orthogonal north/east directions at O.
        corners = [origin + a*east + b*north for a,b in [(-.15,-.12),(.45,-.12),(.45,.43),(-.15,.43)]]
        globe.add_collection3d(Poly3DCollection([corners], facecolors='#f5e4cf',
                                               edgecolors=orange, alpha=0.28, linewidths=0.8))
        globe.scatter(*origin, s=28, color=orange, depthshade=False)
        globe.text(*(origin+0.07*north-0.15*east), '$O$', color=orange, fontsize=12)
        for direction, label in [(north, 'N'), (east, 'E')]:
            globe.quiver(*origin, *(0.55*direction), color='0.15', linewidth=1.5, arrow_length_ratio=.16)
            globe.text(*(origin+.64*direction), label, fontsize=11)

        local.set_aspect('equal')
        local.set_xlim(-0.35, 4.0)
        local.set_ylim(-0.35, 2.0)
        local.set_xticks([0, 1, 2, 3, 4])
        local.set_yticks([0, .5, 1, 1.5, 2])
        local.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'.replace('.', ',')))
        local.spines[['right','top']].set_visible(False)
        local.grid(color='0.9', lw=.6)
        local.set_axisbelow(True)
        local.axhline(0, color='0.55', lw=.8)
        local.axvline(0, color='0.55', lw=.8)
        local.set_xlabel(r'$E$ (m; leste $\rightarrow$)', labelpad=10)
        local.set_ylabel(r'$N$ (m; norte $\uparrow$)')
        aircraft = np.array([2., 1.])
        delta = np.array([2.1/np.sqrt(2), -.7/np.sqrt(2)])
        obj = aircraft + delta
        for start, end, color, lw in [(np.zeros(2), aircraft, BLUE, 1.8), (aircraft, obj, orange, 2.0)]:
            local.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle':'->', 'color':color,'lw':lw})
        local.plot([0,obj[0]], [0,obj[1]], ls='--', color='0.6', lw=1)
        for point, label, offset, color in [(np.zeros(2),'Origem $O$',(6,-17),'0.15'),
                                           (aircraft,'Aeronave',(0,12),BLUE),
                                           (obj,'Objeto',(0,12),orange)]:
            local.scatter(*point, s=30, color=color, zorder=5)
            local.annotate(label, point, xytext=offset, textcoords='offset points',
                           ha='center' if point.any() else 'left', color=color, fontsize=10)
        local.text(2.8, 1.17, r'$(\Delta E,\Delta N)$', color=orange, ha='center', fontsize=10)
        output = Path(__file__).resolve().parents[1] / 'doc/figuras/fig_coordenadas_geograficas.pdf'
        fig.savefig(output, bbox_inches='tight')
        plt.close(fig)
        print(output)


if __name__ == '__main__':
    main()
