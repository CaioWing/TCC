"""Projeção oblíqua de um multirrotor com eixos NED e rotações intrínsecas z-y-x."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

YAW, PITCH, ROLL = '#cc791e', '#496e8a', '#668368'


def project(points):
    """Vista oblíqua: x para a direita, y para cima, z na diagonal inferior esquerda."""
    points = np.asarray(points)
    return np.stack([points[0] - .55*points[2], points[1] - .55*points[2]])


def line(ax, points, **kwargs):
    ax.plot(*project(np.asarray(points).T), **kwargs)


def arrow(ax, start, end, color='0.2', lw=1.5, dashed=False, zorder=8):
    ax.annotate('', xy=project(end), xytext=project(start), zorder=zorder,
                arrowprops={'arrowstyle':'->', 'color':color, 'lw':lw,
                            'linestyle':'--' if dashed else '-'})


def label(ax, point, text, offset=(0,0), color='0.2', size=11, ha='center'):
    ax.annotate(text, project(point), xytext=offset, textcoords='offset points',
                ha=ha, va='center', fontsize=size, color=color, zorder=20,
                bbox={'facecolor':'white','edgecolor':'none','pad':.6})


def arc(ax, points, text, color, offset):
    sector = np.vstack([np.zeros(3), points])
    ax.add_patch(Polygon(project(sector.T).T, facecolor=color,
                         alpha=.10, edgecolor='none', zorder=0))
    line(ax, points, color=color, lw=2.2, zorder=12)
    arrow(ax, points[-8], points[-1], color=color, lw=2.2, zorder=12)
    middle = project(points[len(points)//2])
    ax.plot(*middle, 'o', color=color, ms=3, zorder=12)
    ax.annotate(text, middle, xytext=offset, textcoords='offset points',
                ha='center', va='center', color=color, fontsize=11, zorder=20,
                bbox={'facecolor':'white','edgecolor':'none','pad':2},
                arrowprops={'arrowstyle':'-', 'color':color, 'lw':.9, 'shrinkA':4, 'shrinkB':3})


def main():
    psi, theta, phi = np.deg2rad([-40,35,35])
    cp, sp, ct, st, cr, sr = np.cos(psi),np.sin(psi),np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    rz = np.array([[cp,-sp,0],[sp,cp,0],[0,0,1]])
    ry = np.array([[ct,0,st],[0,1,0],[-st,0,ct]])
    rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    rotation = rz @ ry @ rx
    zero = np.zeros(3)
    e1,e2,e3 = np.eye(3)
    x1, y2 = rz @ e1, rz @ ry @ e2
    with plt.rc_context({'font.size':11, 'pdf.fonttype':42}):
        fig, ax = plt.subplots(figsize=(9,7))
        fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5,3.9)
        ax.set_ylim(-2.3,3.05)
        ax.axis('off')
        # Three-dimensional reference axes in a conventional oblique projection.
        for axis, name, length, offset in [(e1,'$x_0$ · norte',3.05,(15,0)),
                                   (e2,'$y_0$ · leste',2.25,(0,14)),
                                   (e3,'$z_0$ · baixo',2.,(-8,-12))]:
            arrow(ax, zero, length*axis, color='0.5', lw=1.2, dashed=True, zorder=1)
            label(ax, length*axis, name, offset=offset, color='0.4', size=10,
                  ha='left' if name.startswith('$x_0$') else 'center')
        # Intermediate axes provide the actual starting directions of pitch and roll.
        for axis, name, length, offset in [(x1,'$x_1$',3.05,(10,-10)), (y2,'$y_2$',3.35,(-8,12))]:
            arrow(ax, zero, length*axis, color='0.45', lw=1.3, dashed=True, zorder=2)
            label(ax, length*axis, name, offset=offset, color='0.4')
        for basis, name, length, offset in [(e1,'$x_c$',2.3,(15,-8)),
                                            (e2,'$y_c$',3.35,(15,4))]:
            direction = rotation @ basis
            arrow(ax, zero, length*direction, color='0.12', lw=1.6, zorder=3)
            label(ax, length*direction, name, offset=offset, color='0.12')
        # Project the same 3D body and four rotors into the oblique view.
        t = np.linspace(0,2*np.pi,80)
        for x in [-.52,.52]:
            for y in [-.52,.52]:
                center=np.array([x,y,0.])
                line(ax,[zero,rotation@center], color='0.3', lw=6, zorder=5)
                disk=np.array([rotation@(center+np.array([.18*np.cos(a),.18*np.sin(a),-.035])) for a in t])
                ax.add_patch(Polygon(project(disk.T).T, facecolor='0.86', edgecolor='0.35',
                                     lw=1.2, alpha=.9, zorder=6))
                line(ax,[rotation@(center+np.array([-.16,0,-.04])),rotation@(center+np.array([.16,0,-.04]))],
                     color='0.4',lw=1.2,zorder=7)
        corners=np.array([[x,y,z] for z in [-.08,.08] for y in [-.16,.16] for x in [-.28,.28]])
        corners=np.array([rotation@p for p in corners])
        faces=[corners[list(ids)] for ids in [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]]
        faces.sort(key=lambda face: np.mean(face,axis=0)@np.array([.55,.55,1.]))
        for i,face in enumerate(faces):
            ax.add_patch(Polygon(project(face.T).T,facecolor=str(.45+.035*i),edgecolor='0.25',lw=.8,zorder=8+i*.1))
        nose=np.array([rotation@p for p in [np.array([.27,0,.085]),np.array([.08,-.10,.085]),np.array([.08,.10,.085])]])
        ax.add_patch(Polygon(project(nose.T).T,facecolor=YAW,edgecolor=YAW,zorder=9))
        yaw_points=np.array([2.8*np.array([np.cos(a),np.sin(a),0]) for a in np.linspace(0,psi,80)])
        pitch_points=np.array([1.7*(rz@np.array([np.cos(a),0,-np.sin(a)])) for a in np.linspace(0,theta,80)])
        roll_points=np.array([3.1*(rz@ry@np.array([0,np.cos(a),np.sin(a)])) for a in np.linspace(0,phi,80)])
        arc(ax,yaw_points,r'Guinada $\psi$',YAW,(48,-32))
        arc(ax,pitch_points,r'Arfagem $\theta_c$',PITCH,(-90,-20))
        arc(ax,roll_points,r'Rolamento $\phi_c$',ROLL,(65,25))
        output=Path(__file__).resolve().parents[1]/'doc/figuras/fig_angulos_euler.pdf'
        fig.savefig(output,bbox_inches='tight',pad_inches=.1)
        plt.close(fig)
        print(output)


if __name__=='__main__':
    main()
