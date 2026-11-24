"""Compara os pedidos intermediários para a mesma passagem do ponto B no voo 09."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from curva_passagens import mascaras_teste, quadros_geograficos
from dados import ROOT
from figura_exemplo_temporal import VIDEO, fotos_do_video
from figura_passagem import pontos_no_quadro
from figuras_malha import contagem, legenda, painel
from passagens import DESTINO, carregar_voo, melhor_recorte


def desenhar():
    pose, tamanho, grade, _ = carregar_voo(VIDEO)
    eventos = pd.read_csv(DESTINO / "eventos.csv").query("video == @VIDEO")
    # B é o mesmo ponto que aparece pela primeira vez no quadro 48 da figura anterior.
    b, = eventos.query("i == -14 and j == 7 and quadro_entrada == 48").itertuples()
    percurso = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1))]
    dist_b = percurso[b.quadro_entrada:b.quadro_saida+1]
    tercos = b.quadro_entrada + quadros_geograficos(dist_b, 2)[1:-1]
    quadros = np.array([tercos[0], b.quadro_meio, tercos[1]])
    marcos = ["1/3 do percurso", "Metade do percurso", "2/3 do percurso"]
    metas = [1, 2, 3, 4, int(eventos.quadros_visiveis.max())]
    mascaras = mascaras_teste(VIDEO, eventos, quadros, metas)
    fig, eixos = plt.subplots(3, 3, figsize=(7.15, 4.72))
    fig.subplots_adjust(left=.157, right=.993, top=.84, bottom=.045, wspace=.055, hspace=.27)
    legenda(fig, "Ponto")
    for coluna, (q, marco, foto) in enumerate(zip(quadros, marcos, fotos_do_video(quadros))):
        visiveis = pontos_no_quadro(eventos, pose, tamanho, q)
        referencia = visiveis.loc[b.Index, ["u", "v"]].to_numpy()
        for linha, regra in enumerate(["Só extremos", "Um quadro\nno meio", "Dois quadros\nnos terços"]):
            ax = eixos[linha, coluna]
            pedidos = pd.Series(False, index=visiveis.index)
            for p in visiveis.itertuples():
                dist = percurso[p.quadro_entrada:p.quadro_saida+1]
                pedidos.at[p.Index] = q - p.quadro_entrada in quadros_geograficos(dist, linha)
            ativos = visiveis.loc[pedidos, ["u", "v"]].to_numpy()
            outros = visiveis.loc[~pedidos, ["u", "v"]].to_numpy()
            enviados = np.flatnonzero(mascaras[linha + 2][q])
            np.testing.assert_array_equal(enviados, np.unique([melhor_recorte(*xy, grade) for xy in ativos]))
            assert pedidos.loc[b.Index] == ((linha == 1 and coluna == 1) or (linha == 2 and coluna != 1))
            painel(ax, foto, tamanho, grade[enviados], outros, ativos, referencia, "B")
            if coluna == 0:
                pos = ax.get_position()
                centro = (pos.y0 + pos.y1)/2
                fig.text(.005, centro+.013, f"$k_{{\\mathrm{{int}}}}={linha}$", fontsize=10, va="bottom")
                fig.text(.005, centro-.002, regra, fontsize=8.2, va="top")
            if linha == 0:
                ax.set_title(f"{marco}\nQuadro {q}", fontsize=8.5, pad=4)
            ax.text(.5, -.04, contagem(len(enviados)), transform=ax.transAxes,
                    fontsize=8.1, va="top", ha="center")
            print(f"k_int={linha}, quadro {q}: {contagem(len(enviados))}; pedido de B: {pedidos.loc[b.Index]}")
    fig.savefig(ROOT / "doc/figuras/fig_vistas_da_passagem.pdf",
                bbox_inches="tight", pad_inches=.045, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        desenhar()
