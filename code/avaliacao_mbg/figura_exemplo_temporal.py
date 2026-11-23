"""Os mesmos cinco quadros do voo 09 sob três regras de seleção de recortes."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from curva_passagens import mascaras_teste
from dados import ROOT, folder
from memoria_sim import T
from paleta_tcc import ORANGE
from passagens import CONJUNTOS, DESTINO

VIDEO = "video09_regionE"
GRUPO = "novos"
TRECHO = 2
INTERMEDIARIOS = 5
VISTAS = INTERMEDIARIOS + 2
QUADROS_EXIBIDOS = np.arange(3795, 3856, 15)


def selecoes_do_trecho():
    with np.load(CONJUNTOS[GRUPO] / f"{VIDEO}.npz") as arquivo:
        inicio = arquivo["starts"][TRECHO - 1]
        quadros = arquivo["frames"][arquivo["frames"] >= inicio][:120]
        grade = arquivo["tiles"]
    eventos = pd.read_csv(DESTINO / "eventos.csv").query("video == @VIDEO")
    metas = [1, 2, 3, VISTAS, int(eventos.quadros_visiveis.max())]
    geografica = mascaras_teste(VIDEO, eventos, quadros, metas)[VISTAS][quadros]
    integral = np.ones_like(geografica)
    temporal = integral & (quadros[:, None] % 30 == 0)

    consulta = "video == @VIDEO and conjunto == @GRUPO and trecho == @TRECHO"
    curva = pd.read_csv(DESTINO / "curva/por_trecho.csv").query(consulta)
    for mascara, regra in [(integral, "integral"), (geografica, "geografica")]:
        registro = curva.query("estrategia == @regra and (k == 0 or k == @VISTAS)").iloc[0]
        assert mascara.sum() == registro.recortes
        assert mascara.any(axis=1).sum() == registro.quadros_usados
    cadencia = pd.read_csv(DESTINO / "camera/por_trecho.csv").query(consulta)
    registro = cadencia.query("modo == 'temporal' and b == 4 and fase == 0").iloc[0]
    np.testing.assert_array_equal(quadros[temporal.any(axis=1)], np.fromstring(registro.quadros, sep=" ", dtype=int))
    assert temporal.sum() == registro.recortes == geografica.sum()
    return quadros, grade, (integral, temporal, geografica)


def fotos_do_video(quadros=QUADROS_EXIBIDOS, video=VIDEO):
    caminho, = (folder(video) / "avi").glob("*.avi")
    leitor = cv2.VideoCapture(str(caminho))
    fotos = []
    try:
        for numero in quadros:
            leitor.set(cv2.CAP_PROP_POS_FRAMES, int(numero))
            ok, imagem = leitor.read()
            if not ok or round(leitor.get(cv2.CAP_PROP_POS_FRAMES)) != numero + 1:
                raise OSError(f"Quadro {numero} indisponível no voo {video}")
            h, w = imagem.shape[:2]
            foto = cv2.resize(imagem, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
            fotos.append(cv2.cvtColor(foto, cv2.COLOR_BGR2RGB))
    finally:
        leitor.release()
    return fotos


def mostrar_selecao(ax, foto, recortes):
    h, w = foto.shape[:2]
    cinza = cv2.cvtColor(foto, cv2.COLOR_RGB2GRAY)
    fundo = np.repeat((54 + .24 * cinza).astype("uint8")[:, :, None], 3, axis=2)
    for x, y in recortes:
        regiao = np.s_[y // 4:(y + T) // 4, x // 4:(x + T) // 4]
        fundo[regiao] = foto[regiao]
        ax.add_patch(Rectangle((x, y), T, T, fill=False, ec=ORANGE, lw=.55))
    ax.imshow(fundo, extent=(0, w * 4, h * 4, 0), interpolation="antialiased")
    ax.set_axis_off()


def desenhar():
    quadros, grade, selecoes = selecoes_do_trecho()
    fotos = fotos_do_video()
    indices = np.searchsorted(quadros, QUADROS_EXIBIDOS)
    np.testing.assert_array_equal(quadros[indices], QUADROS_EXIBIDOS)

    nomes = ["(a) Integral", "(b) Temporal · um quadro a cada 30",
             f"(c) Geográfica · {INTERMEDIARIOS} quadros intermediários"]
    fig = plt.figure(figsize=(7.15, 3.38))
    linhas = fig.add_gridspec(3, 5, left=.003, right=.997, bottom=.07, top=.915,
                               wspace=.05, hspace=.40)
    for linha, (nome, mascara) in enumerate(zip(nomes, selecoes)):
        posicao = linhas[linha, :].get_position(fig)
        topo = posicao.y1 + .015
        fig.text(posicao.x0, topo, nome, va="bottom", fontsize=10, fontweight="semibold")
        total = f"{mascara.sum():,}".replace(",", ".")
        usados = mascara.any(axis=1).sum()
        fig.text(posicao.x1, topo, f"{total} recortes em {usados} quadros",
                 va="bottom", ha="right", fontsize=9)
        for coluna, (foto, indice) in enumerate(zip(fotos, indices)):
            ax = fig.add_subplot(linhas[linha, coluna])
            mostrar_selecao(ax, foto, grade[mascara[indice]])
            if linha == len(nomes) - 1:
                ax.text(.5, -.07, f"Quadro {quadros[indice]}", transform=ax.transAxes,
                        va="top", ha="center", fontsize=8.7)
    fig.savefig(ROOT / "doc/figuras/fig_exemplo_temporal_por_trecho.pdf",
                bbox_inches="tight", pad_inches=.035, dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        desenhar()
