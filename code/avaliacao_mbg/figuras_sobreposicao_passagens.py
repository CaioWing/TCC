"""Distribuição da IoU e exemplos de revisitas aos mesmos recipientes."""
from itertools import product

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

from dados import OUT, ROOT, folder
from figura_altitude_quadrantes import CLASSES
from memoria_sim import T
from passagens import carregar_voo
from registro_visual_passagens import JANELA, cantos, carregar_recortes, transformar
from sobreposicao_passagens import poligono

PASTA = OUT / "quadrantes/sobreposicao_passagens"
FIGURAS = ROOT / "doc/figuras"
CHAVE = ["video", "quadro_entrada", "recorte_entrada", "quadro_travessia", "recorte_travessia"]
FASES = ("entrada", "travessia")
CORES = ("#156182", "#af520e")


def decimal(valor, casas=2):
    return f"{valor:.{casas}f}".replace(".", ",")


def salvar(figura, nome):
    figura.savefig(FIGURAS / f"{nome}.pdf", bbox_inches="tight", pad_inches=.035, dpi=280,
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figura)


def curva(eixo, valores, cor, estilo, legenda):
    valores = np.sort(valores)
    if not len(valores) or not np.isfinite(valores).all():
        raise ValueError(f"Amostra inválida para {legenda}")
    positivos = valores[valores > 0]
    zeros = len(valores) - len(positivos)
    x = np.r_[0, positivos, 1]
    y = np.r_[zeros, np.arange(zeros + 1, len(valores) + 1), len(valores)] * 100 / len(valores)
    eixo.step(x, y, where="post", color=cor, linestyle=estilo, lw=2, label=legenda)


def distribuicoes(pares, trilhas, registros, alinhadas):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.15, 3.05), layout="constrained")
    a.set_title("(a) Projeção no terreno", loc="left")
    b.set_title("(b) Conferência pelas imagens", loc="left")
    curva(a, pares.iou_recortes, "#383838", ":", "Recortes")
    curva(a, trilhas.iou_deteccoes, CORES[0], "-", "Caixas previstas")
    curva(a, trilhas.iou_anotacoes, CORES[1], "--", "Caixas anotadas")
    a.set(ylim=(0, 102), yticks=np.arange(0, 101, 25), ylabel="Fração acumulada dos pares (%)")
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2, fontsize=8, frameon=False)
    b.plot([0, 1], [0, 1], color="#8b8b8b", lw=1, ls="--", zorder=0)
    b.scatter(registros.iou_gsd, registros.iou_visual, s=20, facecolors="none", edgecolors="#484848",
              lw=.9, label="Recortes")
    caixas = alinhadas.dropna(subset=["iou_visual_deteccoes"])
    b.scatter(caixas.iou_gsd_deteccoes, caixas.iou_visual_deteccoes, marker="x", s=28, color=CORES[0],
              lw=1.25, label="Caixas previstas", clip_on=False)
    b.set(ylim=(-.015, 1.015), yticks=np.linspace(0, 1, 6), ylabel="IoU após alinhar as imagens")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: decimal(v, 1)))
    b.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2, fontsize=8, frameon=False)
    for eixo in (a, b):
        eixo.set(xlim=(-.015, 1.01), xticks=np.linspace(0, 1, 6), xlabel="IoU no terreno")
        eixo.xaxis.set_major_formatter(FuncFormatter(lambda v, _: decimal(v, 1)))
        eixo.grid(axis="both", color="#dedede", lw=.55)
        eixo.set_axisbelow(True)
    salvar(fig, "fig_sobreposicao_distribuicao")


def selecionar_exemplos(trilhas, alinhadas):
    """Aproxima três percentis com classes e regiões distintas e caixas inteiras."""
    candidatos = trilhas.merge(alinhadas[CHAVE + ["classe", "trilha", "iou_visual_deteccoes"]],
                               on=CHAVE + ["classe", "trilha"], how="left", validate="one_to_one")
    candidatos = candidatos[candidatos.anotacao_inteira_nos_dois == 1].copy()
    # Uma rotação comum deixa os dois contornos horizontais sem alterar a IoU.
    mesma_orientacao = pd.Series(False, index=candidatos.index)
    for video, grupo in candidatos.groupby("video"):
        pose, _, _, _ = carregar_voo(video)
        giro = pose[grupo.quadro_travessia, 2] - pose[grupo.quadro_entrada, 2]
        mesma_orientacao.loc[grupo.index] = np.isclose(np.sin(giro / 2), 0, atol=1e-9)
    candidatos = candidatos[mesma_orientacao]
    candidatos = candidatos.assign(regiao=candidatos.video.str.split("_region").str[-1])
    grupos = []
    for q in (.1, .5, .9):
        alvo = trilhas.iou_deteccoes.quantile(q)
        ordem = candidatos.assign(distancia=(candidatos.iou_deteccoes - alvo).abs())
        if q == .1:
            ordem = ordem.dropna(subset=["iou_visual_deteccoes"])
        melhores = ordem.sort_values(["distancia", *CHAVE, "trilha"]).drop_duplicates(["regiao", "classe"])
        grupos.append(list(melhores.itertuples()))
    trios = (trio for trio in product(*grupos)
             if len({r.regiao for r in trio}) == 3 and len({r.classe for r in trio}) == 3)
    escolhidos = min(trios, key=lambda trio: sum(r.distancia for r in trio))
    return [candidatos.loc[r.Index] for r in escolhidos]


def mapa(eixo, areas, iou, base_entrada):
    areas = [pontos @ base_entrada for pontos in areas]
    vertices = np.vstack(areas)
    centro = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    largura = np.ptp(vertices, axis=0).max() * 1.27
    for pontos, cor, estilo in zip(areas, CORES, ("-", "--"), strict=True):
        eixo.add_patch(Polygon(pontos - centro, facecolor=cor, edgecolor="none", alpha=.11))
        eixo.add_patch(Polygon(pontos - centro, fill=False, edgecolor=cor, ls=estilo, lw=1.45))
    eixo.set(xlim=(-largura/2, largura/2), ylim=(-largura/2, largura/2), aspect="equal")
    eixo.set_facecolor("#f7f7f4")
    tamanho = max(x * 10.0**p for p in range(-3, 3) for x in (1, 2, 5)
                  if x * 10.0**p <= largura * .25)
    x, y = -largura * .44, -largura * .45
    eixo.plot([x, x+tamanho], [y, y], lw=1.7, color="#252729")
    escala = f"{tamanho:g}".replace(".", ",")
    eixo.annotate(f"{escala} m", (x+tamanho, y), (3, 0), "data", "offset points",
                  ha="left", va="center", size=7)
    eixo.text(.98, .015, f"IoU {decimal(iou)}", transform=eixo.transAxes, ha="right",
              va="bottom", size=7.3, bbox=dict(facecolor="white", edgecolor="none", pad=.8))


def destacar_area_comum(eixo, registro, caixa_entrada, iou_caixas):
    h = np.array([[registro[f"h{i}{j}"] for j in range(3)] for i in range(3)])
    primeira = transformar(JANELA, h)
    # O contorno da entrada está nas coordenadas locais da imagem de travessia.
    sombra = np.zeros((T, T, 4), dtype=np.uint8)
    sombra[:, :, 3] = 145
    cv2.fillConvexPoly(sombra, np.rint(primeira).astype(np.int32), (0, 0, 0, 0))
    eixo.imshow(sombra, extent=(0, T, T, 0), interpolation="nearest")
    eixo.add_patch(Polygon(primeira, fill=False, edgecolor="white", lw=1, ls=":"))
    caixa_alinhada = transformar(cantos(caixa_entrada), h)
    for cor, largura in (("white", 2.15), (CORES[0], 1.25)):
        eixo.add_patch(Polygon(caixa_alinhada, fill=False, edgecolor=cor, lw=largura,
                               ls=(0, (2.5, 2)), zorder=4))
    eixo.text(.975, .975, f"Caixas · IoU {decimal(iou_caixas)}", transform=eixo.transAxes,
              ha="right", va="top", size=7, color="white",
              bbox=dict(facecolor="#202526", alpha=.8, edgecolor="none", pad=1.5))


def ilustrar_linha(eixos, caso, registro):
    pose, tamanho, tiles, _ = carregar_voo(caso.video)
    quadros = [int(caso[f"quadro_{fase}"]) for fase in FASES]
    fonte, = (folder(caso.video) / "avi").glob("*.avi")
    imagens = carregar_recortes(fonte, tiles, [tuple(int(caso[c]) for c in CHAVE[1:])], colorido=True)
    recortes, caixas, caixas_locais = [], [], []
    for eixo, fase, q, cor in zip(eixos[:2], FASES, quadros, CORES, strict=True):
        j = int(caso[f"recorte_{fase}"])
        x, y = tiles[j]
        caixa = np.array([caso[f"det_{fase}_{k}"] for k in ("x1", "y1", "x2", "y2")])
        recortes.append(poligono((x, y, x+T, y+T), pose[q], tamanho))
        caixas.append(poligono(caixa, pose[q], tamanho))
        eixo.imshow(cv2.cvtColor(imagens[q, j], cv2.COLOR_BGR2RGB),
                    extent=(0, T, T, 0), interpolation="antialiased")
        caixas_locais.append(caixa - [x, y, x, y])
        x1, y1, x2, y2 = caixas_locais[-1]
        if fase == "travessia" and registro is not None:
            destacar_area_comum(eixo, registro, caixas_locais[0], caso.iou_visual_deteccoes)
        for borda, largura in (("white", 2.9), (cor, 1.55)):
            eixo.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                                     edgecolor=borda, lw=largura))
        eixo.text(.975, .025, f"Quadro {q}", transform=eixo.transAxes, size=7, color="white",
                  va="bottom", ha="right", bbox=dict(facecolor="#202526", alpha=.8, edgecolor="none", pad=1.5))
        eixo.set(xlim=(0, T), ylim=(T, 0))
    lado = recortes[0][1] - recortes[0][0]
    ux, uy = lado / np.linalg.norm(lado)
    base_entrada = np.array([[ux, -uy], [uy, ux]])
    mapa(eixos[2], recortes, caso.iou_recortes, base_entrada)
    mapa(eixos[3], caixas, caso.iou_deteccoes, base_entrada)
    numero, regiao = caso.video.removeprefix("video").split("_region")
    classe = CLASSES[caso.classe]
    eixos[0].set_ylabel(f"Voo {numero} · {regiao}\n{classe}", size=8)
    for eixo in eixos:
        eixo.set(xticks=[], yticks=[])
        for spine in eixo.spines.values():
            spine.set_visible(False)


def exemplos(trilhas, registros, alinhadas):
    casos = selecionar_exemplos(trilhas, alinhadas)
    fig, eixos = plt.subplots(len(casos), 4, figsize=(7.15, 5.62))
    fig.subplots_adjust(left=.08, right=.995, top=.945, bottom=.075, wspace=.055, hspace=.08)
    for linha, caso in zip(eixos, casos, strict=True):
        registro = registros.loc[tuple(caso[c] for c in CHAVE)] if pd.notna(caso.iou_visual_deteccoes) else None
        ilustrar_linha(linha, caso, registro)
    for eixo, titulo in zip(eixos[0], ("Entrada", "Primeira travessia", "Terreno dos recortes",
                                       "Caixas no terreno"), strict=True):
        eixo.set_title(titulo, fontsize=9, pad=5)
    chaves = [Line2D([], [], lw=2, color=cor, ls=estilo, label=fase) for cor, estilo, fase
              in zip(CORES, ("-", "--"), ("Entrada", "Primeira travessia"), strict=True)]
    fig.legend(handles=chaves, loc="lower center", ncol=2, frameon=False)
    salvar(fig, "fig_sobreposicao_pares")


if __name__ == "__main__":
    pares = pd.read_csv(PASTA / "pares_recortes.csv").query("origem == 'media_do_voo'")
    trilhas = pd.read_csv(PASTA / "pares_trilhas.csv").query("origem == 'media_do_voo'")
    trilhas = trilhas.dropna(subset=["iou_deteccoes", "iou_anotacoes"])
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        registros = pd.read_csv(PASTA / "amostra_registro_visual.csv").query("aceito == 1")
        alinhadas = pd.read_csv(PASTA / "trilhas_registro_visual.csv")
        distribuicoes(pares, trilhas, registros, alinhadas)
        registros = registros.set_index(CHAVE)
        assert registros.index.is_unique
        exemplos(trilhas, registros, alinhadas)
