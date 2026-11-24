"""Exemplos de quatro regras de seleção com as imagens e máscaras do voo 09."""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from adaptativa import (ESCALA, JANELA, PASSO, TEMP as CACHE_DETALHE, amostras_video, detalhe_no_solo,
                        geometria)
from dados import CACHE, OUT, ROOT, folder
from memoria_sim import CELL, T
from figura_saida_faixas import quadro_da_celula, saidas
from paleta_tcc import BLUE, ORANGE
from selecao_trajetoria import agenda, preparar_grade, selecionar

VIDEO = "video09_regionE"


def imagem(quadro):
    arquivo, = (folder(VIDEO) / "avi").glob("*.avi")
    leitor = cv2.VideoCapture(str(arquivo))
    try:
        leitor.set(cv2.CAP_PROP_POS_FRAMES, int(quadro))
        ok, bgr = leitor.read()
        if not ok or round(leitor.get(cv2.CAP_PROP_POS_FRAMES)) != quadro + 1:
            raise IOError(f"Falha na leitura do quadro {quadro}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        leitor.release()


def cobertas(t, tiles, mask, raster):
    j0, j1, i0, i1, dentro, u, v = raster(t)
    coberta = np.zeros_like(dentro)
    for x, y in tiles[mask[t]]:
        coberta |= (u >= x) & (u < x+T) & (v >= y) & (v < y+T)
    j, i = np.nonzero(dentro & coberta)
    return {(j0 + a, i0 + b): (u[a, b], v[a, b]) for a, b in zip(j, i)}


def exemplos_detalhe(tiles, raster, tamanho):
    config = json.loads((OUT / "adaptativa/protocolo.json").read_text(encoding="utf-8"))
    with np.load(CACHE_DETALHE / "mascaras" / f"{VIDEO}.npz") as arquivo:
        mascara = arquivo["selecao"]
    primeira = cobertas(0, tiles, mascara, raster)
    segunda = cobertas(PASSO, tiles, mascara, raster)
    _, detalhe = next(amostras_video(VIDEO, [0]))
    chaves = list(primeira)
    u, v = np.array(list(primeira.values())).T
    scores = detalhe_no_solo(detalhe, u, v)
    interna = (u > 90) & (u < tamanho[0]-90) & (v > 90) & (v < tamanho[1]-90)
    repetida = np.array([c in segunda for c in chaves])
    escolhidas = []
    for candidatos, quantil in [(interna & repetida & (scores <= config["limiar"]), .2),
                                (interna & (scores > config["limiar"]), .8)]:
        indices = np.flatnonzero(candidatos)
        alvo = np.quantile(scores[indices], quantil)
        i = indices[np.argmin(np.abs(scores[indices] - alvo))]
        pixel_reduzido = np.floor(np.array([u[i], v[i]]) / ESCALA).astype(int)
        centro_vizinhanca = pixel_reduzido * ESCALA + ESCALA // 2
        escolhidas.append((chaves[i], centro_vizinhanca, scores[i]))
    return escolhidas


def exemplos_cobertura():
    with np.load(CACHE / f"{VIDEO}.npz") as arquivo:
        inicio = int(arquivo["starts"][0])
        ids = (arquivo["frames"] >= inicio) & (arquivo["frames"] < inicio + 120)
        trecho = dict(pose=arquivo["pose"][ids], size=arquivo["size"], gsd=float(arquivo["gsd"]))
        quadros = arquivo["frames"][ids]
        fps = float(arquivo["fps"])
    protocolo = json.loads((OUT / "trajetoria/protocolo.json").read_text(encoding="utf-8"))
    tabela = pd.read_csv(OUT / "trajetoria/por_trecho.csv")
    base = tabela[(tabela.video == VIDEO) & (tabela.conjunto == "iniciais")
                  & (tabela.trecho == 1) & (tabela.fase == 0)].set_index("metodo")
    assert len(quadros) == 120
    grade = preparar_grade(trecho)
    indices = agenda(trecho["pose"], fps, protocolo["quadros_escolhidos"], 0)
    # Terceiro quadro do agendamento, após os dois primeiros lotes.
    q = indices[2]
    escolhas = []
    for nome, peso in [("trajetoria", protocolo["peso_escolhido"]), ("rotacao", None)]:
        orcamento = int(base.loc[nome, "recortes"])
        mask, stats = selecionar(trecho["pose"], grade, indices, orcamento, 0, peso)
        assert stats["celulas_observadas"] == base.loc[nome, "celulas_observadas"]
        def uniao(i):
            return np.unique(np.concatenate([grade[0](i)[tile] for tile in np.flatnonzero(mask[i])]))
        anteriores = np.union1d(uniao(indices[0]), uniao(indices[1]))
        novas = np.setdiff1d(uniao(q), anteriores).size
        escolhas.append((mask[q], novas))
    assert escolhas[0][0].sum() == escolhas[1][0].sum()
    return int(quadros[q]), escolhas


def painel_selecao(ax, foto, tiles, mascara):
    escala = 4
    pequena = cv2.resize(foto, None, fx=1/escala, fy=1/escala, interpolation=cv2.INTER_AREA)
    cinza = cv2.cvtColor(pequena, cv2.COLOR_RGB2GRAY)
    fundo = np.repeat((.36 * cinza + 160).astype("uint8")[:, :, None], 3, axis=2)
    h, w = foto.shape[:2]
    for x, y in tiles[mascara]:
        sub = np.s_[y//escala:(y+T)//escala, x//escala:(x+T)//escala]
        fundo[sub] = pequena[sub]
    ax.imshow(fundo, extent=(0, w, h, 0))
    for x, y in tiles[mascara]:
        ax.add_patch(Rectangle((x, y), T, T, fill=False, ec=BLUE, lw=.95))
    ax.set_axis_off()


def desenhar():
    cv2.setNumThreads(1)
    n, tiles, shape, raster = geometria(VIDEO)
    with np.load(CACHE / f"{VIDEO}.npz") as src:
        tamanho = src["size"]
    altura = int(tamanho[1])
    primeiro, fim, celula = saidas(n, tiles, shape, raster, altura)
    detalhes = exemplos_detalhe(tiles, raster, tamanho)
    q, escolhas = exemplos_cobertura()
    print("Faixas (célula, quadros)", celula, primeiro, fim)
    print("Detalhe (célula, pixel, medida)", detalhes)
    print("Cobertura e rodízio (quadro, recortes e células novas)", q,
          [(int(m.sum()), int(novas)) for m, novas in escolhas])
    fig = plt.figure(figsize=(7.45, 4.85))
    grade = fig.add_gridspec(2, 2, left=.06, right=.985, top=.92, bottom=.07,
                             wspace=.24, hspace=.65, height_ratios=(1, 1))
    ax = fig.add_subplot(grade[0, 0])
    t0 = primeiro
    while t0 > 0 and quadro_da_celula(raster, t0 - 1, celula) is not None:
        t0 -= 1
    tempos = np.arange(t0, fim + 1)
    pos = [quadro_da_celula(raster, t, celula)[1] / altura for t in tempos]
    ax.axhspan(0, .5, color=BLUE, alpha=.08)
    ax.axhspan(.5, 1, color=ORANGE, alpha=.09)
    ax.axhline(.5, color=".58", ls="--", lw=.7)
    ax.plot(tempos, pos, color=BLUE, lw=1.8, label="Posição da célula")
    ax.scatter([primeiro, fim], [pos[primeiro-t0], pos[-1]], fc=ORANGE, ec=".25",
               lw=.55, s=34, zorder=3, clip_on=False, label="Envio ao detector")
    ax.set(ylim=(1.04, -.04), xlabel="Quadro do vídeo", xticks=[t0, primeiro, fim],
           yticks=[0, .5, 1], yticklabels=["Topo", "Meio", "Base"])
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.set_title("(a) Saída de cada faixa", loc="left", fontsize=10)
    ax.legend(loc="lower right" if pos[-1] < .5 else "upper right", frameon=False, fontsize=7.4)

    ax = fig.add_subplot(grade[0, 1])
    ax.set_axis_off()
    ax.set_title("(b) Segunda vista por pouco detalhe", loc="left", fontsize=10)
    foto = imagem(0)
    for p, (_, (x, y), _) in enumerate(detalhes):
        mini = ax.inset_axes([.02 + p*.51, .14, .45, .82])
        lado = ESCALA * JANELA
        raio = 3 * lado // 2
        local = foto[y-raio:y+raio, x-raio:x+raio]
        mini.imshow(local)
        mini.add_patch(Rectangle((lado-.5, lado-.5), lado, lado, fill=False, ec="white", lw=1.2))
        mini.set_axis_off()
        ax.text(.245 + p*.51, .04, ["Pede outra vista", "Uma vista atende"][p], ha="center",
                transform=ax.transAxes, fontsize=8)
    foto = imagem(q)
    for col, (mask, novas) in enumerate(escolhas):
        ax = fig.add_subplot(grade[1, col])
        painel_selecao(ax, foto, tiles, mask)
        ax.set_title(["(c) Recortes por cobertura", "(d) Rodízio de recortes"][col],
                     loc="left", fontsize=10)
        ax.text(.5, -.06, f"{mask.sum()} recortes   ·   {int(novas)*CELL**2:.1f} m² novos".replace(".", ","),
                transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
    fig.savefig(ROOT / "doc/figuras/fig_exemplos_regras_geograficas.pdf",
                bbox_inches="tight", pad_inches=.055, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "font.size": 8.4}):
        desenhar()
