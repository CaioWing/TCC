"""Clipe sincronizado das seleções e das caixas do detector no trecho da Figura 13."""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from adaptativa import METRICA, MOTOR, verificar_pesos
from dados import ROOT, folder
from figura_exemplo_temporal import GRUPO, TRECHO, VIDEO, VISTAS, selecoes_do_trecho
from fronteiras import selected_tiles_fronteiras
from memoria_sim import T
from metricas import load_clips
from passagens import CONJUNTOS
from trilhas_caixas import conferir_anotacoes, identidades_visuais, paleta, trilhas_geograficas

LARGURA, ALTURA = 960, 540
TOPO, RODAPE = 68, 66
FPS_LEITURA = 4
LARANJA = (255, 163, 46)
VERDE = (70, 255, 136)
FUNDO = (20, 29, 33)
BRANCO = (244, 247, 245)


def desenhar_rastros(imagem, historico, cores):
    for identidade, observacoes in historico.items():
        if len(observacoes) < 2:
            continue
        segmentos = []
        for (t0, a), (t1, b) in zip(observacoes, observacoes[1:]):
            if t1 == t0 + 1:
                segmentos.append((a, b))
                continue
            delta = np.subtract(b, a)
            comprimento = float(np.linalg.norm(delta))
            if comprimento > 0:
                for d in np.arange(0, comprimento, 10):
                    p = np.rint(a + delta * (d / comprimento)).astype(int)
                    q = np.rint(a + delta * (min(d + 6, comprimento) / comprimento)).astype(int)
                    segmentos.append((p, q))
        if segmentos:
            linhas = np.asarray(segmentos, dtype=np.int32)
            cv2.polylines(imagem, linhas, False, FUNDO, 4, cv2.LINE_AA)
            cv2.polylines(imagem, linhas, False, cores[identidade], 2, cv2.LINE_AA)


def painel(foto, grade, mascara, caixas, *, ids=None, historico=None, cores=None, t=None):
    pequena = cv2.resize(foto, (LARGURA, ALTURA), interpolation=cv2.INTER_AREA)
    cinza = cv2.cvtColor(pequena, cv2.COLOR_RGB2GRAY)
    painel = np.repeat((27 + .36 * cinza).astype("uint8")[:, :, None], 3, axis=2)
    escala = np.array([LARGURA / foto.shape[1], ALTURA / foto.shape[0]])
    bordas = []
    for pixel in grade[mascara]:
        x, y = np.rint(pixel * escala).astype(int)
        r, b = np.rint((pixel + T) * escala).astype(int)
        painel[y:b, x:r] = pequena[y:b, x:r]
        bordas.append(((x, y), (r - 1, b - 1)))
    for inicio, fim in bordas:
        cv2.rectangle(painel, inicio, fim, LARANJA, 1, cv2.LINE_AA)
    if ids is not None:
        for identidade, caixa in zip(ids, caixas, strict=True):
            centro = tuple(np.rint((caixa[:2] + caixa[2:4]) * .5 * escala).astype(int))
            historico[identidade].append((t, centro))
        desenhar_rastros(painel, historico, cores)
    for j, caixa in enumerate(caixas):
        inicio, fim = np.rint(caixa[:4].reshape(2, 2) * escala).astype(int)
        cor = VERDE if ids is None else cores[ids[j]]
        cv2.rectangle(painel, inicio, fim, (11, 36, 25), 5, cv2.LINE_AA)
        cv2.rectangle(painel, inicio, fim, cor, 3, cv2.LINE_AA)
        if ids is not None:
            centro = historico[ids[j]][-1][1]
            cv2.circle(painel, centro, 4, FUNDO, -1, cv2.LINE_AA)
            cv2.circle(painel, centro, 2, cor, -1, cv2.LINE_AA)
    return painel


def fonte(tamanho, negrito=False):
    nome = "DejaVuSans-Bold.ttf" if negrito else "DejaVuSans.ttf"
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf" / nome), tamanho)


def descrever(n, termo):
    return f"{n} {termo}" if n == 1 else f"{n} {termo}s"


def main(trajetorias=False):
    verificar_pesos()
    numeros, grade, mascaras = selecoes_do_trecho()
    clip = next(c for c in load_clips(VIDEO, MOTOR, cache=CONJUNTOS[GRUPO]) if c["clip"] == TRECHO)
    np.testing.assert_array_equal(numeros, clip["frames"])
    assert (np.diff(numeros) == 1).all()
    conf = METRICA["conf"]
    caixas = []
    for mascara in mascaras:
        final = selected_tiles_fronteiras(clip, mascara)
        exibidas = [c[c[:, 4] >= conf] for c in final]
        assert all(not len(c) or m.any() for m, c in zip(mascara, exibidas))
        caixas.append(exibidas)
    print(f"Confiança mínima {conf}, caixas por estratégia (somadas por quadro):",
          [sum(map(len, cs)) for cs in caixas], flush=True)

    nomes = ["Integral", "Temporal · um quadro a cada 30", f"Geográfica · {VISTAS} vistas por ponto"]
    if trajetorias:
        trilhas = [trilhas_geograficas(clip, cs, METRICA["raio_m"]) for cs in caixas]
        ids = identidades_visuais(caixas, trilhas)
        cores = paleta(1 + max(k for estrategia in ids for quadro in estrategia for k in quadro))
        historicos = [defaultdict(list) for _ in caixas]
        for nome, cs, locais in zip(nomes, caixas, trilhas):
            print(nome, conferir_anotacoes(clip, cs, locais), flush=True)
    tit, corpo = fonte(32, True), fonte(25)
    altura, largura = TOPO + ALTURA + RODAPE, 3 * LARGURA
    base = Image.new("RGB", (largura, altura), FUNDO)
    d = ImageDraw.Draw(base)
    for i, nome in enumerate(nomes):
        d.text((i * LARGURA + 23, TOPO // 2), nome, font=tit, fill=BRANCO, anchor="lm")
    rodape = TOPO + ALTURA + RODAPE // 2
    legendas = [(1068, LARANJA, "Recorte ativo"), (1384, VERDE, "Caixa prevista")]
    if trajetorias:
        legendas = [(630, LARANJA, "Recorte ativo")]
        d.text((965, rodape), "Cor por objeto estimado   ·   Linha entre centros   ·   Tracejado entre quadros espaçados",
               font=corpo, fill=BRANCO, anchor="lm")
    for x, cor, rotulo in legendas:
        d.rectangle((x, rodape - 10, x + 28, rodape + 10), outline=cor, width=3)
        d.text((x + 43, rodape), rotulo, font=corpo, fill=BRANCO, anchor="lm")
    d.text((largura - 23, rodape), "4 quadros de origem por segundo", font=corpo,
           fill=BRANCO, anchor="rm")

    pasta = ROOT / "doc/videos"
    pasta.mkdir(exist_ok=True)
    nome_video = "voo09-trajetorias.mp4" if trajetorias else "voo09-recortes-e-deteccoes.mp4"
    destino = pasta / nome_video
    parcial = destino.with_suffix(".partial.mp4")
    arquivo, = (folder(VIDEO) / "avi").glob("*.avi")
    leitor = cv2.VideoCapture(str(arquivo))
    codificador = subprocess.Popen([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{largura}x{altura}",
        "-framerate", str(FPS_LEITURA), "-i", "pipe:0", "-an", "-vf", "fps=24",
        "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "15",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(parcial)], stdin=subprocess.PIPE)
    try:
        leitor.set(cv2.CAP_PROP_POS_FRAMES, int(numeros[0]))
        for t, n in enumerate(numeros):
            ok, bgr = leitor.read()
            if not ok or round(leitor.get(cv2.CAP_PROP_POS_FRAMES)) != n + 1:
                raise OSError(f"Não foi possível decodificar o quadro {n}")
            foto = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            saida = base.copy()
            d = ImageDraw.Draw(saida)
            d.text((23, rodape), f"Voo 09   ·   Quadro {n}", font=corpo, fill=BRANCO, anchor="lm")
            for i, (ms, cs) in enumerate(zip(mascaras, caixas)):
                opcoes = dict(ids=ids[i][t], historico=historicos[i], cores=cores, t=t) if trajetorias else {}
                quadro = painel(foto, grade, ms[t], cs[t], **opcoes)
                saida.paste(Image.fromarray(quadro), (i * LARGURA, TOPO))
                resumo = descrever(int(ms[t].sum()), "recorte") + " · " + descrever(len(cs[t]), "caixa")
                d.text(((i+1)*LARGURA - 23, TOPO//2), resumo, font=corpo, fill=BRANCO, anchor="rm")
                if i:
                    d.line((i*LARGURA, TOPO, i*LARGURA, TOPO+ALTURA-1), fill=FUNDO, width=3)
            codificador.stdin.write(saida.tobytes())
    finally:
        leitor.release()
        codificador.stdin.close()
        status = codificador.wait()
    if status:
        raise RuntimeError("O FFmpeg não concluiu a codificação")
    parcial.replace(destino)
    print(destino)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajetorias", action="store_true", help="Ligar os centros das caixas reassociadas")
    main(parser.parse_args().trajetorias)
