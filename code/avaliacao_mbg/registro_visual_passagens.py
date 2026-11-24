"""Compara a projeção por GSD ao registro independente dos pixels em uma amostra de pares."""
import json
from collections import defaultdict

import av
import cv2
import numpy as np
import pandas as pd

from dados import folder
from inferir import quadros_opencv
from sobreposicao_passagens import CHAVE, COORDS, DESTINO, T, VOOS, carregar_voo, sobrepor


AMOSTRA_POR_VOO = 30
SEMENTE = 639
ORIGEM = "media_do_voo"
LIMIAR_ERRO_PX = 3


def cantos(box):
    x, y, z, t = box
    return np.array([[x, y], [z, y], [z, t], [x, t]], dtype="float32")


JANELA = cantos([0, 0, T, T])


def transformar(pontos, h):
    return cv2.perspectiveTransform(pontos[None, :, :], h)[0]


def carregar_recortes(arquivo, tiles, chaves, *, colorido=False):
    """Retorna recortes por (quadro, índice), em cinza ou BGR para figuras."""
    pedidos = defaultdict(set)
    for q0, j0, q1, j1 in chaves:
        pedidos[q0].add(j0)
        pedidos[q1].add(j1)
    saida = {}
    with av.open(str(arquivo)) as video:
        stream = video.streams.video[0]
        mjpeg = stream.codec_context.name == "mjpeg"
        if mjpeg:
            for q in sorted(pedidos):
                instante = q / (stream.average_rate * stream.time_base)
                assert instante.denominator == 1
                video.seek(int(instante), stream=stream)
                frame = next(video.decode(stream))
                assert frame.pts * stream.average_rate * stream.time_base == q
                imagem = frame.to_ndarray(format="bgr24" if colorido else "gray")
                for j in pedidos[q]:
                    x, y = tiles[j]
                    saida[q, j] = imagem[y:y + T, x:x + T].copy()
    if not mjpeg:
        # O ensaio define os índices H.264 pela ordem de decodificação contínua de quadros.
        quadros = sorted(pedidos)
        for q, imagem in zip(quadros, quadros_opencv(arquivo, quadros, leitura_continua=True)):
            for j in pedidos[q]:
                x, y = tiles[j]
                rec = imagem[y:y + T, x:x + T]
                saida[q, j] = rec.copy() if colorido else cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)
    assert len(saida) == sum(map(len, pedidos.values()))
    return saida


def bons_vizinhos(descritores_a, descritores_b):
    busca = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descritores_a, descritores_b, k=2)
    return {(m.queryIdx, m.trainIdx) for m, n in busca if m.distance < .75 * n.distance}


def estimar(a, b, semente):
    sift = cv2.SIFT_create(nfeatures=1500)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    linha = dict(aceito=0, motivo="", pontos_entrada=len(ka), pontos_travessia=len(kb))
    if len(ka) < 2 or len(kb) < 2:
        return {**linha, "motivo": "descritores_insuficientes"}, None
    diretos, inversos = bons_vizinhos(da, db), bons_vizinhos(db, da)
    pares = sorted((i, j) for i, j in diretos if (j, i) in inversos)
    linha["correspondencias"] = len(pares)
    if len(pares) < 40:
        return {**linha, "motivo": "menos_de_40_correspondencias"}, None
    p = np.array([ka[i].pt for i, _ in pares], dtype="float32")
    q = np.array([kb[j].pt for _, j in pares], dtype="float32")
    ordem = np.random.default_rng(semente).permutation(len(p))
    teste, ajuste = np.split(ordem, [len(p) // 5])
    cv2.setRNGSeed(semente)
    h, mascara = cv2.findHomography(p[ajuste], q[ajuste], cv2.RANSAC, LIMIAR_ERRO_PX)
    if h is None or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        return {**linha, "motivo": "ajuste_singular"}, None
    for i in range(3):
        for j in range(3):
            linha[f"h{i}{j}"] = h[i, j]
    frontal = np.linalg.norm(transformar(p, h) - q, axis=1)
    reverso = np.linalg.norm(transformar(q, np.linalg.inv(h)) - p, axis=1)
    erro = np.maximum(frontal, reverso)
    tr = ajuste[mascara.ravel().astype(bool) & (erro[ajuste] <= LIMIAR_ERRO_PX)]
    te = teste[erro[teste] <= LIMIAR_ERRO_PX]
    linha.update(n_ajuste=len(ajuste), n_validacao=len(teste), acertos_ajuste=len(tr),
                 acertos_validacao=len(te), erro_mediano_validacao_px=float(np.median(erro[teste])))
    if len(tr) < max(20, len(ajuste) / 2):
        return {**linha, "motivo": "pouco_apoio_no_ajuste"}, None
    if len(te) < max(8, len(teste) / 2) or linha["erro_mediano_validacao_px"] > 2:
        return {**linha, "motivo": "erro_ou_pouco_apoio_na_validacao"}, None
    suporte = np.r_[tr, te]
    for nome, coord in [("entrada", p), ("travessia", q)]:
        linha[f"fracao_suporte_{nome}"] = cv2.contourArea(cv2.convexHull(coord[suporte])) / (T*T)
    if min(linha["fracao_suporte_entrada"], linha["fracao_suporte_travessia"]) < .15:
        return {**linha, "motivo": "apoio_concentrado"}, None
    projetada = transformar(JANELA, h)
    if not cv2.isContourConvex(projetada) or cv2.contourArea(projetada, oriented=True) <= 0:
        return {**linha, "motivo": "janela_projetada_invalida"}, None
    linha["iou_visual"] = sobrepor(projetada, JANELA)[0]
    # Os dois valores diferem quando a transformação altera a área aparente.
    linha["fracao_travessia_visivel_na_entrada"] = sobrepor(JANELA, projetada)[1]
    linha.update(aceito=1, motivo="aceito")
    return linha, h


def caixas_registradas(t, h, tiles):
    saida = []
    for linha in t.itertuples(index=False):
        row = {c: getattr(linha, c) for c in ["origem", "video", *CHAVE, "classe", "trilha"]}
        for nome, sufixo in [("anot", "anotacoes"), ("det", "deteccoes")]:
            original = getattr(linha, f"iou_{sufixo}")
            if pd.isna(original):
                continue
            row[f"iou_gsd_{sufixo}"] = original
            poligonos = []
            for lado in ["entrada", "travessia"]:
                limites = [getattr(linha, f"{nome}_{lado}_{campo}") for campo in COORDS]
                poligonos.append(cantos(limites) - tiles[getattr(linha, f"recorte_{lado}")].astype("float32"))
            row[f"iou_visual_{sufixo}"] = sobrepor(transformar(poligonos[0], h), poligonos[1])[0]
        saida.append(row)
    return saida


def executar():
    cv2.setNumThreads(3)
    pares = pd.read_csv(DESTINO / "pares_recortes.csv")
    trilhas = pd.read_csv(DESTINO / "pares_trilhas.csv")
    linhas, objetos = [], []
    for idx, nome in enumerate(VOOS):
        _, _, tiles, _ = carregar_voo(nome)
        g = pares[(pares.origem == ORIGEM) & (pares.video == nome)].sort_values(CHAVE)
        t = trilhas[(trilhas.origem == ORIGEM) & (trilhas.video == nome)].set_index(CHAVE).sort_index()
        rng = np.random.default_rng(SEMENTE + idx)
        pos = ((np.arange(AMOSTRA_POR_VOO) + rng.random()) * len(g) / AMOSTRA_POR_VOO).astype(int)
        arquivo, = (folder(nome) / "avi").glob("*.avi")
        amostra = g.iloc[pos]
        chaves = list(amostra[CHAVE].itertuples(index=False, name=None))
        recortes = carregar_recortes(arquivo, tiles, chaves)
        for k, (chave, iou) in enumerate(zip(chaves, amostra.iou_recortes)):
            resultado, h = estimar(recortes[chave[:2]], recortes[chave[2:]], SEMENTE + idx * AMOSTRA_POR_VOO + k)
            linhas.append(dict(origem=ORIGEM, video=nome, **dict(zip(CHAVE, chave)), iou_gsd=iou, **resultado))
            if h is not None and chave in t.index:
                objetos.extend(caixas_registradas(t.loc[[chave]].reset_index(), h, tiles))
        print(nome, sum(row["aceito"] for row in linhas[-AMOSTRA_POR_VOO:]), "aceitos", flush=True)
    pares, objetos = pd.DataFrame(linhas), pd.DataFrame(objetos)
    stats = []
    for nome in [*VOOS, "total"]:
        p = pares if nome == "total" else pares[pares.video == nome]
        t = objetos if nome == "total" else objetos[objetos.video == nome]
        ok = p[p.aceito == 1]
        det = t.dropna(subset=["iou_visual_deteccoes"])
        stats.append(dict(video=nome, amostrados=len(p), aceitos=len(ok),
                          iou_gsd_recortes_media=ok.iou_gsd.mean(), iou_visual_recortes_media=ok.iou_visual.mean(),
                          diferenca_absoluta_mediana=(ok.iou_visual - ok.iou_gsd).abs().median(),
                          erro_validacao_px_mediana=ok.erro_mediano_validacao_px.median(),
                          pares_trilha_anotada=len(t), pares_trilha_detectada=len(det),
                          iou_gsd_anot_media=t.iou_gsd_anotacoes.mean(),
                          iou_visual_anot_media=t.iou_visual_anotacoes.mean(),
                          iou_gsd_det_media=det.iou_gsd_deteccoes.mean(),
                          iou_visual_det_media=det.iou_visual_deteccoes.mean(),
                          iou_gsd_anot_pareadas_media=det.iou_gsd_anotacoes.mean(),
                          iou_visual_anot_pareadas_media=det.iou_visual_anotacoes.mean()))
    resumo = pd.DataFrame(stats)
    for nome, dados in [("amostra", pares), ("trilhas", objetos), ("resumo", resumo)]:
        dados.to_csv(DESTINO / f"{nome}_registro_visual.csv", index=False, float_format="%.10f")
    protocolo = dict(
        origem=ORIGEM, opencv=cv2.__version__, semente=SEMENTE, pares_por_voo=AMOSTRA_POR_VOO,
        amostragem="No conjunto principal de cada voo, ordenar por quadro e recorte de entrada e da primeira travessia. "
        "Selecionar 30 pares em passos de N/30 com fase inicial uniforme e semente fixa por voo, sem consultar imagens, "
        "anotações, detecções ou IoU. Estatísticas descritivas da amostra; os seis voos têm o mesmo peso na seleção.",
        registro="SIFT nas imagens em cinza, máximo de 1500 características por recorte. Vizinhos por distância L2 "
        "com razão <0,75 nos dois sentidos e reciprocidade. Com pelo menos 40 correspondências, reservar 1/5 para "
        "validação por permutação reprodutível do par. Ajustar homografia com RANSAC apenas nas demais, limiar de 3 px. "
        "A matriz h00..h22 leva pixels locais do recorte de entrada a pixels locais do recorte de travessia.",
        aceitacao="Ao menos 20 correspondências sustentam o ajuste e 8 na validação, cobrindo pelo menos metade das "
        "respectivas amostras, com erro máximo de 3 px nos dois sentidos. Mediana do erro na validação <=2 px. "
        "Envoltória convexa dos pontos aceitos cobre ao menos 15% de cada imagem. A janela transformada precisa ser "
        "convexa, com a orientação preservada. Todos os pares sorteados constam do CSV, inclusive os rejeitados.",
        medidas="IoU visual no plano da segunda imagem entre as janelas, ou entre caixas da mesma trilha, transportando "
        "os quatro cantos da primeira por homografia. A transformação usa apenas os pixels. A IoU GSD de comparação "
        "usa os mesmos pares. Detecções exigem o pareamento um a um em ambas as vistas do protocolo principal. "
        "As homografias aceitas descrevem o plano predominante e podem sofrer influência da paralaxe e de objetos móveis. "
        "A análise não fornece posição absoluta no solo nem mede isoladamente o erro do GSD.")
    (DESTINO / "protocolo_registro_visual.json").write_text(
        json.dumps(protocolo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(resumo.to_string(index=False), flush=True)


if __name__ == "__main__":
    executar()
