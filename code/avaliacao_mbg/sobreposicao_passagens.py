"""Sobreposição no solo entre a entrada da malha e a primeira travessia de quadrante."""
import json
from collections import Counter, defaultdict
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd

from adaptativa import METRICA, MOTOR, VOOS
from dados import CACHE, GSD, OUT, folder
from memoria_sim import T, load_pose
from metricas import CLASSES, box_ious, ground, labels
from passagens import carregar_voo, melhor_recorte, passagens_do_voo


DESTINO = OUT / "quadrantes" / "sobreposicao_passagens"
ORIGENS = {"media_do_voo": CACHE / "quadrantes_v1",
           "primeiro_gps": CACHE / "quadrantes_primeiro_gps_v1"}
CHAVE = ["quadro_entrada", "recorte_entrada", "quadro_travessia", "recorte_travessia"]
COORDS = ["x1", "y1", "x2", "y2"]


def reconstruir(pose, tamanho, tiles, mascara):
    """Reconstrói todos os disparos e confere sua igualdade com o plano executado."""
    w, h = tamanho
    usado = np.zeros_like(mascara)
    contagem, pares = Counter(), defaultdict(list)
    for p in passagens_do_voo(pose, tamanho, tiles):
        q0, fim = p["quadro_entrada"], p["quadro_saida"]
        pos = pose[q0:fim + 1]
        de = (p["leste_m"] - pos[:, 0]) / GSD
        dn = (p["norte_m"] - pos[:, 1]) / GSD
        c, s = np.cos(pos[:, 2]), np.sin(pos[:, 2])
        u, v = w / 2 + c * de - s * dn, h / 2 - s * de - c * dn
        quadrante = (u >= w / 2).astype(int) + 2 * (v >= h / 2)
        indices = np.sort(np.unique(quadrante, return_index=True)[1])
        enviados = []
        for t in indices:
            j = melhor_recorte(u[t], v[t], tiles)
            usado[q0 + t, j] = True
            enviados.append(j)
        contagem["passagens"] += 1
        if q0 == 0:
            contagem["ja_visiveis_no_inicio"] += 1
            contagem["iniciais_com_travessia"] += int(len(indices) > 1)
        elif len(indices) == 1:
            contagem["entradas_sem_travessia"] += 1
        else:
            contagem["entradas_com_travessia"] += 1
            key = (q0, enviados[0], q0 + indices[1], enviados[1])
            pares[key].append((p["i"], p["j"], p["passagem"]))
    np.testing.assert_array_equal(usado, mascara)
    assert contagem["passagens"] == sum(contagem[k] for k in
        ("ja_visiveis_no_inicio", "entradas_sem_travessia", "entradas_com_travessia"))
    return contagem, pares


def poligono(box, pose, tamanho):
    x1, y1, x2, y2 = box[:4]
    pontos = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return ground(np.tile(pontos, (1, 2)), pose, tamanho, GSD)


def sobrepor(a, b):
    # A origem local da dupla preserva a precisão do cálculo de interseção em float32.
    origem = a[0].copy()
    a, b = (a - origem).astype("float32"), (b - origem).astype("float32")
    area_a, area_b = cv2.contourArea(a), cv2.contourArea(b)
    assert min(area_a, area_b) > 0
    comum, _ = cv2.intersectConvexConvex(a, b)
    comum = min(max(float(comum), 0.), area_a, area_b)
    return comum / (area_a + area_b - comum), comum / area_a


def associar(gt, pred):
    """Pareia em pixels do mesmo quadro, uma vez por anotação e por detecção no recorte."""
    pareadas = {}
    for cls in np.unique(gt[:, 5]):
        refs = gt[gt[:, 5] == cls]
        candidatas = pred[(pred[:, 5] == cls) & (pred[:, 4] >= METRICA["conf"])]
        candidatas = candidatas[np.argsort(-candidatas[:, 4], kind="stable")[:100]]
        overlaps = box_ious(candidatas, refs)
        for i, caixa in enumerate(candidatas):
            k = overlaps[i].argmax()
            if overlaps[i, k] >= .5:
                pareadas[int(cls), int(refs[k, 4])] = caixa
                overlaps[:, k] = -1
    return pareadas


def coordenadas(linha, nome, box):
    for coord, valor in zip(COORDS, box[:4]):
        linha[f"{nome}_{coord}"] = valor


def medidas_voo(origem, video):
    cache = ORIGENS[origem]
    pose_mapa, tamanho, tiles, fps = carregar_voo(video)
    pose = pose_mapa
    if origem == "primeiro_gps":
        log, = (folder(video) / "log").glob("*.csv")
        pose = np.column_stack(load_pose(log, len(pose), fps, origem_primeiro_gps=True))
    with np.load(cache / f"{video}.npz") as plano:
        frames, masks, pedidos = plano["frames"], plano["masks"], plano["requested"]
        np.testing.assert_array_equal(tiles, plano["tiles"])
        np.testing.assert_array_equal(tamanho, plano["size"])
        np.testing.assert_array_equal(pose_mapa[frames], plano["pose"])
        assert GSD == plano["gsd"] and len(pose) == plano["total_frames"]
    with np.load(cache / MOTOR / f"{video}.npz") as rede:
        np.testing.assert_array_equal(rede["frames"], frames)
        np.testing.assert_array_equal(rede["computed"], pedidos)
        det, offsets = rede["det"], rede["offsets"]
        assert len(offsets) == len(frames) * len(tiles) + 1 and offsets[-1] == len(det)
    usado = np.zeros((len(pose), len(tiles)), dtype=bool)
    usado[frames] = masks[:, :, 0]
    contagem, pares = reconstruir(pose, tamanho, tiles, usado)
    anotadas = labels(video, {"frames": frames})
    frame_idx = {f: i for i, f in enumerate(frames)}

    @lru_cache(maxsize=None)
    def vista(q, j):
        assert usado[q, j]
        x, y = tiles[j]
        limites = np.array([x, y, x + T, y + T])
        idx = frame_idx[q]
        k = idx * len(tiles) + j
        pred = det[offsets[k]:offsets[k + 1]].astype(float)
        pred[:, :4] += [x, y, x, y]
        gt = anotadas[idx]
        marcas = associar(gt, pred)
        centros = (gt[:, :2] + gt[:, 2:4]) / 2
        dentro = ((centros >= limites[:2]) & (centros < limites[2:])).all(axis=1)
        ref = {}
        for box in gt[dentro]:
            chave = (int(box[5]), int(box[4]))
            assert chave not in ref
            inteira = (box[:2] >= limites[:2]).all() and (box[2:4] <= limites[2:]).all()
            ref[chave] = box, marcas.get(chave), bool(inteira)
        return poligono(limites, pose_mapa[q], tamanho), ref

    info = dict(origem=origem, video=video)
    recortes, trilhas = [], []
    for chave, pontos in sorted(pares.items()):
        q0, j0, q1, j1 = chave
        ids = {**info, **dict(zip(CHAVE, chave))}
        a, refs_a = vista(q0, j0)
        b, refs_b = vista(q1, j1)
        iou, fracao = sobrepor(a, b)
        comuns = sorted(refs_a.keys() & refs_b.keys())
        recortes.append(dict(**ids, passagens_no_par=len(pontos),
                             pontos_i_j_passagem="|".join(",".join(map(str, p)) for p in pontos),
                             intervalo_s=(q1 - q0) / fps, iou_recortes=iou,
                             fracao_area_comum=fracao, trilhas_comuns=len(comuns)))
        for cls, tid in comuns:
            gt_a, det_a, inteira_a = refs_a[cls, tid]
            gt_b, det_b, inteira_b = refs_b[cls, tid]
            linha = dict(**ids, classe=CLASSES[cls], trilha=tid, iou_recortes=iou,
                         anotacao_inteira_nos_dois=int(inteira_a and inteira_b))
            for nome, box_a, box_b in [("anot", gt_a, gt_b), ("det", det_a, det_b)]:
                for lado, box in [("entrada", box_a), ("travessia", box_b)]:
                    if box is not None:
                        coordenadas(linha, f"{nome}_{lado}", box)
                        if nome == "det":
                            linha[f"confianca_{lado}"] = box[4]
                if box_a is not None and box_b is not None:
                    pa = poligono(box_a, pose_mapa[q0], tamanho)
                    pb = poligono(box_b, pose_mapa[q1], tamanho)
                    sufixo = "anotacoes" if nome == "anot" else "deteccoes"
                    linha[f"iou_{sufixo}"] = sobrepor(pa, pb)[0]
                    linha[f"deslocamento_{sufixo}_m"] = np.linalg.norm(pa.mean(axis=0) - pb.mean(axis=0))
            trilhas.append(linha)
    vista.cache_clear()
    return dict(**info, **contagem, recortes_enviados=int(usado.sum())), recortes, trilhas


def descrever(serie, nome):
    positivos = serie[serie > 0]
    return {f"{nome}_media": serie.mean(), f"{nome}_mediana": serie.median(),
            f"{nome}_zeros": int((serie == 0).sum()),
            f"{nome}_media_positivos": positivos.mean(),
            f"{nome}_ge_050": int((serie >= .5).sum())}


def resumir(passagens, recortes, trilhas):
    linhas = []
    for origem, r_origem in recortes.groupby("origem", sort=False):
        for video in [*VOOS, "total"]:
            def selecionar(tabela):
                do_grupo = tabela[tabela.origem == origem]
                return do_grupo if video == "total" else do_grupo[do_grupo.video == video]
            p, r, t = selecionar(passagens), selecionar(r_origem), selecionar(trilhas)
            ok = t.dropna(subset=["iou_deteccoes"])
            entrada, travessia = t.confianca_entrada.notna(), t.confianca_travessia.notna()
            ids = ["video", "classe", "trilha"]
            somas = p.drop(columns=["origem", "video"]).sum().to_dict()
            assert len(t) == r.trilhas_comuns.sum()
            assert len(ok) == (entrada & travessia).sum()
            assert r.passagens_no_par.sum() == p.entradas_com_travessia.sum()
            macro = ok.groupby(ids).iou_deteccoes.mean()
            linhas.append(dict(origem=origem, video=video, **somas, pares_recortes=len(r),
                               pares_com_trilha=int((r.trilhas_comuns > 0).sum()),
                               oportunidades=len(t), trilhas_distintas=len(t[ids].drop_duplicates()),
                               deteccao_ambas=len(ok), deteccao_so_entrada=int((entrada & ~travessia).sum()),
                               deteccao_so_travessia=int((~entrada & travessia).sum()),
                               deteccao_nenhuma=int((~entrada & ~travessia).sum()),
                               trilhas_detectadas_nas_duas=len(macro), iou_det_media_por_trilha=macro.mean(),
                               tempo_entre_vistas_mediana_s=r.intervalo_s.median(),
                               fracao_area_comum_media=r.fracao_area_comum.mean(),
                               **descrever(r.iou_recortes, "iou_rec"),
                               **descrever(t.iou_anotacoes, "iou_anot_todas"),
                               **descrever(ok.iou_anotacoes, "iou_anot_pareadas"),
                               **descrever(ok.iou_deteccoes, "iou_det"),
                               deslocamento_det_mediana_m=ok.deslocamento_deteccoes_m.median(),
                               deslocamento_anot_pareadas_mediana_m=ok.deslocamento_anotacoes_m.median()))
    return pd.DataFrame(linhas)


def executar():
    passagens, recortes, trilhas = [], [], []
    for origem in ORIGENS:
        for video in VOOS:
            p, r, t = medidas_voo(origem, video)
            passagens.append(p)
            recortes.extend(r)
            trilhas.extend(t)
            print(origem, video, p["passagens"], len(r), len(t), flush=True)
    passos, pares, oportunidades = map(pd.DataFrame, (passagens, recortes, trilhas))
    chave = ["origem", "video", *CHAVE]
    assert not pares.duplicated(chave).any()
    assert not oportunidades.duplicated(chave + ["classe", "trilha"]).any()
    resumo = resumir(passos, pares, oportunidades)
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nome, dados in [("pares_recortes", pares), ("pares_trilhas", oportunidades), ("resumo", resumo)]:
        tmp = DESTINO / f"{nome}.partial.csv"
        dados.to_csv(tmp, index=False, float_format="%.8f")
        tmp.replace(DESTINO / f"{nome}.csv")
    protocolo = {
        "voos": list(VOOS), "altura_nominal_m": 40, "gsd_m_pixel": GSD,
        "lado_recorte_px": T, "motor": MOTOR, "pesos_sha256": METRICA["pesos_sha256"],
        "origens_da_malha": list(ORIGENS),
        "par": "Entrada posterior ao quadro zero e primeira visita a outro quadrante, dentro da mesma passagem contínua. "
               "Excluem-se pontos já visíveis no primeiro quadro e entradas que não chegam a outro quadrante. "
               "Conta-se uma vez cada par físico de quadros e recortes, mesmo se mais de um ponto gerou o par. "
               "Conferem-se todos os pedidos reconstruídos, inclusive quadrantes posteriores, com a máscara efetivamente executada.",
        "oportunidade": "Mesma classe e ID de trilha anotada no vídeo; centro da caixa anotada dentro das duas respectivas janelas. "
                        "Uma observação por par físico e trilha. A mesma trilha pode contribuir em pares diferentes. "
                        "A tabela também identifica anotações inteiras nos dois recortes.",
        "deteccao": f"Predições individuais do recorte efetivamente inferido, antes da supressão entre recortes. "
                    f"Em cada recorte, confiança >= {METRICA['conf']}, até 100 predições por classe em ordem de confiança, "
                    "casamento guloso um a um com anotações da mesma classe por IoU no mesmo quadro >= 0,50. "
                    "O teste espacial no terreno não intervém na associação. Uma vista sem detecção aparece em branco.",
        "projecao": "Quatro cantos de cada caixa ou recorte em pixels do quadro original, projetados na mesma base local "
                    "das métricas do voo por GSD constante, posição e guinada. IoU = interseção/união dos polígonos no solo. "
                    "Fração de área comum = interseção/área de um recorte. Índices de quadros e recortes começam em zero.",
        "estatistica": "Médias e medianas usam todas as IoUs definidas, inclusive zeros. Média de positivos exclui apenas os zeros. "
                       "IoU do detector requer detecção pareada em ambas as vistas. As anotações são avaliadas em todas as "
                       "oportunidades e separadamente no subconjunto de detecções duplas. Média por trilha dá peso igual a cada "
                       "ID com ao menos uma comparação válida. Não juntar as origens, pois reusam os mesmos voos.",
        "limite": "Concordância das projeções de caixas visíveis. As anotações controlam o efeito conjunto das imagens, da telemetria "
                  "e da projeção. Sem pontos de controle topográfico, não há medida independente da área real nem atribuição isolada ao GSD."
    }
    (DESTINO / "protocolo.json").write_text(json.dumps(protocolo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(resumo[resumo.video == "total"].T.to_string(), flush=True)


if __name__ == "__main__":
    executar()
