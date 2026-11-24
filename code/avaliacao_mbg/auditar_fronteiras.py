"""Confere a reintegração de caixas usando IDs anotados apenas para avaliação."""
import json

import numpy as np
import pandas as pd

from adaptativa import METRICA, MOTOR, TEMP, VOOS, verificar_pesos
from avaliar import ratios
from dados import CACHE, OUT, SPLITS
from fronteiras import (ALINHAMENTO, CONFIANCA_OBSERVACAO, CONFIANCA_PARTE,
                        INTERSECAO_MENOR, MARGEM_PX, PROPORCAO_LADOS,
                        grade_fronteiras, quadro_fronteiras)
from memoria_sim import T
from metricas import CLASSES, box_ious, focus_counts, load_clips, projected, selected_tiles

DESTINO = OUT / "fronteiras"
IOU_AUDITORIA = .5
CONTADORES = ("unioes", "unioes_retidas", "quadros_com_uniao", "mesma_trilha",
              "trilhas_distintas", "ambiguo", "sem_identificacao_das_duas_partes",
              "acertos_quadro_recuperados", "acertos_quadro_perdidos")


def ids_parte(caixa, origem, anotadas):
    """Compara o fragmento com a porção realmente anotada dentro do seu recorte."""
    cls = anotadas[anotadas[:, 5] == caixa[5]].copy()
    cls[:, :2] = np.maximum(cls[:, :2], origem)
    cls[:, 2:4] = np.minimum(cls[:, 2:4], np.array(origem) + T)
    cls = cls[np.all(cls[:, 2:4] > cls[:, :2], axis=1)]
    ious = box_ious(caixa[None], cls)[0]
    return sorted(map(int, cls[ious >= IOU_AUDITORIA, 4]))


def diagnostico(r, origens, gt):
    p, q = [ids_parte(r[k], origens[r[j]], gt) for k, j in (("a", "i"), ("b", "j"))]
    if len(p) > 1 or len(q) > 1:
        status = "ambiguo"
    elif not p or not q:
        status = "sem_identificacao_das_duas_partes"
    elif p == q:
        status = "mesma_trilha"
    else:
        status = "trilhas_distintas"
    info = dict(eixo="vertical" if r["eixo"] == 0 else "horizontal",
                recorte_a=int(r["i"]), recorte_b=int(r["j"]), ids_a="|".join(map(str, p)),
                ids_b="|".join(map(str, q)), classe=CLASSES[int(r["uniao"][5])],
                situacao=status, id_observacao=r["id_observacao"])
    for nome in ("a", "b", "uniao"):
        info[nome] = json.dumps([round(float(v), 3) for v in r[nome][:4]])
        info[f"escore_{nome}"] = float(r[nome][4])
    if status == "mesma_trilha":
        ref = gt[(gt[:, 4] == p[0]) & (gt[:, 5] == r["a"][5])]
        assert len(ref) == 1
        iou = box_ious(np.stack([r["a"], r["b"], r["uniao"]]), ref)[:, 0]
        info.update(trilha=p[0], anotacao=json.dumps(list(ref[0, :4])),
                    iou_a=float(iou[0]), iou_b=float(iou[1]), iou_uniao=float(iou[2]))
        info["recuperou_caixa"] = int(max(iou[:2]) < IOU_AUDITORIA <= iou[2] and r["id_observacao"] >= 0)
    return info


def acertos_quadro(caixas, gt):
    ativas = caixas[caixas[:, 4] >= CONFIANCA_OBSERVACAO]
    ativas = ativas[np.argsort(-ativas[:, 4], kind="stable")]
    usados = set()
    if len(ativas) and len(gt):
        ious = box_ious(ativas, gt)
        ious[ativas[:, 5, None] != gt[None, :, 5]] = -1.
        for row in ious:
            if usados:
                row[list(usados)] = -1.
            melhor = int(row.argmax())
            if row[melhor] >= IOU_AUDITORIA:
                usados.add(melhor)
    return usados, len(ativas) - len(usados)


def auditar():
    verificar_pesos()
    DESTINO.mkdir(exist_ok=True)
    protocolo = dict(margem_px=MARGEM_PX, alinhamento_minimo=ALINHAMENTO,
                     razao_lados_minima=PROPORCAO_LADOS, ios_minima=INTERSECAO_MENOR,
                     escore_minimo_parte=CONFIANCA_PARTE, escore_minimo_uniao=CONFIANCA_OBSERVACAO,
                     iou_identidade_fragmento=IOU_AUDITORIA, origem_parametros="Exploratórios, fixados antes da auditoria.",
                     estrategia="Integral", motor_validacao="torchscript_fp32", motor_teste=MOTOR,
                     criterio="NMS global por classe antes e depois da união. Arestas ambíguas "
                              "permanecem separadas. Escore da união igual ao maior dos dois.",
                     identidade="IoU da caixa de cada fragmento com a anotação recortada à área "
                                "de origem. IDs CVAT usados exclusivamente na auditoria.",
                     sha256=METRICA["pesos_sha256"])
    (DESTINO / "protocolo.json").write_text(json.dumps(protocolo, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    eventos, trechos = [], []
    for conjunto, voos, cache, motor in [
        ("validacao", SPLITS["voos"]["val"], CACHE, "torchscript_fp32"),
        ("iniciais", VOOS, CACHE, MOTOR), ("novos", VOOS, TEMP, MOTOR)
    ]:
        for video in voos:
            for clip in load_clips(video, motor, cache=cache):
                grade = grade_fronteiras(*map(int, clip["size"]))
                mask = np.ones((len(clip["frames"]), len(grade[0])), dtype=bool)
                base = selected_tiles(clip, mask)
                com_uniao = []
                ident = dict(conjunto=conjunto, video=video, trecho=clip["clip"])
                diag = dict.fromkeys(CONTADORES, 0)
                stats = {"nms": np.zeros(3, dtype=int), "reintegracao": np.zeros(3, dtype=int)}
                for k, tiles in enumerate(clip["tiles"]):
                    gt = clip["gt"][k]
                    antes, depois, pares = quadro_fronteiras(tiles, mask[k], grade)
                    np.testing.assert_array_equal(base[k], antes)
                    com_uniao.append(depois)
                    acertos_antes, fp_antes = acertos_quadro(antes, gt)
                    if pares:
                        acertos_depois, fp_depois = acertos_quadro(depois, gt)
                    else:
                        acertos_depois, fp_depois = acertos_antes, fp_antes
                    for metodo, acertos, fp in [("nms", acertos_antes, fp_antes),
                                                ("reintegracao", acertos_depois, fp_depois)]:
                        stats[metodo] += [len(acertos), fp, len(gt) - len(acertos)]
                    diag["quadros_com_uniao"] += bool(pares)
                    diag["acertos_quadro_recuperados"] += len(acertos_depois - acertos_antes)
                    diag["acertos_quadro_perdidos"] += len(acertos_antes - acertos_depois)
                    for r in pares:
                        rel = diagnostico(r, grade[0], gt)
                        diag[rel["situacao"]] += 1
                        diag["unioes"] += 1
                        diag["unioes_retidas"] += int(r["id_observacao"] >= 0)
                        eventos.append(ident | dict(quadro=int(clip["frames"][k])) | rel)
                for metodo, saidas in [("nms", base), ("reintegracao", com_uniao)]:
                    ct = focus_counts(clip, projected(clip, saidas), *(
                        METRICA[k] for k in ("conf", "raio_m", "dmax_m"))).sum(axis=0)
                    tq, fq, nq = stats[metodo]
                    row = ident | dict(metodo=metodo, quadros=len(base), vp_quadro=tq,
                                       fp_quadro=fq, fn_quadro=nq, **ratios(ct))
                    row.update(diag if metodo == "reintegracao" else dict.fromkeys(diag, 0))
                    trechos.append(row)
            print(conjunto, video, flush=True)
    tabela = pd.DataFrame(trechos)
    tabela.to_csv(DESTINO / "por_trecho.csv", index=False, float_format="%.8f")
    ev = pd.DataFrame(eventos)
    ev.to_csv(DESTINO / "pares.csv", index=False, float_format="%.8f")
    somar = ["quadros", "vp_quadro", "fp_quadro", "fn_quadro", "vp", "fp", "fn", *CONTADORES]
    resumo = tabela.groupby(["conjunto", "metodo"], as_index=False)[somar].sum()
    resumo.to_csv(DESTINO / "resumo.csv", index=False)
    exemplos = ev[(ev.conjunto != "validacao") & (ev.situacao == "mesma_trilha") &
                  (ev.id_observacao >= 0) & (ev.iou_uniao >= .5)]
    if len(exemplos):
        # Prefere um corte sem sobreposição e uma união que recupera o objeto inteiro.
        exemplos = exemplos.assign(vertical=exemplos.eixo == "vertical")
        ex = exemplos.sort_values(["vertical", "recuperou_caixa", "iou_uniao"],
                                  ascending=False).iloc[0].dropna().to_dict()
        for key in ("anotacao", "a", "b", "uniao"):
            ex[key] = json.loads(ex[key])
        (DESTINO / "exemplo.json").write_text(json.dumps(ex, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")


if __name__ == "__main__":
    auditar()
