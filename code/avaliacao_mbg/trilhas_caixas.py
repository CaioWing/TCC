"""Identidade das caixas por estratégia e correspondências de cor para o vídeo."""
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from metricas import associate_observations, box_ious, ground


def trilhas_geograficas(clip, quadros, raio):
    memorias = {}
    trilhas = []
    for caixas, pose in zip(quadros, clip["pose"], strict=True):
        solo = ground(caixas, pose, clip["size"], clip["gsd"])
        chaves = [None] * len(caixas)
        for classe in np.unique(caixas[:, 5]).astype(int):
            if classe not in memorias:
                memorias[classe] = np.empty((0, 2)), np.empty(0, dtype=int)
            pos = np.flatnonzero(caixas[:, 5] == classe)
            centros, contagens, ids = associate_observations(*memorias[classe], solo[pos], raio)
            memorias[classe] = centros, contagens
            for indice, identidade in zip(pos, ids):
                chaves[indice] = (int(classe), int(identidade))
        assert len(chaves) == len(set(chaves))
        trilhas.append(chaves)
    return trilhas


def pares_na_imagem(a, b):
    """Pares da mesma classe com IoU >= 0,50, um a um dentro do mesmo quadro."""
    if not len(a) or not len(b):
        return []
    iou = box_ious(a, b)
    elegivel = (a[:, 5, None] == b[None, :, 5]) & (iou >= .5)
    # Prioriza quantos pares são válidos. Entre soluções empatadas, maximiza a IoU.
    custo = np.where(elegivel, -iou, min(iou.shape) + 1)
    linhas, colunas = linear_sum_assignment(custo)
    return [(i, j) for i, j in zip(linhas, colunas) if elegivel[i, j]]


def identidades_visuais(caixas, trilhas):
    """Alinha somente a cor após o trecho. As três memórias são independentes."""
    grupos = {}
    for painel, quadros in enumerate(trilhas):
        for quadro in quadros:
            for chave in quadro:
                item = (painel, chave)
                grupos.setdefault(item, {item})
    for a, b in combinations(range(len(caixas)), 2):
        ida, volta = defaultdict(set), defaultdict(set)
        for t, (qa, qb) in enumerate(zip(caixas[a], caixas[b], strict=True)):
            for i, j in pares_na_imagem(qa, qb):
                x, y = (a, trilhas[a][t][i]), (b, trilhas[b][t][j])
                ida[x].add(y)
                volta[y].add(x)
        for x, vizinhos in ida.items():
            if len(vizinhos) == 1:
                y, = vizinhos
                grupo = grupos[x] | grupos[y]
                if len(volta[y]) == 1 and len({p for p, _ in grupo}) == len(grupo):
                    for membro in grupo:
                        grupos[membro] = grupo
    numeros = {}
    for chave, grupo in grupos.items():
        if chave not in numeros:
            novo = max(numeros.values(), default=-1) + 1
            numeros.update(dict.fromkeys(grupo, novo))
    return [[[numeros[painel, chave] for chave in quadro] for quadro in quadros]
            for painel, quadros in enumerate(trilhas)]


def paleta(n):
    # Máxima distância de cor em Lab, reservando o laranja dos contornos de recorte.
    hsv = np.array([(h, s, v) for h in range(0, 180, 4) for s in (120, 200, 250)
                    for v in (190, 250)], dtype=np.uint8)[None, :, :]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255., cv2.COLOR_RGB2Lab)[0]
    reservas = np.array([[(255, 163, 46), (20, 29, 33)]], dtype=np.float32) / 255.
    referencias = cv2.cvtColor(reservas, cv2.COLOR_RGB2Lab)[0]
    dist = np.linalg.norm(lab[:, None] - referencias, axis=2).min(axis=1)
    cores = []
    for _ in range(n):
        indice = int(np.argmax(dist))
        cores.append(tuple(map(int, rgb[0, indice])))
        dist = np.minimum(dist, np.linalg.norm(lab - lab[indice], axis=1))
    assert len(set(cores)) == n
    return cores


def conferir_anotacoes(clip, caixas, trilhas):
    """Usa IDs anotados somente depois da associação, para auditar as repetições."""
    por_objeto, por_trilha = defaultdict(list), defaultdict(set)
    for qa, chaves, gt in zip(caixas, trilhas, clip["gt"], strict=True):
        for i, j in pares_na_imagem(qa, gt):
            objeto = int(gt[j, 5]), int(gt[j, 4])
            por_objeto[objeto].append(chaves[i])
            por_trilha[chaves[i]].add(objeto)
    repetidos = [ids for ids in por_objeto.values() if len(ids) > 1]
    return dict(trilhas=len({ch for quadro in trilhas for ch in quadro}),
                objetos_com_par=len(por_objeto), objetos_repetidos=len(repetidos),
                repetidos_com_id_unico=sum(len(set(ids)) == 1 for ids in repetidos),
                trilhas_com_mais_de_um_objeto=sum(len(objs) > 1 for objs in por_trilha.values()))
