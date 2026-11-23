"""Associa caixas interrompidas na borda de recortes vizinhos dentro de um quadro."""
from collections import Counter
from functools import lru_cache

import numpy as np

from dados import tiles_grid
from memoria_sim import T
from metricas import indices_nms_quadro

# Valores exploratórios fixados antes da leitura dos resultados.
MARGEM_PX = 4
ALINHAMENTO = .7
PROPORCAO_LADOS = .6
INTERSECAO_MENOR = .6
CONFIANCA_PARTE = .15
CONFIANCA_OBSERVACAO = .5


@lru_cache(maxsize=8)
def grade_fronteiras(largura, altura):
    origens = tiles_grid(largura, altura)
    vizinhos = []
    for eixo in (0, 1):
        linhas = {}
        for i, origem in enumerate(origens):
            linhas.setdefault(origem[1 - eixo], []).append(i)
        for linha in linhas.values():
            ordem = sorted(linha, key=lambda i: origens[i][eixo])
            for i, j in zip(ordem, ordem[1:]):
                fim, inicio = origens[i][eixo] + T, origens[j][eixo]
                if inicio <= fim:
                    vizinhos.append((i, j, eixo, fim, inicio))
    return origens, vizinhos


def compativeis(a, b, eixo, fim, inicio):
    if a[5] != b[5] or max(a[4], b[4]) < CONFIANCA_OBSERVACAO:
        return False
    outro = 1 - eixo
    tam_a, tam_b = a[outro + 2] - a[outro], b[outro + 2] - b[outro]
    comum = min(a[outro + 2], b[outro + 2]) - max(a[outro], b[outro])
    if min(tam_a, tam_b) <= 0 or min(tam_a, tam_b) < PROPORCAO_LADOS * max(tam_a, tam_b):
        return False
    if comum < ALINHAMENTO * min(tam_a, tam_b):
        return False
    toca_a, toca_b = a[eixo + 2] >= fim - MARGEM_PX, b[eixo] <= inicio + MARGEM_PX
    if fim == inicio:
        return toca_a and toca_b
    # Na faixa de sobreposição, uma das duas vistas pode mostrar o objeto inteiro.
    if not ((toca_a and b[eixo + 2] > fim) or (toca_b and a[eixo] < inicio)):
        return False
    area_a = tam_a * (a[eixo + 2] - a[eixo])
    area_b = tam_b * (b[eixo + 2] - b[eixo])
    inter = comum * max(0., min(a[eixo + 2], b[eixo + 2]) - max(a[eixo], b[eixo]))
    return min(area_a, area_b) > 0 and inter >= INTERSECAO_MENOR * min(area_a, area_b)


def quadro_fronteiras(tiles, mascara, grade):
    origens, vizinhos = grade
    if len(tiles) != len(origens) or len(mascara) != len(origens):
        raise ValueError("A grade real, a máscara e as predições devem coincidir.")
    usados = [i for i in np.flatnonzero(mascara) if len(tiles[i])]
    if not usados:
        vazio = np.empty((0, 6), np.float32)
        return vazio, vazio, []
    fontes = np.repeat(usados, [len(tiles[i]) for i in usados])
    caixas = np.concatenate([tiles[i] for i in usados])
    manter = indices_nms_quadro(caixas)
    caixas, fontes = caixas[manter], fontes[manter]
    possiveis = np.flatnonzero(caixas[:, 4] >= CONFIANCA_PARTE)
    por_recorte = {}
    for p in possiveis:
        por_recorte.setdefault(fontes[p], []).append(p)
    pares = []
    for i, j, eixo, fim, inicio in vizinhos:
        if i not in por_recorte or j not in por_recorte:
            continue
        # Descarta combinações fora da faixa compartilhada antes de comparar as caixas.
        for p in por_recorte[i]:
            a = caixas[p]
            if a[eixo + 2] < inicio - MARGEM_PX:
                continue
            for q in por_recorte[j]:
                b = caixas[q]
                if b[eixo] <= fim + MARGEM_PX and compativeis(a, b, eixo, fim, inicio):
                    pares.append((p, q, i, j, eixo))
    frequencias = Counter(x for p, q, *_ in pares for x in (p, q))
    pares = [par for par in pares if frequencias[par[0]] == frequencias[par[1]] == 1]
    if not pares:
        return caixas, caixas, []
    resultado = caixas.copy()
    ativas = np.ones(len(caixas), dtype=bool)
    registros = []
    for p, q, i, j, eixo in pares:
        partes = caixas[[p, q]]
        melhor, outra = (p, q) if caixas[p, 4] >= caixas[q, 4] else (q, p)
        uniao = resultado[melhor]
        uniao[:2], uniao[2:4] = partes[:, :2].min(axis=0), partes[:, 2:4].max(axis=0)
        ativas[outra] = False
        registros.append(dict(i=i, j=j, eixo=eixo, a=caixas[p], b=caixas[q],
                              uniao=uniao, indice=melhor))
    indices = np.flatnonzero(ativas)
    finais = indices[indices_nms_quadro(resultado[indices])]
    ids = {k: i for i, k in enumerate(finais)}
    for r in registros:
        r["id_observacao"] = ids.get(r.pop("indice"), -1)
    return caixas, resultado[finais], registros


def selected_tiles_fronteiras(clip, masks):
    """Mesma interface de selected_tiles. O ID da observação só vale no próprio quadro."""
    if len(masks) != len(clip["tiles"]):
        raise ValueError("A seleção e as previsões precisam usar os mesmos quadros.")
    grade = grade_fronteiras(*map(int, clip["size"]))
    return [quadro_fronteiras(tiles, mask, grade)[1] for tiles, mask in zip(clip["tiles"], masks)]
