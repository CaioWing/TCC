"""Seleção geográfica com entrada, saída e um número configurável de quadros intermediários."""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from avaliar import ratios
from adaptativa import MOTOR, VOOS, verificar_pesos
from dados import GSD
from fronteiras import selected_tiles_fronteiras
from memoria_sim import T
from metricas import load_clips
from passagens import (CONJUNTOS, DESTINO, LADO, MASCARAS, carregar_voo,
                       pontuar_mascara, resumir, salvar)
from seguimento import FASES, temporal

SAIDA = "curva"
INTERMEDIARIOS = [*range(11), 14, 22, 30, 46, 62, 94, 126, 190, 254, 510]


def quadros_geograficos(distancias, k_intermediarios):
    """Índices da entrada, dos quadros intermediários por distância e da saída."""
    if not isinstance(k_intermediarios, (int, np.integer)) or k_intermediarios < 0:
        raise ValueError("O número de quadros intermediários deve ser um inteiro não negativo.")
    n = len(distancias)
    k = int(k_intermediarios) + 2
    if k >= n:
        return np.arange(n)
    if distancias[-1] - distancias[0] < 1e-6:
        distancias = np.arange(n)
    alvos = np.linspace(distancias[0], distancias[-1], k)
    escolhidos = [0]
    for j in range(1, k - 1):
        inicio = escolhidos[-1] + 1
        ultimo = n - k + j
        melhor = np.abs(distancias[inicio:ultimo + 1] - alvos[j]).argmin()
        escolhidos.append(inicio + int(melhor))
    return np.array([*escolhidos, n - 1])


def quadros_da_passagem(distancias, k):
    """Lê o total de vistas do protocolo arquivado, incluindo o controle só na entrada."""
    if k == 1:
        return np.arange(min(1, len(distancias)))
    return quadros_geograficos(distancias, k - 2)


def mascaras_teste(video, eventos, quadros, metas):
    pose, tamanho, tiles, _ = carregar_voo(video)
    avaliados = np.zeros(len(pose), dtype=bool)
    avaliados[quadros] = True
    percurso = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1))]
    masks = {k: np.zeros((len(pose), len(tiles)), dtype=bool) for k in metas}
    for ev in eventos.itertuples():
        a, b = ev.quadro_entrada, ev.quadro_saida + 1
        locais = np.flatnonzero(avaliados[a:b])
        if not len(locais):
            continue
        f = locais + a
        de, dn = (ev.i * LADO - pose[f, 0]) / GSD, (ev.j * LADO - pose[f, 1]) / GSD
        c, s = np.cos(pose[f, 2]), np.sin(pose[f, 2])
        uv = np.column_stack((tamanho[0]/2 + c*de - s*dn, tamanho[1]/2 - s*de - c*dn))
        delta = uv[:, None, :] - tiles
        dentro = ((delta >= 0) & (delta < T)).all(axis=2)
        assert dentro.any(axis=1).all()
        dist2 = ((delta - T / 2) ** 2).sum(axis=2)
        indices = np.full(b - a, -1, dtype=int)
        indices[locais] = np.where(dentro, dist2, np.inf).argmin(axis=1)
        for k, mask in masks.items():
            pos = quadros_da_passagem(percurso[a:b], k)
            assert len(pos) == min(k, b-a) and (np.diff(pos) > 0).all()
            pos = pos[indices[pos] >= 0]
            mask[a + pos, indices[pos]] = True
    with np.load(MASCARAS / f"{video}.npz") as original:
        for k, nome in [(1, "entrada"), (2, "extremos"), (3, "tres")]:
            np.testing.assert_array_equal(masks[k][quadros], original[nome][quadros])
    for mask in masks.values():
        assert not (mask & ~masks[metas[-1]]).any()
    return masks


def avaliar_voo(video, eventos, metas):
    trechos = [(grupo, clip) for grupo, cache in CONJUNTOS.items()
               for clip in load_clips(video, MOTOR, cache=cache)]
    quadros = np.concatenate([clip["frames"] for _, clip in trechos])
    assert len(np.unique(quadros)) == len(quadros)
    masks = mascaras_teste(video, eventos, quadros, metas)
    linhas = []
    for grupo, clip in trechos:
        memo = {}
        anteriores = {}
        global_id = VOOS.index(video) * 2 + clip["clip"] - 1
        for k in [0, *metas]:
            if k == 0:
                selecoes = [("integral", -1., np.ones((len(clip["frames"]), 24), dtype=bool))]
            else:
                geografica = masks[k][clip["frames"]]
                orcamento = int(geografica.sum())
                selecoes = [("geografica", -1., geografica)]
                selecoes.extend(("temporal", fase, temporal(geografica.shape, orcamento, fase, global_id))
                                 for fase in FASES)
            for estrategia, fase, mask in selecoes:
                contagem, acertos = pontuar_mascara(clip, mask, selected_tiles_fronteiras, memo)
                if estrategia == "integral":
                    integral = acertos
                else:
                    assert int(mask.sum()) == orcamento
                anterior = anteriores.get((estrategia, fase))
                anteriores[estrategia, fase] = acertos
                linhas.append(dict(video=video, conjunto=grupo, trecho=clip["clip"], k=k,
                                   estrategia=estrategia, fase=fase, recortes=int(mask.sum()),
                                   quadros_usados=int(mask.any(axis=1).sum()),
                                   novos_vs_k_anterior=len(acertos - anterior) if anterior is not None else None,
                                   perdidos_vs_k_anterior=len(anterior - acertos) if anterior is not None else None,
                                   comuns_integral=len(acertos & integral), so_metodo=len(acertos - integral),
                                   so_integral=len(integral - acertos), **ratios(contagem.sum(axis=0))))
    print(video, "avaliado", flush=True)
    return linhas


def avaliar():
    verificar_pesos()
    eventos = pd.read_csv(DESTINO / "eventos.csv")
    maximo = int(eventos.quadros_visiveis.max())
    # Os arquivos do ensaio usam o total de vistas. A meta 1 é o controle de entrada.
    metas = sorted({1, maximo} | {k + 2 for k in INTERMEDIARIOS if k + 2 < maximo})
    (DESTINO / SAIDA).mkdir(exist_ok=True)
    protocolo = dict(metas=metas, todos_os_instantes=maximo, motor=MOTOR,
                     origem_grade="A mesma do ensaio principal de passagens.",
                     marcos="Entrada para K=1. Para K>=2, extremos e frações uniformes da distância "
                            "percorrida. Em repouso, usar o tempo. Usar quadros distintos em ordem; "
                            "se a passagem for mais curta que K, usar todos os quadros disponíveis.",
                     avaliacao="24 trechos, quatro fases temporais pareadas por orçamento em cada trecho. "
                               "Reintegração idêntica e mesmo modelo e métricas do ensaio por passagem.")
    (DESTINO / SAIDA / "protocolo.json").write_text(json.dumps(protocolo, ensure_ascii=False, indent=2)
                                                     + "\n", encoding="utf-8")
    with ProcessPoolExecutor(max_workers=3) as pool:
        trabalhos = [pool.submit(avaliar_voo, video, eventos[eventos.video == video], metas)
                     for video in VOOS]
        df = pd.DataFrame([r for trabalho in trabalhos for r in trabalho.result()])
    base = pd.read_csv(DESTINO / "por_trecho_fronteiras.csv").query("plano != 'meio'")
    chave = ["conjunto", "video", "trecho", "k", "estrategia", "fase"]
    campos = ["vp", "fp", "fn", "comuns_integral", "so_integral", "so_metodo", "recortes", "quadros_usados"]
    pd.testing.assert_frame_equal(df[df.k <= 3].set_index(chave)[campos].sort_index(),
                                  base.set_index(chave)[campos].sort_index())
    pares = df[df.k > 0].groupby(["video", "conjunto", "trecho", "k"])
    assert len(pares) == 24 * len(metas) and pares.size().eq(5).all()
    assert pares.recortes.nunique().eq(1).all() and pares.referencias.nunique().eq(1).all()
    assert (df.vp == df.comuns_integral + df.so_metodo).all()
    salvar(df, f"{SAIDA}/por_trecho.csv")
    chaves = ["k", "estrategia", "fase"]
    transicoes = df.groupby(chaves)[["novos_vs_k_anterior", "perdidos_vs_k_anterior"]].sum(min_count=1)
    salvar(resumir(df, chaves).join(transicoes, on=chaves), f"{SAIDA}/resumo.csv")


if __name__ == "__main__":
    avaliar()
