"""Seguimento multivistas nos mesmos seis voos, com orçamentos iguais por trecho."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dados import CACHE, GSD, OUT, WEIGHTS, folder
from memoria_sim import load_pose, simulate
from multivistas import MemoriaFaixas, TEMP as PRIMEIRAS_MASCARAS

RESULTADOS = OUT / "seguimento"
TEMP = CACHE / "seguimento"
PLANO = json.loads((RESULTADOS / "protocolo.json").read_text(encoding="utf-8"))
METAS = PLANO["metas"]
FASES = PLANO["fases"]
VOOS = PLANO["voos"]
MOTOR = PLANO["motor"]


def subamostra(propostas, limite):
    """Retém posições distribuídas ao longo das propostas ordenadas por quadro e recorte."""
    posicoes = np.flatnonzero(propostas)
    assert 0 <= limite <= len(posicoes)
    if limite == len(posicoes):
        return propostas
    mascara = np.zeros_like(propostas)
    if limite:
        indices = (2 * np.arange(limite) + 1) * len(posicoes) // (2 * limite)
        mascara.flat[posicoes[indices]] = True
    return mascara


def temporal(shape, limite, fase, trecho_global):
    duracao, grade = shape
    assert grade == 24 and 0 <= limite <= duracao * grade
    mascara = np.zeros(shape, dtype=bool)
    n = (limite + grade - 1) // grade
    if n:
        quadros = np.floor((np.arange(n) + fase) * duracao / n).astype(int)
        mascara[quadros] = True
        resto = limite % grade
        if resto:
            parcial = quadros[trecho_global % n]
            mascara[parcial] = False
            # Sete é coprimo com 24. A origem roda entre trechos, sem consultar as imagens.
            mascara[parcial, (trecho_global + 7 * np.arange(resto)) % grade] = True
    assert mascara.sum() == limite
    return mascara


def comparacoes(geograficas, trecho_global):
    grade = geograficas[1].shape[1]
    minimo = min(int(m.sum()) for m in geograficas.values())
    comum = minimo // grade * grade
    for k, propostas in geograficas.items():
        n = int(propostas.sum())
        for regra, limite in [
            ("natural", n),
            ("inteiros", n // grade * grade),
            ("fixo", comum),
            ("fixo_bruto", minimo),
        ]:
            yield regra, k, "memoria", -1.0, subamostra(propostas, limite)
            for fase in FASES:
                yield regra, k, "temporal", fase, temporal(
                    propostas.shape, limite, fase, trecho_global
                )


def preparar(video):
    with np.load(CACHE / f"{video}.npz") as anterior:
        n, fps = int(anterior["total_frames"]), float(anterior["fps"])
        tiles, tamanho = anterior["tiles"], anterior["size"]
        antigos = anterior["frames"]
        assert np.isclose(anterior["gsd"], GSD)
    duracao = round(PLANO["janela_s"] * fps)
    starts = [round(f * n) - duracao // 2 for f in PLANO["centros_fracao"]]
    trechos = np.array([np.arange(s, s + duracao) for s in starts])
    frames = trechos.ravel()
    assert len(np.unique(frames)) == len(frames) and not np.intersect1d(frames, antigos).size
    assert 0 <= frames[0] and frames[-1] < n
    (log,) = (folder(video) / "log").glob("*.csv")
    w, h = map(int, tamanho)
    politica = MemoriaFaixas(n, tiles.tolist(), h)
    simulate(log, n, fps, w, h, GSD, on_deferred_frame=politica.frame)
    with np.load(PRIMEIRAS_MASCARAS / f"{video}.npz") as referencia:
        for k in METAS:
            assert np.array_equal(politica.selected[k][antigos], referencia[f"geo{k}"])
    requested = np.zeros((len(frames), len(tiles)), dtype=bool)
    for i, bloco in enumerate(trechos):
        geo = {k: politica.selected[k][bloco] for k in METAS}
        for *_, mascara in comparacoes(geo, VOOS.index(video) * len(starts) + i):
            requested[i * duracao : (i + 1) * duracao] |= mascara
    pose = np.column_stack(load_pose(log, n, fps))[frames]
    tmp = TEMP / f"{video}.partial.npz"
    np.savez_compressed(
        tmp,
        frames=frames,
        starts=starts,
        size=tamanho,
        tiles=tiles,
        gsd=GSD,
        pose=pose,
        fps=fps,
        requested=requested,
        masks=requested[:, :, None],
        **{f"geo{k}": politica.selected[k][frames] for k in METAS},
    )
    tmp.replace(TEMP / f"{video}.npz")
    print(video, starts, int(requested.sum()), "recortes distintos a inferir", flush=True)


def inferir():
    import torch
    from inferir import infer_video
    from motores import runner, setup

    origem = json.loads((OUT / "protocolo.json").read_text(encoding="utf-8"))
    motor = json.loads((OUT / f"motor_{MOTOR}.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == origem["pesos_sha256"]
    assert (
        hashlib.sha256((CACHE / f"{MOTOR}.opt1.engine").read_bytes()).hexdigest()
        == motor["engine_sha256"]
    )
    setup()
    with torch.inference_mode():
        run = runner(MOTOR)
        for video in VOOS:
            infer_video(video, MOTOR, run, cache=TEMP, sparse=True)


def gravar(tabela, nome):
    arquivo = RESULTADOS / nome
    temporario = arquivo.with_suffix(".partial.csv")
    tabela.to_csv(temporario, index=False, float_format="%.8f")
    temporario.replace(arquivo)


def avaliar():
    from avaliar import ratios
    from metricas import CLASSES, focus_counts, load_clips, projected, selected_tiles

    p = json.loads((OUT / "protocolo.json").read_text(encoding="utf-8"))
    params = [p[key] for key in ("conf", "raio_m", "dmax_m")]
    anteriores = pd.read_csv(OUT / f"pareamento_{MOTOR}.csv")
    rows, referencia, cadastro = [], [], []
    for v, video in enumerate(VOOS):
        vistos = anteriores.loc[anteriores.video == video]
        ids_anteriores = set(zip(vistos.classe, vistos.trilha))
        with np.load(TEMP / f"{video}.npz") as info:
            for i, clip in enumerate(load_clips(video, MOTOR, cache=TEMP)):
                idx = np.searchsorted(info["frames"], clip["frames"])
                geo = {k: info[f"geo{k}"][idx] for k in METAS}
                chaves = sorted(clip["centers"])
                refs = {
                    key: dict(
                        video=video,
                        trecho=i + 1,
                        classe=CLASSES[key[0]],
                        trilha=key[1],
                        presente_no_estudo_inicial=int((CLASSES[key[0]], key[1]) in ids_anteriores),
                    )
                    for key in chaves
                }
                cadastro.append(
                    dict(
                        video=video,
                        trecho=i + 1,
                        primeiro=int(clip["frames"][0]),
                        ultimo=int(clip["frames"][-1]),
                        fps=float(info["fps"]),
                        referencias=len(chaves),
                    )
                )
                memo = {}
                for regra, k, metodo, fase, mask in comparacoes(
                    geo, v * len(PLANO["centros_fracao"]) + i
                ):
                    assert not (mask & ~info["requested"][idx]).any()
                    key = np.packbits(mask).tobytes()
                    if key not in memo:
                        boxes = selected_tiles(clip, mask)
                        memo[key] = focus_counts(
                            clip, projected(clip, boxes), *params, return_matches=True
                        )
                    counts, acertos = memo[key]
                    met = ratios(counts.sum(axis=0))
                    assert met["referencias"] == len(chaves)
                    rows.append(
                        dict(
                            video=video,
                            trecho=i + 1,
                            orcamento=regra,
                            k=k,
                            metodo=metodo,
                            fase=fase,
                            recortes=int(mask.sum()),
                            **met,
                        )
                    )
                    for key_ref, entry in refs.items():
                        entry[f"{regra}_{metodo}{k}_{fase:g}"] = int(key_ref in acertos)
                referencia.extend(refs.values())
        print(video, "pontuado", flush=True)
    dados = pd.DataFrame(rows)
    assert dados.groupby(["video", "trecho", "orcamento", "k"]).recortes.nunique().eq(1).all()
    fixos = dados[dados.orcamento.isin(["fixo", "fixo_bruto"])]
    assert fixos.groupby(["orcamento", "video", "trecho"]).recortes.nunique().eq(1).all()
    totals = []
    for (regra, k, metodo, fase), grupo in dados.groupby(["orcamento", "k", "metodo", "fase"]):
        totals.append(
            dict(
                orcamento=regra,
                k=k,
                metodo=metodo,
                fase=fase,
                recortes=grupo.recortes.sum(),
                **ratios(grupo[["vp", "fp", "fn"]].sum()),
            )
        )
    gravar(pd.DataFrame(cadastro), "janelas.csv")
    gravar(pd.DataFrame(referencia), "pareamento.csv")
    gravar(dados, "por_trecho.csv")
    gravar(pd.DataFrame(totals), "resumo.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["preparar", "inferir", "avaliar"])
    args = ap.parse_args()
    TEMP.mkdir(parents=True, exist_ok=True)
    if args.etapa == "preparar":
        with ProcessPoolExecutor(max_workers=3) as pool:
            list(pool.map(preparar, VOOS))
    elif args.etapa == "inferir":
        inferir()
    else:
        avaliar()
