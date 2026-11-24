"""Calibra em F/I quando dispensar um quadro da cadência original de 30."""
import numpy as np
import pandas as pd

from adaptativa import MOTOR, gravar_npz, verificar_pesos
from avaliar import ratios
from dados import CACHE, SPLITS
from fronteiras import selected_tiles_fronteiras
from metricas import load_clips
from movimento_camera import deslocamentos
from passagens import CONJUNTOS, DESTINO, carregar_voo, pontuar_mascara, salvar


VALIDACAO = CACHE / "camera_validacao"
# Frações de um passo entre recortes; zero reproduz a referência.
LIMIARES = np.arange(9) / 8


def preparar_validacao():
    import torch
    from inferir import infer_video
    from motores import runner, setup

    VALIDACAO.mkdir(exist_ok=True)
    setup()
    with torch.inference_mode():
        modelo = runner(MOTOR)
        for video in SPLITS["voos"]["val"]:
            with np.load(CACHE / f"{video}.npz") as dados:
                informacao = dict(dados)
            agendado = (informacao["frames"] % 30) == 0
            pedidos = np.broadcast_to(agendado[:, None], (len(agendado), len(informacao["tiles"])))
            gravar_npz(VALIDACAO / f"{video}.npz", **informacao, requested=pedidos)
            infer_video(video, MOTOR, modelo, cache=VALIDACAO, sparse=True)


def calendario(video, limiares):
    pose, tamanho, tiles, _ = carregar_voo(video)
    with np.load(CACHE / f"{video}.npz") as dados:
        gsd = float(dados["gsd"])
    grade = [np.diff(np.unique(tiles[:, i])).mean() for i in [0, 1]]
    indices = np.arange(0, len(pose), 30)
    # Ao omitir uma oportunidade, a seguinte é obrigatória (intervalo máximo de 60).
    distancias = abs(deslocamentos(pose[indices], tamanho, gsd)).mean(axis=1)
    movimento = (distancias / grade).sum(axis=1)
    for limiar in limiares:
        mantidos = np.ones(len(indices), dtype=bool)
        for j, d in enumerate(movimento, 1):
            mantidos[j] = (not mantidos[j-1]) or d >= limiar
        yield limiar, indices[mantidos]


def avaliar(voos, conjuntos, limiares):
    linhas = []
    for video in voos:
        planos = dict(calendario(video, limiares))
        for conjunto, cache in conjuntos.items():
            for clip in load_clips(video, MOTOR, cache=cache):
                memo = {}
                for limiar, mantidos in planos.items():
                    inclusos = np.isin(clip["frames"], mantidos)
                    mask = np.broadcast_to(inclusos[:, None], (len(inclusos), len(clip["tiles"][0])))
                    cont, ids = pontuar_mascara(clip, mask, selected_tiles_fronteiras, memo)
                    if limiar == 0:
                        referencia = ids
                    linhas.append(dict(video=video, conjunto=conjunto, trecho=clip["clip"], limiar=limiar,
                                       quadros=inclusos.sum(), recortes=mask.sum(),
                                       perdas=len(referencia-ids), ganhos=len(ids-referencia),
                                       **ratios(cont.sum(axis=0))))
    return pd.DataFrame(linhas)


def totais(df):
    linhas = []
    for limiar, g in df.groupby("limiar"):
        somas = g[["quadros", "recortes", "perdas", "ganhos"]].sum().to_dict()
        linhas.append(dict(limiar=limiar, **somas, **ratios(g[["vp","fp","fn"]].sum())))
    return pd.DataFrame(linhas)


def main():
    verificar_pesos()
    preparar_validacao()
    (DESTINO / "camera").mkdir(exist_ok=True)
    val = avaliar(SPLITS["voos"]["val"], {"validacao": VALIDACAO}, LIMIARES)
    resultado = totais(val)
    escolha = resultado[resultado.perdas == 0].sort_values(["quadros", "limiar"]).iloc[0].limiar
    teste = avaliar(SPLITS["voos"]["test"], CONJUNTOS, sorted({0., escolha}))
    base = pd.read_csv(DESTINO / "camera/por_trecho.csv").query("modo=='temporal' and b==4 and fase==0")
    chaves = ["video", "conjunto", "trecho"]
    campos = ["vp","fp","fn","recortes"]
    pd.testing.assert_frame_equal(teste[teste.limiar==0].set_index(chaves)[campos].sort_index(),
                                  base.set_index(chaves)[campos].sort_index())
    salvar(val, "camera/filtro_validacao_por_trecho.csv")
    salvar(resultado, "camera/filtro_validacao.csv")
    salvar(teste, "camera/filtro_teste_por_trecho.csv")
    salvar(totais(teste), "camera/filtro_teste.csv")
    print("Limiar da validação:", escolha)
    print(totais(teste).to_string(index=False))


if __name__ == "__main__":
    main()
