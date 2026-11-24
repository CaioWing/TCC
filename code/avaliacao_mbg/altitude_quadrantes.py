"""Perda de detalhe com os planos pareados de quadrantes dos seis voos completos."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from adaptativa import METRICA, MOTOR, VOOS, gravar_npz, verificar_pesos
from alertas_cobertura import avaliar_voo, estatisticas
from avaliar import ratios
from dados import CACHE, OUT, folder
from ensaio_altitude_integral import (ALTURAS, assinatura, escrever_csv, escrever_json,
                                     inferencia_recorte)
from motores import runner, setup
from quadrantes_voos import DESTINO, METODOS, TEMP, ler_quadros

SAIDA = DESTINO / "altitude"
TRABALHO = CACHE / "altitude_quadrantes"


def inferir_escalas(video, motor, identificador):
    arquivo = TEMP / f"{video}.npz"
    fonte, = (folder(video) / "avi").glob("*.avi")
    stat = fonte.stat()
    chave = hashlib.sha256(
        f"{identificador}/{assinatura(arquivo)}/{stat.st_size}/{stat.st_mtime_ns}".encode()
    ).hexdigest()
    pastas = {h: TRABALHO / chave / str(h) for h in ALTURAS}
    faltam = [h for h, pasta in pastas.items() if not (pasta / MOTOR / arquivo.name).exists()]
    if not faltam:
        return pastas
    with np.load(arquivo) as npz:
        plano = dict(npz)
    frames, tiles, pedidos = (plano[k] for k in ("frames", "tiles", "requested"))
    np.testing.assert_array_equal(pedidos, plano["masks"].any(axis=2))
    caixas = {h: [] for h in faltam}
    contagens = {h: np.zeros(pedidos.shape, dtype=np.int32) for h in faltam}
    for i, (q, imagem) in enumerate(zip(frames, ler_quadros(fonte, frames, True), strict=True)):
        assert tuple(imagem.shape[1::-1]) == tuple(plano["size"])
        for j in np.flatnonzero(pedidos[i]):
            x, y = tiles[j]
            for h in faltam:
                det = inferencia_recorte(imagem, x, y, h, motor)
                contagens[h][i, j] = len(det)
                caixas[h].append(det)
        if (i + 1) % 500 == 0:
            print(video, f"{i + 1}/{len(frames)} quadros selecionados", flush=True)
    for h in faltam:
        destino = pastas[h]
        gravar_npz(destino / arquivo.name, **plano)
        offsets = np.r_[0, np.cumsum(contagens[h])]
        det = np.concatenate(caixas[h])
        assert len(caixas[h]) == pedidos.sum() and offsets[-1] == len(det)
        gravar_npz(destino / MOTOR / arquivo.name, frames=frames, computed=pedidos,
                   offsets=offsets, det=det)
    return pastas


def originais(nome):
    integral = pd.read_csv(OUT / "altitude_integral" / nome).query("metodo == 'integral' and altura_m == 40")
    selecao = pd.read_csv(DESTINO / nome)
    selecao = selecao[selecao.metodo.isin(METODOS)].assign(altura_m=40)
    return pd.concat([selecao[integral.columns], integral], ignore_index=True)


def executar():
    verificar_pesos()
    setup()
    motor = runner(MOTOR)
    engine = assinatura(CACHE / f"{MOTOR}.opt1.engine")
    fonte_escala = assinatura(Path(__file__).with_name("ensaio_altitude_integral.py"))
    versao = f"{engine}/{fonte_escala}/{assinatura(Path(__file__))}"
    voos = [originais("por_voo.csv")]
    trilhas = [originais("por_trilha.csv")]
    for video in VOOS:
        print("Escalas", video, flush=True)
        pastas = inferir_escalas(video, motor, versao)
        for h, pasta in pastas.items():
            v, t = avaliar_voo(video, cache=pasta, metodos=METODOS)
            t = pd.DataFrame(t).assign(altura_m=h)
            v = pd.DataFrame(v).assign(altura_m=h)
            medidas = [dict(metodo=m, **estatisticas(g)) for m, g in t.groupby("metodo")]
            v = v.merge(pd.DataFrame(medidas), on="metodo", validate="one_to_one")
            voos.append(v[voos[0].columns])
            trilhas.append(t[trilhas[0].columns])
        print("Avaliação concluída", video, flush=True)
    voos, trilhas = pd.concat(voos, ignore_index=True), pd.concat(trilhas, ignore_index=True)
    chave = ["video", "metodo", "altura_m"]
    assert not voos.duplicated(chave).any()
    assert not trilhas.duplicated([*chave, "classe", "trilha"]).any()
    cotas = voos[voos.metodo.isin(METODOS)].groupby("video").recortes
    assert len(cotas) == len(VOOS) and cotas.nunique().eq(1).all()
    resumo = []
    for (m, h), g in voos.groupby(["metodo", "altura_m"]):
        t = trilhas[(trilhas.metodo == m) & (trilhas.altura_m == h)]
        assert len(g) == len(VOOS) and len(t) == g.referencias.sum() == 1924
        assert t.acerto_foco.sum() == g.vp.sum()
        resumo.append(dict(metodo=m, altura_m=h, voos=len(g), recortes=g.recortes.sum(),
                           **ratios(g[["vp", "fp", "fn"]].sum()), **estatisticas(t)))
    resumo = pd.DataFrame(resumo)
    assert len(resumo) == len(METODOS) * (1 + len(ALTURAS)) + 1
    SAIDA.mkdir(exist_ok=True)
    for nome, tabela in [("por_voo", voos), ("por_trilha", trilhas), ("resumo_seis_voos", resumo)]:
        escrever_csv(tabela, SAIDA / f"{nome}.csv")
    escrever_json(SAIDA / "protocolo.json", dict(
        voos=VOOS, metodos=METODOS, alturas_simuladas_m=ALTURAS, integral_m=40,
        motor=MOTOR, pesos_sha256=METRICA["pesos_sha256"], engine_sha256=engine,
        planos_sha256={v: assinatura(TEMP / f"{v}.npz") for v in VOOS},
        escala="Mesmos instantes e posições dos recortes do plano a 40 m. As quatro escalas usam "
               "a transformação de ensaio_altitude_integral.inferencia_recorte. "
               "A avaliação por foco e primeiro alerta usa alertas_cobertura.avaliar_voo."))
    print(resumo.to_string(index=False), flush=True)


if __name__ == "__main__":
    TRABALHO.mkdir(parents=True, exist_ok=True)
    with FileLock(str(TRABALHO / "execucao.lock"), timeout=0), torch.inference_mode():
        executar()
