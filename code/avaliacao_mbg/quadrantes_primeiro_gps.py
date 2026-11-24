"""Testa a origem da malha no primeiro GPS dos seis voos, com intervalo de mesma cota."""
import json

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from adaptativa import METRICA, MOTOR, VOOS, gravar_npz, verificar_pesos
from alertas_cobertura import avaliar_voo, estatisticas
from avaliar import ratios
from dados import CACHE, GSD, folder
from ensaio_altitude_integral import escrever_csv, escrever_json
from inferir import infer_video
from memoria_sim import load_pose
from motores import runner, setup
from passagens import carregar_voo
from quadrantes_voos import DESTINO as BASE, METODOS, ler_quadros, plano
from seguimento import temporal

TEMP = CACHE / "quadrantes_primeiro_gps_v1"
DESTINO = BASE / "primeiro_gps"
CHAVE = ["video", "classe", "trilha"]


def preparar(video, indice):
    # Predições e referências no mesmo sistema de avaliação do ensaio a 40 m.
    pose_mapa, tamanho, tiles, fps = carregar_voo(video)
    log, = (folder(video) / "log").glob("*.csv")
    n = len(pose_mapa)
    pose_malha = np.column_stack(load_pose(log, n, fps, origem_primeiro_gps=True))
    np.testing.assert_array_equal(pose_malha[0, :2], [0., 0.])
    np.testing.assert_array_equal(pose_malha[:, 2], pose_mapa[:, 2])
    geo, passagens, adiados = plano(pose_malha, tamanho, tiles)
    controle = temporal(geo.shape, int(geo.sum()), 0, indice)
    assert geo.sum() == controle.sum()
    for q in range(0, n, 60):
        assert geo[q:q + 60].sum() <= len(tiles) * len(range(q, min(q + 60, n), 30))
    masks = np.stack([geo, controle], axis=-1)
    pedidos = masks.any(axis=2)
    frames = np.flatnonzero(pedidos.any(axis=1))
    dados = dict(frames=frames, total_frames=n, fps=fps, starts=np.array([0]),
                 tiles=tiles, pose=pose_mapa[frames], gsd=GSD, size=tamanho,
                 masks=masks[frames], requested=pedidos[frames])
    caminho = TEMP / f"{video}.npz"
    if caminho.exists():
        with np.load(caminho) as anterior:
            for chave, valor in dados.items():
                np.testing.assert_array_equal(anterior[chave], valor)
    else:
        gravar_npz(caminho, **dados)
    print(f"{video}: {geo.sum()} recortes por método, {passagens} passagens, "
          f"{adiados} pedidos adiados", flush=True)


def juntar_origens(nome, novos):
    anterior = pd.read_csv(BASE / f"{nome}.csv")
    anterior = anterior[anterior.metodo.isin(METODOS)]
    return pd.concat([anterior.assign(origem="media_do_voo"),
                      novos.assign(origem="primeiro_gps")], ignore_index=True)


def consolidar(voos, trilhas):
    voos = pd.DataFrame(voos)
    trilhas = pd.DataFrame(trilhas)
    medidas = [dict(video=v, metodo=m, **estatisticas(g))
               for (v, m), g in trilhas.groupby(["video", "metodo"])]
    voos = voos.merge(pd.DataFrame(medidas), on=["video", "metodo"], validate="one_to_one")
    por_voo = juntar_origens("por_voo", voos)
    por_trilha = juntar_origens("por_trilha", trilhas)
    assert len(por_voo) == 4 * len(VOOS)
    assert not por_voo.duplicated(["video", "metodo", "origem"]).any()
    assert por_voo.groupby(["origem", "video"]).recortes.nunique().eq(1).all()
    assert not por_trilha.duplicated([*CHAVE, "origem", "metodo"]).any()
    assert por_trilha.groupby(CHAVE).size().eq(4).all()
    assert por_trilha.groupby(CHAVE).quadro_entrada.nunique().eq(1).all()
    resumo, comuns = [], []
    for (origem, metodo), g in por_voo.groupby(["origem", "metodo"]):
        t = por_trilha[(por_trilha.origem == origem) & (por_trilha.metodo == metodo)]
        assert len(t) == g.referencias.sum() == 1924 and t.acerto_foco.sum() == g.vp.sum()
        resumo.append(dict(origem=origem, metodo=metodo, voos=len(g), recortes=g.recortes.sum(),
                           recortes_s=g.recortes.sum() / g.duracao_s.sum(), **ratios(g[["vp", "fp", "fn"]].sum()),
                           **estatisticas(t)))
    for origem, t in por_trilha.groupby("origem"):
        pares = t.pivot(index=CHAVE, columns="metodo", values="atraso_s").dropna()
        geo, inter = pares[METODOS[0]], pares[METODOS[1]]
        comuns.append(dict(origem=origem, confirmados_por_ambos=len(pares),
                           quadrantes_primeiro=(geo < inter).sum(), mesmo_quadro=(geo == inter).sum(),
                           intervalo_primeiro=(inter < geo).sum(), mediana_quadrantes_s=geo.median(),
                           mediana_intervalo_s=inter.median(), mediana_diferenca_s=(geo - inter).median()))
    resumo = pd.DataFrame(resumo)
    for nome, df in [("resumo", resumo), ("por_voo", por_voo), ("por_trilha", por_trilha),
                     ("tempos_pareados", pd.DataFrame(comuns))]:
        escrever_csv(df, DESTINO / f"{nome}.csv")
    print(resumo.to_string(index=False), flush=True)


def executar():
    verificar_pesos()
    setup()
    motor = runner(MOTOR)
    voos, trilhas = [], []
    for indice, video in enumerate(VOOS):
        preparar(video, indice)
        infer_video(video, MOTOR, motor, cache=TEMP, sparse=True, leitura_continua=True, leitor=ler_quadros)
        medidas, tracks = avaliar_voo(video, cache=TEMP, metodos=METODOS)
        voos.extend(medidas)
        trilhas.extend(tracks)
        print(video, [(m["metodo"], m["vp"], m["fp"]) for m in medidas], flush=True)
    DESTINO.mkdir(exist_ok=True)
    consolidar(voos, trilhas)
    protocolo = json.loads((BASE / "protocolo.json").read_text(encoding="utf-8"))
    protocolo.update(origem_malha="Primeiro registro GPS válido na série usada para a pose. "
                     "O referencial local WGS84 usa a latitude e longitude desse registro. "
                     "Mantidas a guinada e a interpolação temporal do ensaio original.",
                     referencial_avaliacao="O plano de enquadramento usa o primeiro GPS. A projeção das "
                     "detecções e das referências usa o mesmo referencial do ensaio original.",
                     identificador_origem_anterior="media_do_voo",
                     identificador_origem_atual="primeiro_gps")
    assert protocolo["pesos_sha256"] == METRICA["pesos_sha256"]
    escrever_json(DESTINO / "protocolo.json", protocolo)


if __name__ == "__main__":
    TEMP.mkdir(parents=True, exist_ok=True)
    with FileLock(str(TEMP / "avaliacao.lock")), torch.inference_mode():
        executar()
