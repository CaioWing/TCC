"""Avalia a primeira vista de cada quadrante por passagem nos seis voos originais."""
import json

import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from adaptativa import METRICA, MOTOR, VOOS, gravar_npz, verificar_pesos
from alertas_cobertura import avaliar_voo, estatisticas
from avaliar import ratios
from cobertura_continua import DESTINO as BASE
from dados import CACHE, GSD, OUT
from inferir import infer_video, quadros_opencv
from memoria_sim import T
from motores import runner, setup
from passagens import LADO, carregar_voo, melhor_recorte
from seguimento import temporal

TEMP = CACHE / "quadrantes_v1"
DESTINO = OUT / "quadrantes"
METODOS = ("quadrantes", "intervalo_pareado")


def ler_quadros(fonte, frames, leitura_continua):
    """Em MJPEG cada pacote AVI é uma imagem independente; só decodifica as selecionadas."""
    try:
        import av
    except ImportError:
        yield from quadros_opencv(fonte, frames, leitura_continua)
        return
    with av.open(str(fonte)) as video:
        stream = video.streams.video[0]
        if stream.codec_context.name == "mjpeg":
            pedidos = set(map(int, frames))
            for i, pacote in enumerate(video.demux(stream)):
                if i not in pedidos:
                    continue
                imagem, = pacote.decode()
                assert imagem.pts * stream.time_base * stream.average_rate == i
                yield imagem.to_ndarray(format="bgr24")
                if i == frames[-1]:
                    return
            return
    yield from quadros_opencv(fonte, frames, leitura_continua)


def plano(pose, tamanho, tiles):
    """Usa só a pose atual e a memória dos quadrantes observados na passagem em curso."""
    w, h = tamanho
    raio = int(np.ceil(np.hypot(w, h) * GSD / (2 * LADO))) + 1
    di, dj = np.mgrid[-raio:raio + 1, -raio:raio + 1]
    mascara = np.zeros((len(pose), len(tiles)), dtype=bool)
    memoria = {}
    passagens = pedidos_adiados = 0
    for q, (leste, norte, yaw) in enumerate(pose):
        if q % 60 == 0:
            saldo = len(tiles) * len(range(q, min(q + 60, len(pose)), 30))
        i = di + int(np.floor(leste / LADO))
        j = dj + int(np.floor(norte / LADO))
        de, dn = (i * LADO - leste) / GSD, (j * LADO - norte) / GSD
        c, s = np.cos(yaw), np.sin(yaw)
        u, v = w / 2 + c * de - s * dn, h / 2 - s * de - c * dn
        dentro = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        chaves = list(zip(i[dentro], j[dentro]))
        u, v = u[dentro], v[dentro]
        bits = 1 << ((u >= w / 2).astype(int) + 2 * (v >= h / 2).astype(int))
        passagens += sum(chave not in memoria for chave in chaves)
        # A saída encerra a passagem; uma reentrada começa com as quatro vistas disponíveis.
        memoria = {chave: memoria.get(chave, 0) for chave in chaves}
        pendentes = [k for k, chave in enumerate(chaves) if not (memoria[chave] & bits[k])]
        if not pendentes:
            continue
        pedidos = [melhor_recorte(u[k], v[k], tiles) for k in pendentes]
        candidatos, contagens = np.unique(pedidos, return_counts=True)
        ordem = np.argsort(-contagens, kind="stable")
        enviados = candidatos[ordem[:saldo]]
        mascara[q, enviados] = True
        saldo -= len(enviados)
        observados = np.zeros(len(chaves), dtype=bool)
        for x, y in tiles[enviados]:
            observados |= (x <= u) & (u < x + T) & (y <= v) & (v < y + T)
        for k in np.flatnonzero(observados):
            memoria[chaves[k]] |= int(bits[k])
        pedidos_adiados += np.count_nonzero(~observados[pendentes])
    return mascara, passagens, pedidos_adiados


def preparar(video, numero_voo, eventos):
    pose, tamanho, tiles, fps = carregar_voo(video)
    geo, passagens, adiados = plano(pose, tamanho, tiles)
    # Controle independente da projeção: número de passagens e pedidos no primeiro quadro.
    assert passagens == len(eventos)
    iniciais = eventos[eventos.quadro_entrada == 0]
    np.testing.assert_array_equal(np.flatnonzero(geo[0]), np.unique(iniciais.recorte_entrada))
    for q in range(0, len(pose), 60):
        limite = 24 * len(range(q, min(q + 60, len(pose)), 30))
        assert geo[q:q + 60].sum() <= limite
    controle = temporal(geo.shape, int(geo.sum()), 0, numero_voo)
    masks = np.stack((geo, controle), axis=-1)
    requested = masks.any(axis=2)
    frames = np.flatnonzero(requested.any(axis=1))
    dados = dict(frames=frames, total_frames=len(pose), fps=fps, starts=np.array([0]),
                 tiles=tiles, pose=pose[frames], gsd=GSD, size=tamanho,
                 masks=masks[frames], requested=requested[frames])
    caminho = TEMP / f"{video}.npz"
    if caminho.exists():
        with np.load(caminho) as anterior:
            for chave, valor in dados.items():
                np.testing.assert_array_equal(anterior[chave], valor)
    else:
        gravar_npz(caminho, **dados)
    return dict(video=video, passagens=passagens, pedidos_ponto_quadro_adiados=int(adiados),
                recortes=int(geo.sum()), quadros_enviados=int(geo.any(axis=1).sum()),
                pico_recortes_no_quadro=int(geo.sum(axis=1).max()))


def escrever_csv(df, nome):
    caminho = DESTINO / nome
    tmp = caminho.with_suffix(".partial.csv")
    df.to_csv(tmp, index=False, float_format="%.8f")
    tmp.replace(caminho)


def executar():
    verificar_pesos()
    setup()
    eventos = pd.read_csv(OUT / "passagens/eventos.csv")
    linhas, trilhas, demandas = [], [], []
    with torch.inference_mode():
        run = runner(MOTOR)
        for numero_voo, video in enumerate(VOOS):
            demanda = preparar(video, numero_voo, eventos[eventos.video == video])
            print("Plano", demanda, flush=True)
            infer_video(video, MOTOR, run, cache=TEMP, sparse=True, leitura_continua=True, leitor=ler_quadros)
            voos, tr = avaliar_voo(video, cache=TEMP, metodos=METODOS)
            linhas.extend(voos)
            trilhas.extend(tr)
            demandas.append(demanda)
            print("Avaliação", video, [(v["metodo"], v["vp"], v["fp"]) for v in voos], flush=True)

    por_trilha = pd.DataFrame(trilhas)
    por_voo = pd.DataFrame(linhas)
    medidas = [dict(video=v, metodo=m, **estatisticas(g)) for (v, m), g in
               por_trilha.groupby(["video", "metodo"])]
    por_voo = por_voo.merge(pd.DataFrame(medidas), on=["video", "metodo"], validate="one_to_one")
    assert por_voo.groupby("video").recortes.nunique().eq(1).all()
    por_voo = pd.concat([pd.read_csv(BASE / "por_voo.csv"), por_voo], ignore_index=True)
    por_trilha = pd.concat([pd.read_csv(BASE / "por_trilha.csv"), por_trilha], ignore_index=True)
    chaves = ["video", "classe", "trilha"]
    assert not por_trilha.duplicated(chaves + ["metodo"]).any()
    assert por_trilha.groupby(chaves).metodo.nunique().eq(4).all()
    resumo = []
    for m, g in por_voo.groupby("metodo", sort=False):
        tr = por_trilha[por_trilha.metodo == m]
        assert len(tr) == g.referencias.sum() == 1924 and tr.acerto_foco.sum() == g.vp.sum()
        resumo.append(dict(metodo=m, voos=len(g), recortes=g.recortes.sum(),
                           recortes_s=g.recortes.sum() / g.duracao_s.sum(),
                           **ratios(g[["vp","fp","fn"]].sum()), **estatisticas(tr)))
    DESTINO.mkdir(exist_ok=True)
    for dados, nome in [(pd.DataFrame(demandas), "demanda.csv"), (por_voo, "por_voo.csv"),
                        (por_trilha, "por_trilha.csv"), (pd.DataFrame(resumo), "resumo.csv")]:
        escrever_csv(dados, nome)
    protocolo = dict(voos=VOOS, altura_original_m=40, lado_m=LADO, motor=MOTOR,
                     pesos_sha256=METRICA["pesos_sha256"], limiares={k: METRICA[k] for k in
                     ("conf", "raio_m", "dmax_m", "iou_nms")},
                     regra="Em cada quadro, projetar pontos fixos de 8,64 m. Pedir um recorte na primeira "
                     "vista de cada um dos quatro quadrantes 2×2 durante a passagem contínua. Na saída "
                     "a memória do ponto expira. Uma reentrada abre nova passagem. Sem vista de saída obrigatória.",
                     selecao="Usar o recorte que contém o ponto e tem o centro mais próximo. Fundir pedidos "
                     "iguais. Todos os pontos contidos em um recorte enviado recebem crédito no quadrante atual.",
                     teto="Até 48 recortes por bloco de 60 quadros, limitado à cota do intervalo no bloco final. "
                     "Selecionar a qualquer quadro, sem completar a cota por obrigação. Em falta de saldo, "
                     "priorizar mais pedidos e depois o menor índice de recorte. Pedidos sem vista podem "
                     "ser atendidos em outro quadro enquanto o ponto estiver visível no quadrante.",
                     controle="Mesmo número de recortes por voo. Quadros uniformemente espaçados desde o "
                     "quadro zero, com um quadro parcial quando necessário, pela função temporal(). "
                     "O controle pareado usa o total conhecido após cada voo.",
                     avaliacao="Mesma avaliação dos focos e primeiro alerta usada nos voos completos. "
                     "Mediana e p90 condicionais à confirmação na imagem, em tempo de vídeo.")
    caminho = DESTINO / "protocolo.json"
    tmp = caminho.with_suffix(".partial.json")
    tmp.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(caminho)


if __name__ == "__main__":
    TEMP.mkdir(parents=True, exist_ok=True)
    with FileLock(str(TEMP / "avaliacao.lock")):
        executar()
