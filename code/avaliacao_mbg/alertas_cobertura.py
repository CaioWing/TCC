"""Métricas por foco no voo inteiro e primeiro alerta confirmado na imagem."""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from adaptativa import METRICA, MOTOR, VOOS
from avaliar import ratios
from cobertura_continua import DESTINO, METODOS, TEMP
from dados import folder
from memoria_sim import load_pose
from metricas import (CLASSES, focus_counts, frame_metrics, ground, labels, load_clips,
                      projected, selected_tiles)


def referencias(video, n, fps, size, gsd):
    todas = labels(video, {"frames": np.arange(n)})
    log, = (folder(video) / "log").glob("*.csv")
    poses = np.column_stack(load_pose(log, n, fps))
    inicio, pontos = {}, defaultdict(list)
    for frame, (caixas, pose) in enumerate(zip(todas, poses)):
        for caixa, xy in zip(caixas, ground(caixas, pose, size, gsd)):
            chave = int(caixa[5]), int(caixa[4])
            inicio.setdefault(chave, frame)
            pontos[chave].append(xy)
    return inicio, {k: np.median(v, axis=0) for k, v in pontos.items()}


def primeiros_alertas(clip, previsoes):
    """Pareamento por classe e IoU de 0,5 no próprio quadro, como em frame_metrics."""
    primeiros = {}
    conf = METRICA["conf"]
    for q, gt, pred in zip(clip["frames"], clip["gt"], previsoes):
        candidatas = pred[pred[:, 4] >= conf]
        if not len(candidatas) or not len(gt):
            continue
        scores = {}
        frame_metrics({"gt": [gt]}, [candidatas], conf, track_scores=scores)
        for chave in scores:
            primeiros.setdefault(chave, int(q))
    return primeiros


def avaliar_voo(video, *, cache=TEMP, metodos=METODOS):
    with np.load(cache / f"{video}.npz") as info:
        n, fps = int(info["total_frames"]), float(info["fps"])
        inicio, centros = referencias(video, n, fps, info["size"], info["gsd"])
    clip, = load_clips(video, MOTOR, cache=cache)
    clip["centers"] = centros  # Inclui as trilhas ausentes de todos os quadros enviados.
    voos, trilhas = [], []
    for m, metodo in enumerate(metodos):
        mascara = clip["masks"][:, :, m]
        previsoes = selected_tiles(clip, mascara)
        contagens, acertos = focus_counts(clip, projected(clip, previsoes), METRICA["conf"],
                                          METRICA["raio_m"], METRICA["dmax_m"], return_matches=True)
        primeiros = primeiros_alertas(clip, previsoes)
        assert set(primeiros) <= set(inicio)
        for (classe, id_), entrada in inicio.items():
            q = primeiros.get((classe, id_))
            assert q is None or q >= entrada
            trilhas.append(dict(video=video, metodo=metodo, classe=CLASSES[classe], trilha=id_,
                                quadro_entrada=entrada, acerto_foco=int((classe, id_) in acertos),
                                primeiro_alerta_quadro=q, atraso_s=(q-entrada)/fps if q is not None else None))
        voos.append(dict(video=video, metodo=metodo, quadros_voo=n, fps=fps, duracao_s=n/fps,
                         recortes=int(mascara.sum()), quadros_enviados=int(mascara.any(axis=1).sum()),
                         primeiro_alerta_voo_s=min(primeiros.values())/fps if primeiros else None,
                         **ratios(contagens.sum(axis=0))))
    return voos, trilhas


def estatisticas(grupo):
    atraso = grupo.atraso_s
    return dict(confirmados_imagem=int(atraso.count()), sem_confirmacao=int(atraso.isna().sum()),
                confirmados_ate_1s=int((atraso <= 1).sum()), mediana_s=atraso.median(),
                p90_s=atraso.quantile(.9))


def gravar(df, nome):
    caminho = DESTINO / nome
    tmp = caminho.with_suffix(".partial.csv")
    df.to_csv(tmp,index=False,float_format="%.8f")
    tmp.replace(caminho)


def curva_de_alertas(trilhas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    for metodo, nome, cor in [("cobertura", "Memória por cobertura", "#007a76"),
                              ("intervalo", "Um quadro a cada 30", "#a44819")]:
        grupo = trilhas[trilhas.metodo == metodo]
        tempos = np.sort(grupo.atraso_s.dropna())
        x = np.r_[0., tempos]
        y = 100 * np.arange(len(x)) / len(grupo)
        ax.step(x, y, where="post", label=nome, color=cor, lw=2)
        ax.scatter(x[-1], y[-1], color=cor, s=18)
    ax.set_xscale("symlog", linthresh=1., base=2)
    maximo = trilhas.atraso_s.max()
    ticks = [t for t in [0, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if t <= maximo]
    ax.set_xticks(ticks, [str(t).replace(".", ",") for t in ticks])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set(xlim=(0, maximo*1.05), ylim=(0, 100),
           xlabel="Segundos desde a primeira aparição anotada",
           ylabel="Trilhas com alerta confirmado",
           title="Quando chega a primeira detecção correta?")
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(DESTINO / "tempo_ate_alerta.png", dpi=180)
    plt.close(fig)


def avaliar():
    voos, trilhas = [], []
    for video in VOOS:
        v, t = avaliar_voo(video)
        voos.extend(v)
        trilhas.extend(t)
    voos, trilhas = pd.DataFrame(voos), pd.DataFrame(trilhas)
    colunas = ["video","classe","trilha"]
    pares = trilhas.pivot(index=colunas, columns="metodo", values="atraso_s").dropna()
    assert not trilhas.duplicated([*colunas,"metodo"]).any()
    assert voos.groupby("video")[["recortes","referencias"]].nunique().eq(1).all().all()
    estat = [dict(video=v,metodo=m,**estatisticas(g)) for (v,m),g in trilhas.groupby(["video","metodo"])]
    voos = voos.merge(pd.DataFrame(estat), on=["video","metodo"], validate="one_to_one")
    assert (voos.confirmados_imagem + voos.sem_confirmacao == voos.referencias).all()
    resumo = []
    for m, g in voos.groupby("metodo"):
        tr = trilhas[trilhas.metodo == m]
        assert g.vp.sum() == tr.acerto_foco.sum()
        resumo.append(dict(metodo=m, **ratios(g[["vp","fp","fn"]].sum()), recortes=g.recortes.sum(),
                           quadros_enviados=g.quadros_enviados.sum(), duracao_s=g.duracao_s.sum(),
                           **estatisticas(tr), confirmados_comuns=len(pares), mediana_comuns_s=pares[m].median()))
    DESTINO.mkdir(parents=True, exist_ok=True)
    gravar(voos, "por_voo.csv")
    gravar(trilhas, "por_trilha.csv")
    gravar(pd.DataFrame(resumo), "resumo.csv")
    curva_de_alertas(trilhas)
    config = dict(motor=MOTOR, pesos_sha256=METRICA["pesos_sha256"], voos=VOOS, fase=0,
                  referencias="Todas as trilhas das seis classes com caixa presente em algum quadro do vídeo.",
                  intervalo="24 recortes em cada quadro múltiplo de 30, desde zero.",
                  cobertura="Malha de 0,5 m, prioridade à área ainda sem vista. Sem reinício durante o voo. "
                            "Uma seleção a cada 20 quadros. Em blocos completos de 60, 16 recortes por seleção. "
                            "No bloco final distribui-se a cota real do intervalo pelos instantes disponíveis. "
                            "Empates seguem a rotação original em passo de sete, iniciada em zero.",
                  foco="Mesma NMS, projeção, associação e métricas do ensaio por cobertura. "
                       "Mediana das posições anotadas em todo o voo para cada trilha, uma associação por foco.",
                  confianca=METRICA["conf"], raio_associacao_m=METRICA["raio_m"],
                  distancia_referencia_m=METRICA["dmax_m"],
                  atraso="Tempo de vídeo da primeira caixa presente até a primeira previsão no mesmo quadro "
                         "com classe correta, confiança >=0,5 e IoU >=0,5, com pareamento único por quadro. "
                         "Mediana e p90 condicionais à confirmação. Ausência de alerta = campo vazio. "
                         "Primeiro alerta do voo contado desde o quadro zero. Sem latência computacional.")
    (DESTINO/"protocolo.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(pd.DataFrame(resumo).to_string(index=False))
