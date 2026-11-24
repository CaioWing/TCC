"""Ensaio de escala com as seleções originais, seguido da referência integral em 4K.

Executar com o mesmo ambiente TensorRT de cobertura_continua.py. Uma nova execução
retoma os blocos validados da mesma versão do protocolo e das mesmas entradas.
"""
import ctypes
import hashlib
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from filelock import FileLock

from adaptativa import METRICA, MOTOR, VOOS, gravar_npz, verificar_pesos
from alertas_cobertura import estatisticas, primeiros_alertas, referencias
from avaliar import ratios
from cobertura_continua import DESTINO as BASE, METODOS, TEMP as CACHE_BASE
from dados import CACHE, DATA, OUT, folder
from memoria_sim import T, load_pose
from metricas import CLASSES, focus_counts, indices_nms_quadro, labels, projected
from motores import blob, postprocess, runner, setup


ALTURAS = (50, 60, 70, 80)
BLOCO = 300
DESTINO = OUT / "altitude_integral"
TRABALHO = CACHE / "altitude_integral_v1"
ETAPAS = [("controle", 40), *(("escala", h) for h in ALTURAS), ("integral", 40)]
VAZIO = np.empty((0, 6), dtype=np.float32)


def escrever_json(caminho, dados):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(".partial.json")
    tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(caminho)


def escrever_csv(df, caminho):
    tmp = caminho.with_suffix(".partial.csv")
    df.to_csv(tmp, index=False, float_format="%.8f")
    tmp.replace(caminho)


def hora():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def assinatura(arquivo):
    with arquivo.open("rb") as entrada:
        return hashlib.file_digest(entrada, "sha256").hexdigest()


def protocolo():
    itens = {}
    for video in VOOS:
        fonte, = (folder(video) / "avi").glob("*.avi")
        log, = (folder(video) / "log").glob("*.csv")
        itens[video] = dict(video_bytes=fonte.stat().st_size, video_mtime_ns=fonte.stat().st_mtime_ns,
                            plano=assinatura(CACHE_BASE / f"{video}.npz"),
                            previsoes=assinatura(CACHE_BASE / MOTOR / f"{video}.npz"),
                            anotacao=assinatura(DATA / "ann" / f"{video}.xml"), telemetria=assinatura(log))
    return dict(versao=1, motor=MOTOR, engine_sha256=assinatura(CACHE / f"{MOTOR}.opt1.engine"),
                pesos_sha256=METRICA["pesos_sha256"], limiares={k: METRICA[k] for k in ("conf","raio_m","dmax_m","iou_nms")},
                alturas_simuladas_m=ALTURAS, referencia_m=40, voos=VOOS, quadros_por_bloco=BLOCO,
                escala="Mesmos quadros, recortes selecionados, cotas e 1.924 trilhas da referência. "
                       "Para cada recorte, janela concêntrica de lado 640*h/40 no vídeo 4K, reduzida a 640 com INTER_AREA. "
                       "Usa contexto real adjacente e preenchimento 114 só além do quadro. "
                       "Reprojeta as caixas para os pixels de origem, aceita centros no recorte originalmente selecionado "
                       "e limita a caixa aos seus contornos antes da NMS do quadro. "
                       "Isola a perda por escala, sem simular terreno novo, alterações de trajetória ou novos sensores.",
                integral="Somente 40 m originais. Os 24 recortes de todos os quadros dos seis voos. "
                         "Reutiliza previsões originais já calculadas com o mesmo motor e infere os pares restantes.",
                codigo=assinatura(Path(__file__)), entradas=itens)


def carregar_plano(video):
    with np.load(CACHE_BASE / f"{video}.npz") as arq:
        plano = dict(arq)
    with np.load(CACHE_BASE / MOTOR / f"{video}.npz") as arq:
        np.testing.assert_array_equal(arq["frames"], plano["frames"])
        np.testing.assert_array_equal(arq["computed"], plano["requested"])
        plano["det"], plano["offsets"] = arq["det"], arq["offsets"]
    np.testing.assert_array_equal(plano["requested"], plano["masks"].any(axis=2))
    assert len(plano["offsets"]) == len(plano["frames"])*len(plano["tiles"])+1
    assert plano["offsets"][-1] == len(plano["det"]) and plano["offsets"][0] == 0
    plano["indice"] = {int(q): i for i,q in enumerate(plano["frames"])}
    return plano


def inferencia_recorte(imagem, x, y, h, motor):
    if h == 40:
        return postprocess(motor(blob(imagem[y:y+T, x:x+T])))
    lado = round(T*h/40)
    margem = (lado-T)//2
    x0, y0 = x-margem, y-margem
    altura, largura = imagem.shape[:2]
    janela = imagem[max(y0,0):min(y0+lado,altura), max(x0,0):min(x0+lado,largura)]
    topo, base = max(0,-y0), max(0,y0+lado-altura)
    esq, direita = max(0,-x0), max(0,x0+lado-largura)
    if topo or base or esq or direita:
        janela = cv2.copyMakeBorder(janela, topo, base, esq, direita, cv2.BORDER_CONSTANT, value=(114,114,114))
    assert janela.shape[:2] == (lado,lado)
    reduzida = cv2.resize(janela, (T,T), interpolation=cv2.INTER_AREA)
    det = postprocess(motor(blob(reduzida)))
    det[:, :4] = det[:, :4]*(lado/T)-margem
    centros = (det[:,:2]+det[:,2:4])/2
    det = det[((centros >= 0) & (centros < T)).all(axis=1)]
    det[:, :4] = np.clip(det[:, :4], 0, T)
    return det


def ler_quadro(camera, q, mjpg):
    proximo = round(camera.get(cv2.CAP_PROP_POS_FRAMES))
    if proximo != q:
        if proximo > q or mjpg or q-proximo > BLOCO:
            if not camera.set(cv2.CAP_PROP_POS_FRAMES, q):
                raise IOError(f"Falha ao posicionar no quadro {q}")
        else:
            for _ in range(proximo, q):
                if not camera.grab():
                    raise IOError(f"Vídeo interrompido antes do quadro {q}")
    ok, imagem = camera.read()
    if not ok or round(camera.get(cv2.CAP_PROP_POS_FRAMES)) != q+1:
        raise IOError(f"Falha ao ler o quadro {q}")
    return imagem


def calcular_bloco(plano, etapa, h, inicio, fim, camera, mjpg, motor):
    integral = etapa == "integral"
    selecionados = np.ones((len(plano["tiles"]),1), dtype=bool)
    n_metodos = 1 if integral else len(METODOS)
    caixas = [[] for _ in range(n_metodos)]
    offsets = [[0] for _ in range(n_metodos)]
    recortes = np.zeros(n_metodos, dtype=int)
    enviados = np.zeros(n_metodos, dtype=int)
    quadros = np.arange(inicio, fim) if integral else plano["frames"][(plano["frames"]>=inicio)&(plano["frames"]<fim)]
    novos = reusados = 0
    for q in quadros:
        i = plano["indice"].get(int(q))
        mascara = selecionados if integral else plano["masks"][i]
        recortes += mascara.sum(axis=0)
        enviados += mascara.any(axis=0)
        por_metodo = [[] for _ in range(n_metodos)]
        imagem = None
        for j in np.flatnonzero(mascara.any(axis=1)):
            x, y = plano["tiles"][j]
            if h == 40 and i is not None and plano["requested"][i,j]:
                k = i*len(plano["tiles"])+j
                a, b = plano["offsets"][k:k+2]
                det = plano["det"][a:b]
                reusados += 1
            else:
                if imagem is None:
                    imagem = ler_quadro(camera, int(q), mjpg)
                    assert tuple(imagem.shape[1::-1]) == tuple(plano["size"])
                det = inferencia_recorte(imagem, x, y, h, motor)
                novos += 1
            det = det[det[:,4] >= METRICA["conf"]]
            if len(det):
                det[:, :4] += (x,y,x,y)
                for m in np.flatnonzero(mascara[j]):
                    por_metodo[m].append(det)
        for m, partes in enumerate(por_metodo):
            final = VAZIO
            if partes:
                juntas = np.concatenate(partes)
                final = juntas[indices_nms_quadro(juntas)]
            caixas[m].append(final)
            offsets[m].append(offsets[m][-1]+len(final))
    retorno = dict(frames=quadros, recortes=recortes, enviados=enviados, novos=novos, reusados=reusados)
    for m in range(n_metodos):
        retorno[f"det{m}"] = np.concatenate(caixas[m]) if caixas[m] else VAZIO
        retorno[f"idx{m}"] = np.asarray(offsets[m])
    return retorno


def avaliar_blocos(video, plano, arquivos, etapa, h):
    metodos = ("integral",) if etapa == "integral" else METODOS
    quadros, previsoes = [], [[] for _ in metodos]
    recortes, enviados = np.zeros(len(metodos),int), np.zeros(len(metodos),int)
    novos = reusados = 0
    for caminho in arquivos:
        with np.load(caminho) as dados:
            frames = dados["frames"]
            quadros.extend(frames)
            recortes += dados["recortes"]
            enviados += dados["enviados"]
            novos += int(dados["novos"])
            reusados += int(dados["reusados"])
            for m in range(len(metodos)):
                coords, idx = dados[f"det{m}"], dados[f"idx{m}"]
                assert len(idx)==len(frames)+1 and idx[0]==0 and idx[-1]==len(coords) and np.all(np.diff(idx)>=0)
                previsoes[m].extend(np.split(coords, idx[1:-1]) if len(frames) else [])
    n, fps = int(plano["total_frames"]), float(plano["fps"])
    quadros = np.asarray(quadros, dtype=int)
    if etapa == "integral":
        np.testing.assert_array_equal(quadros, np.arange(n))
        assert recortes[0] == novos+reusados == n*len(plano["tiles"]) and reusados==plano["requested"].sum()
    else:
        np.testing.assert_array_equal(quadros, plano["frames"])
        np.testing.assert_array_equal(recortes, plano["masks"].sum(axis=(0,1)))
        assert recortes[0]==recortes[1] and novos+reusados==plano["requested"].sum()
    inicio, centros = referencias(video, n, fps, plano["size"], plano["gsd"])
    log, = (folder(video) / "log").glob("*.csv")
    pose = np.column_stack(load_pose(log,n,fps))[quadros]
    clip = dict(frames=quadros, gt=labels(video, {"frames":quadros}), centers=centros,
                pose=pose, size=plano["size"], gsd=plano["gsd"])
    trilhas, voos = [], []
    for m, metodo in enumerate(metodos):
        assert len(previsoes[m])==len(quadros)
        contagens, acertos = focus_counts(clip, projected(clip,previsoes[m]), METRICA["conf"],
                                          METRICA["raio_m"], METRICA["dmax_m"], return_matches=True)
        primeiros = primeiros_alertas(clip, previsoes[m])
        assert set(primeiros) <= set(inicio)
        identificacao = dict(video=video, metodo=metodo, altura_m=h)
        linhas = []
        for (classe,tid), entrada in inicio.items():
            q = primeiros.get((classe,tid))
            assert q is None or q >= entrada
            linhas.append(dict(**identificacao, classe=CLASSES[classe], trilha=tid, quadro_entrada=entrada,
                               acerto_foco=int((classe,tid) in acertos), primeiro_alerta_quadro=q,
                               atraso_s=(q-entrada)/fps if q is not None else np.nan))
        voos.append(dict(**identificacao, quadros_voo=n, recortes=recortes[m], quadros_enviados=enviados[m],
                         **ratios(contagens.sum(axis=0)), **estatisticas(pd.DataFrame(linhas))))
        trilhas.extend(linhas)
    return pd.DataFrame(voos), pd.DataFrame(trilhas)


def validar_controle(video, voos, trilhas):
    for df, nome, chaves in [(voos,"por_voo.csv",["metodo"]), (trilhas,"por_trilha.csv",["metodo","classe","trilha"])]:
        ref = pd.read_csv(BASE/nome).query("video == @video").set_index(chaves).sort_index()
        atual = df.drop(columns="altura_m").set_index(chaves).sort_index()
        pd.testing.assert_frame_equal(atual, ref[atual.columns], check_dtype=False, atol=5.1e-8, rtol=0)


def consolidar(pasta, completo):
    fontes = [pasta/f"{etapa}_{h}"/v for etapa,h in ETAPAS for v in VOOS]
    prontas = [p for p in fontes if (p/"fim.json").is_file()]
    voos = pd.concat([pd.read_csv(p/"por_voo.csv") for p in prontas], ignore_index=True)
    tr = pd.concat([pd.read_csv(p/"por_trilha.csv") for p in prontas], ignore_index=True)
    assert not tr.duplicated(["video","metodo","altura_m","classe","trilha"]).any()
    resumo = []
    for (metodo,h), grupo in voos.groupby(["metodo","altura_m"], sort=False):
        if set(grupo.video) != set(VOOS):
            continue
        trilhas = tr[(tr.metodo==metodo)&(tr.altura_m==h)]
        assert len(trilhas)==grupo.referencias.sum()==1924 and trilhas.acerto_foco.sum()==grupo.vp.sum()
        resumo.append(dict(metodo=metodo, altura_m=h, voos=len(grupo), recortes=grupo.recortes.sum(),
                           **ratios(grupo[["vp","fp","fn"]].sum()), **estatisticas(trilhas)))
    if completo:
        assert len(prontas)==len(fontes) and len(resumo)==11
    if resumo:
        r = pd.DataFrame(resumo)
        referencias = r[r.altura_m==40].set_index("metodo")
        for col, nome, fator in [("recall","delta_recall_pp",100),("fp","delta_fp",1),
                                 ("mediana_s","delta_mediana_s",1),("confirmados_ate_1s","delta_confirmados_ate_1s",1)]:
            r[nome] = (r[col]-r.metodo.map(referencias[col]))*fator
            r.loc[r.metodo=="integral",nome] = np.nan
        escrever_csv(r, DESTINO/"resumo_seis_voos.csv")
    escrever_csv(voos, DESTINO/"por_voo.csv")
    escrever_csv(tr, DESTINO/"por_trilha.csv")


def executar(estado):
    verificar_pesos()
    setup()
    config = protocolo()
    chave = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    pasta = TRABALHO/chave
    anterior = DESTINO/"protocolo.json"
    if anterior.exists() and json.loads(anterior.read_text(encoding="utf-8")).get("assinatura") != chave:
        raise RuntimeError("Os resultados existentes correspondem a outra versão do ensaio.")
    escrever_json(anterior, dict(assinatura=chave, **config))
    motor = runner(MOTOR)
    for etapa,h in ETAPAS:
        for video in VOOS:
            local = pasta/f"{etapa}_{h}"/video
            local.mkdir(parents=True, exist_ok=True)
            if (local/"fim.json").is_file():
                continue
            plano = carregar_plano(video)
            n = int(plano["total_frames"])
            fonte, = (folder(video)/"avi").glob("*.avi")
            camera = cv2.VideoCapture(str(fonte)) if etapa != "controle" else None
            arquivos = []
            try:
                if camera is not None and (not camera.isOpened() or round(camera.get(cv2.CAP_PROP_FRAME_COUNT)) != n):
                    raise IOError(f"Contagem de quadros incompatível com o plano de {video}")
                mjpg = camera is not None and int(camera.get(cv2.CAP_PROP_FOURCC)) == cv2.VideoWriter_fourcc(*"MJPG")
                for inicio in range(0,n,BLOCO):
                    caminho = local/f"q{inicio:06d}.npz"
                    arquivos.append(caminho)
                    if caminho.exists():
                        continue
                    fim = min(inicio+BLOCO,n)
                    estado.update(etapa=etapa, altura_m=h, video=video, quadros_concluidos=inicio,
                                  quadros_voo=n, atualizado=hora())
                    escrever_json(TRABALHO/"progresso.json",estado)
                    t = time.monotonic()
                    dados = calcular_bloco(plano,etapa,h,inicio,fim,camera,mjpg,motor)
                    gravar_npz(caminho,**dados)
                    print(hora(),etapa,h,video,f"{fim}/{n}",dados["novos"],"inferências novas",
                          round(time.monotonic()-t,1),"s",flush=True)
            finally:
                if camera is not None:
                    camera.release()
            estado.update(etapa=etapa, altura_m=h, video=video, quadros_concluidos=n,
                          quadros_voo=n, atualizado=hora())
            escrever_json(TRABALHO/"progresso.json",estado)
            voos,trilhas = avaliar_blocos(video,plano,arquivos,etapa,h)
            if etapa=="controle":
                validar_controle(video,voos,trilhas)
            escrever_csv(voos,local/"por_voo.csv")
            escrever_csv(trilhas,local/"por_trilha.csv")
            escrever_json(local/"fim.json",dict(assinatura=chave, video=video, etapa=etapa, altura_m=h))
            consolidar(pasta,False)
    consolidar(pasta,True)


def main():
    TRABALHO.mkdir(parents=True,exist_ok=True)
    DESTINO.mkdir(parents=True,exist_ok=True)
    with FileLock(str(TRABALHO/"execucao.lock"),timeout=0), torch.inference_mode():
        estado = dict(situacao="executando", pid=os.getpid(), inicio=hora())
        escrever_json(TRABALHO/"progresso.json",estado)
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # Mantém o sistema acordado durante o cálculo.
        try:
            executar(estado)
            estado.update(situacao="concluido", atualizado=hora())
        except BaseException as exc:
            estado.update(situacao="interrompido", atualizado=hora(), erro=str(exc))
            traceback.print_exc()
            raise
        finally:
            escrever_json(TRABALHO/"progresso.json",estado)
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
