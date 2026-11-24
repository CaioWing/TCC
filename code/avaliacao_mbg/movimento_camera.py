"""Compara cadências por tempo e por deslocamento aparente nos mesmos quadros de teste."""
import numpy as np
import pandas as pd

from adaptativa import MOTOR, VOOS, verificar_pesos
from avaliar import ratios
from dados import tiles_grid
from fronteiras import selected_tiles_fronteiras
from metricas import ground, load_clips
from passagens import CONJUNTOS, DESTINO, pontuar_mascara, salvar
from seguimento import FASES


def pontos_medicao(tamanho):
    """Centro e centros dos quadrantes, em pixels em relação ao centro do quadro."""
    return np.array([[0, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]]) * tamanho / 4


def deslocamentos(pose, tamanho, gsd):
    """Reprojeta pontos do quadro corrente no próximo, incluindo a guinada do gimbal."""
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    matrizes = np.stack([c, -s, -s, -c], axis=1).reshape(-1, 2, 2)
    pontos = pontos_medicao(tamanho)
    terreno = (matrizes[:-1] @ pontos.T) - np.diff(pose[:, :2], axis=0)[:, :, None] / gsd
    return (matrizes[1:] @ terreno).transpose(0, 2, 1) - pontos


def indices_por_movimento(indices, passos):
    percurso = np.r_[0., np.cumsum(passos)]
    if percurso[-1] < 1e-8:
        return indices.copy()
    alvos = indices * percurso[-1] / (len(percurso) - 1)
    escolhidos = []
    inicio = 0
    for n, alvo in enumerate(alvos):
        fim = len(percurso) - len(alvos) + n + 1
        q = inicio + np.abs(percurso[inicio:fim] - alvo).argmin()
        escolhidos.append(q)
        inicio = q + 1
    return np.array(escolhidos)


def direcao_das_trilhas(clip):
    """Valida deslocamentos em pares separados por 30 quadros sem usar caixas na seleção."""
    tamanho = np.asarray(clip["size"])
    poses = clip["pose"]
    pares = []
    for t in range(len(poses) - 30):
        f = t + 30
        primeiro, segundo = clip["gt"][t], clip["gt"][f]
        caixas = {tuple(box[4:].astype(int)): box for box in segundo}
        solo = ground(primeiro, poses[t], tamanho, clip["gsd"])
        delta = (solo - poses[f, :2]) / clip["gsd"]
        c, s = np.cos(poses[f, 2]), np.sin(poses[f, 2])
        proj = tamanho / 2 + delta @ np.array([[c, -s], [-s, -c]])
        for box, previsao in zip(primeiro, proj):
            chave = tuple(box[4:].astype(int))
            if chave not in caixas:
                continue
            ambas = np.stack([box[:4], caixas[chave][:4]])
            if (ambas[:, :2] <= 0).any() or (ambas[:, 2:] >= tamanho - 1).any():
                continue
            a, b = (ambas[:, :2] + ambas[:, 2:]) / 2
            obs, est = b - a, previsao - a
            if min(np.linalg.norm(obs), np.linalg.norm(est)) < 1:
                continue
            angulo = np.degrees(np.arctan2(min(abs(obs)), max(abs(obs))))
            dif = np.degrees(np.arctan2(abs(obs[0]*est[1] - obs[1]*est[0]), obs @ est))
            pares.append(dict(trilha=chave[0], classe=chave[1], angulo_observado=angulo,
                              erro_direcao=dif, erro_px=np.linalg.norm(obs - est)))
    if not pares:
        return pd.DataFrame()
    return (pd.DataFrame(pares).groupby(["classe", "trilha"], as_index=False)
            .agg(pares=("erro_px","size"), angulo_observado=("angulo_observado","median"),
                 erro_direcao=("erro_direcao","median"), erro_px=("erro_px","median")))


def planos(clip):
    frames = clip["frames"]
    assert len(frames) == 120 and (np.diff(frames) == 1).all()
    tamanho = np.asarray(clip["size"])
    fluxo = deslocamentos(clip["pose"], tamanho, clip["gsd"])
    origens = np.asarray(tiles_grid(*tamanho))
    passos_grade = [np.diff(np.unique(origens[:, eixo])).mean() for eixo in [0, 1]]
    eixos = abs(fluxo).mean(axis=1) / passos_grade
    dominante = eixos.sum(axis=0).argmax()
    yield "integral", 120, 0., np.arange(120)
    for b in [3, 4]:
        intervalo = len(frames) // b
        for fase in FASES:
            temporal = np.flatnonzero(frames % intervalo == int(fase * intervalo))
            assert len(temporal) == b
            yield "temporal", b, fase, temporal
            yield "dois_eixos", b, fase, indices_por_movimento(temporal, eixos.sum(axis=1))
            yield "eixo_dominante", b, fase, indices_por_movimento(temporal, eixos[:, dominante])


def avaliar():
    verificar_pesos()
    (DESTINO / "camera").mkdir(exist_ok=True)
    linhas, movimentos, trilhas = [], [], []
    for video in VOOS:
        for conjunto, cache in CONJUNTOS.items():
            for clip in load_clips(video, MOTOR, cache=cache):
                chave = dict(video=video, conjunto=conjunto, trecho=clip["clip"])
                trilhas.append(direcao_das_trilhas(clip).assign(**chave))
                movimento = deslocamentos(clip["pose"], clip["size"], clip["gsd"])
                u, v = movimento[:, 0].T
                modulo = np.hypot(u, v)
                ang = np.degrees(np.arctan2(np.minimum(abs(u), abs(v)), np.maximum(abs(u), abs(v))))
                movimentos.append(dict(**chave, primeiro=int(clip["frames"][0]),
                                       percurso_px=modulo.sum(), angulo_graus=np.average(ang, weights=modulo)
                                       if modulo.sum() > 0 else np.nan))
                memo, medidas = {}, {}
                for modo, b, fase, idx in planos(clip):
                    mask = np.zeros((len(clip["frames"]), len(clip["tiles"][0])), bool)
                    mask[idx] = True
                    assert mask.any(axis=1).sum() == b
                    cont, acertos = pontuar_mascara(clip, mask, selected_tiles_fronteiras, memo)
                    medidas[modo, b, fase] = (idx, mask.sum(), ratios(cont.sum(axis=0)), acertos)
                integral = medidas["integral", 120, 0.][3]
                for (modo, b, fase), (idx, recortes, metrica, ids) in medidas.items():
                    base = medidas["temporal", 4, fase][3]
                    linhas.append(dict(**chave, modo=modo, b=b, fase=fase, recortes=recortes,
                                       quadros=" ".join(str(q) for q in clip["frames"][idx]),
                                       comuns_integral=len(ids & integral), comuns_cada30=len(ids & base),
                                       ganhos_cada30=len(ids - base), perdas_cada30=len(base - ids), **metrica))
        print(video, flush=True)
    df = pd.DataFrame(linhas)
    chave = ["video", "conjunto", "trecho"]
    integral = df[df.modo == "integral"].set_index(chave)
    base = pd.read_csv(DESTINO / "por_trecho_fronteiras.csv").query("estrategia == 'integral'").set_index(chave)
    campos = ["vp", "fp", "fn", "recortes"]
    pd.testing.assert_frame_equal(integral[campos].sort_index(), base[campos].sort_index())
    assert len(integral) == 24 and len(df) == 24 * 25
    resumo = []
    for (modo, b, fase), g in df.groupby(["modo", "b", "fase"]):
        somas = g[["recortes", "comuns_integral", "comuns_cada30", "ganhos_cada30", "perdas_cada30"]].sum()
        resumo.append(dict(modo=modo, b=b, fase=fase, quadros_usados=len(g)*b, **somas.to_dict(),
                           **ratios(g[["vp", "fp", "fn"]].sum())))
    salvar(pd.DataFrame(movimentos), "camera/movimento.csv")
    salvar(pd.concat(trilhas, ignore_index=True), "camera/direcao_trilhas.csv")
    salvar(df, "camera/por_trecho.csv")
    salvar(pd.DataFrame(resumo), "camera/resumo.csv")


if __name__ == "__main__":
    avaliar()
