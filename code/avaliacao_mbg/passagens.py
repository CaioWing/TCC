"""K vistas por passagem de uma parcela fixa do terreno pelo campo da câmera."""
import argparse
import json

import numpy as np
import pandas as pd

from adaptativa import METRICA, MOTOR, TEMP as NOVOS, VOOS, gravar_npz, verificar_pesos
from avaliar import ratios
from dados import CACHE, GSD, OUT, folder
from memoria_sim import T, load_pose
from metricas import CLASSES, focus_counts, load_clips, projected, selected_tiles
from seguimento import FASES, temporal

DESTINO = OUT / "passagens"
MASCARAS = CACHE / "passagens"
PLANOS = {
    "entrada": ("entrada",),
    "meio": ("meio",),
    "extremos": ("entrada", "saida"),
    "tres": ("entrada", "meio", "saida"),
}
CONJUNTOS = {"iniciais": CACHE, "novos": NOVOS}
LADO = T * GSD


def salvar(df, arquivo):
    destino = DESTINO / arquivo
    tmp = destino.with_suffix(".partial.csv")
    df.to_csv(tmp, index=False, float_format="%.8f")
    tmp.replace(destino)


def marco_meio(inicio, fim, percurso):
    if fim - inicio < 2:
        return inicio
    deslocamento = percurso[fim] - percurso[inicio]
    if deslocamento < 1e-6:
        return (inicio + fim) // 2
    alvo = percurso[inicio] + deslocamento / 2
    return inicio + 1 + int(np.abs(percurso[inicio + 1:fim] - alvo).argmin())


def melhor_recorte(u, v, tiles):
    dentro = ((tiles[:, 0] <= u) & (u < tiles[:, 0] + T) &
              (tiles[:, 1] <= v) & (v < tiles[:, 1] + T))
    candidatos = np.flatnonzero(dentro)
    distancia = np.square(tiles[candidatos] + T/2 - (u, v)).sum(axis=1)
    return int(candidatos[distancia.argmin()])


def passagens_do_voo(pose, tamanho, tiles, deslocamento_grade=(0., 0.)):
    largura, altura = tamanho
    margem = GSD * np.hypot(largura, altura) / 2
    origem = np.array(deslocamento_grade)
    minimo = np.ceil((pose[:, :2].min(axis=0) - margem - origem) / LADO).astype(int)
    maximo = np.floor((pose[:, :2].max(axis=0) + margem - origem) / LADO).astype(int)
    cos, sen = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    percurso = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1))]
    for i in range(minimo[0], maximo[0] + 1):
        for j in range(minimo[1], maximo[1] + 1):
            leste, norte = i * LADO + origem[0], j * LADO + origem[1]
            de = (leste - pose[:, 0]) / GSD
            dn = (norte - pose[:, 1]) / GSD
            u = largura / 2 + cos * de - sen * dn
            v = altura / 2 - sen * de - cos * dn
            dentro = (u >= 0) & (u < largura) & (v >= 0) & (v < altura)
            mudancas = np.flatnonzero(np.diff(np.r_[False, dentro, False]))
            for numero, (inicio, apos) in enumerate(mudancas.reshape(-1, 2), 1):
                fim = apos - 1
                meio = marco_meio(inicio, fim, percurso)
                linha = dict(i=i, j=j, passagem=numero, leste_m=leste, norte_m=norte,
                             quadros_visiveis=apos-inicio, percurso_m=percurso[fim]-percurso[inicio])
                for nome, quadro in zip(("entrada", "meio", "saida"), (inicio, meio, fim)):
                    tile = melhor_recorte(u[quadro], v[quadro], tiles)
                    linha.update({f"quadro_{nome}": quadro, f"recorte_{nome}": tile,
                                  f"u_{nome}": u[quadro], f"v_{nome}": v[quadro]})
                yield linha


def exemplo_para_figura(registros, video, tiles, altura):
    """Exemplo geométrico do primeiro voo: travessia vertical bem centrada nos recortes."""
    avaliados = []
    for cache in CONJUNTOS.values():
        with np.load(cache / f"{video}.npz") as dados:
            avaliados.extend(dados["frames"].tolist())
    candidatos = registros.copy()
    marcos = PLANOS["tres"]
    candidatos["avaliados"] = sum(candidatos[f"quadro_{m}"].isin(avaliados) for m in marcos)
    candidatos["margem_horizontal"] = np.min([
        T/2 - np.abs(candidatos[f"u_{m}"] - tiles[candidatos[f"recorte_{m}"], 0] - T/2)
        for m in marcos], axis=0)
    travessia = (candidatos.v_saida - candidatos.v_entrada).abs()
    centrais = candidatos[(travessia > .97*altura) & (candidatos.margem_horizontal >= T/3)
                          & ((candidatos.v_meio - altura/2).abs() < .04*altura)]
    caso = centrais.sort_values(["avaliados", "margem_horizontal"], ascending=False).iloc[0]
    linhas = []
    for m in marcos:
        t, rec = int(caso[f"quadro_{m}"]), int(caso[f"recorte_{m}"])
        x, y = tiles[rec]
        linhas.append(dict(video=video, fase=m, quadro=t, u=caso[f"u_{m}"], v=caso[f"v_{m}"],
                           x=x, y=y, recorte=rec, leste_m=caso.leste_m, norte_m=caso.norte_m,
                           nos_trechos=int(t in avaliados)))
    salvar(pd.DataFrame(linhas), "exemplo_passagem.csv")


def carregar_voo(video):
    with np.load(CACHE / f"{video}.npz") as dados:
        n, fps = int(dados["total_frames"]), float(dados["fps"])
        tamanho, tiles = dados["size"], dados["tiles"]
    log, = (folder(video) / "log").glob("*.csv")
    return np.column_stack(load_pose(log, n, fps)), tamanho, tiles, fps


def mascara_eventos(registros, marcos, n, grade):
    mask = np.zeros((n, grade), dtype=bool)
    for marco in marcos:
        mask[registros[f"quadro_{marco}"], registros[f"recorte_{marco}"]] = True
    return mask


def preparar():
    eventos, demandas = [], []
    for video in VOOS:
        pose, tamanho, tiles, fps = carregar_voo(video)
        n = len(pose)
        registros = pd.DataFrame(passagens_do_voo(pose, tamanho, tiles))
        registros.insert(0, "video", video)
        if video == VOOS[0]:
            exemplo_para_figura(registros, video, tiles, tamanho[1])
        eventos.append(registros)
        masks = {}
        for nome, marcos in PLANOS.items():
            mask = mascara_eventos(registros, marcos, n, len(tiles))
            masks[nome] = mask
            k = len(marcos)
            vistas_possiveis = np.minimum(k, registros.quadros_visiveis)
            # Todos os marcos são distintos quando a passagem tem pelo menos K quadros.
            vistos = registros[[f"quadro_{m}" for m in marcos]].nunique(axis=1)
            assert np.array_equal(vistos, vistas_possiveis)
            demandas.append(dict(video=video, plano=nome, k=k, lado_m=LADO,
                                 quadros_voo=n, fps=fps, passagens=len(registros),
                                 passagens_com_k=int((vistos == k).sum()),
                                 pedidos_distintos_por_passagem=int(vistos.sum()),
                                 recortes_unicos=int(mask.sum()), quadros_usados=int(mask.any(axis=1).sum()),
                                 recortes_s=mask.sum()*fps/n))
        assert np.all(~masks["entrada"] | masks["extremos"])
        assert np.all(~masks["extremos"] | masks["tres"])
        assert np.all(~masks["meio"] | masks["tres"])
        gravar_npz(MASCARAS / f"{video}.npz", **masks)
    salvar(pd.concat(eventos, ignore_index=True), "eventos.csv")
    salvar(pd.DataFrame(demandas), "geometria.csv")


def planos(mascaras, frames, id_trecho):
    yield "integral", "integral", -1., np.ones((len(frames), 24), dtype=bool)
    for plano in PLANOS:
        mask = mascaras[plano][frames]
        yield plano, "geografica", -1., mask
        for fase in FASES:
            yield plano, "temporal", fase, temporal(mask.shape, int(mask.sum()), fase, id_trecho)


def resumir(df, chaves):
    totais = []
    for chave, grupo in df.groupby(chaves):
        extra = grupo[["recortes", "quadros_usados", "comuns_integral", "so_integral",
                       "so_metodo"]].sum().to_dict()
        totais.append(dict(zip(chaves, chave)) | extra | ratios(grupo[["vp", "fp", "fn"]].sum()))
    return pd.DataFrame(totais)


def pontuar_mascara(clip, mask, recuperar, memo):
    """Reutiliza projeções apenas entre seleções idênticas no mesmo quadro do trecho."""
    pontos = []
    for t in np.flatnonzero(mask.any(axis=1)):
        chave = (t, mask[t].tobytes())
        if chave not in memo:
            frame = dict(tiles=[clip["tiles"][t]], size=clip["size"],
                         pose=clip["pose"][t:t+1], gsd=clip["gsd"])
            boxes = recuperar(frame, mask[t:t+1])
            memo[chave] = projected(frame, boxes)[0]
        pontos.append(memo[chave])
    return focus_counts(clip, pontos, METRICA["conf"], METRICA["raio_m"],
                        METRICA["dmax_m"], return_matches=True)


def avaliar(pos):
    if pos == "fronteiras":
        from fronteiras import selected_tiles_fronteiras as recuperar
    else:
        recuperar = selected_tiles
    resultado, ids = [], []
    anterior = pd.read_csv(OUT / "adaptativa/por_trecho.csv").set_index(
        ["conjunto", "video", "trecho", "metodo", "fase"])
    for numero, video in enumerate(VOOS):
        with np.load(MASCARAS / f"{video}.npz") as dados:
            mascaras = dict(dados)
        for conjunto, cache in CONJUNTOS.items():
            for clip in load_clips(video, MOTOR, cache=cache):
                memo = {}
                for plano, estrategia, fase, mask in planos(mascaras, clip["frames"], numero*2 + clip["clip"]-1):
                    contagem, acertos = pontuar_mascara(clip, mask, recuperar, memo)
                    if estrategia == "integral":
                        integral = acertos
                        if pos == "nms":
                            registro = anterior.loc[(conjunto, video, clip["clip"], "integral", -1.)]
                            np.testing.assert_array_equal(contagem.sum(axis=0), registro[["vp","fp","fn"]])
                    r = dict(conjunto=conjunto, video=video, trecho=clip["clip"], pos=pos, plano=plano,
                             k=len(PLANOS.get(plano, ())), estrategia=estrategia, fase=fase,
                             recortes=int(mask.sum()), quadros_usados=int(mask.any(axis=1).sum()),
                             comuns_integral=len(acertos & integral), so_integral=len(integral - acertos),
                             so_metodo=len(acertos - integral), **ratios(contagem.sum(axis=0)))
                    resultado.append(r)
                    if estrategia != "temporal":
                        ids.extend(dict(conjunto=conjunto, video=video, trecho=clip["clip"], pos=pos,
                                        plano=plano, classe=CLASSES[cls], trilha=trilha,
                                        acerto=int((cls, trilha) in acertos))
                                   for cls, trilha in clip["centers"])
                assert len(integral) <= len(clip["centers"])
        print(video, pos, flush=True)
    df = pd.DataFrame(resultado)
    assert len(df) == 24*(1+len(PLANOS)*(1+len(FASES)))
    assert (df.vp == df.comuns_integral + df.so_metodo).all()
    pareadas = df[df.estrategia != "integral"].groupby(["conjunto","video","trecho","plano"])
    assert pareadas.size().eq(1+len(FASES)).all() and pareadas.recortes.nunique().eq(1).all()
    campos = ["pos", "plano", "k", "estrategia", "fase"]
    salvar(df, f"por_trecho_{pos}.csv")
    salvar(pd.DataFrame(ids), f"pareamento_{pos}.csv")
    salvar(resumir(df, ["conjunto", *campos]), f"resumo_conjuntos_{pos}.csv")
    salvar(resumir(df, ["video", *campos]), f"por_voo_{pos}.csv")
    salvar(resumir(df, campos), f"resumo_{pos}.csv")


def sensibilidade_origem():
    from fronteiras import selected_tiles_fronteiras
    linhas = []
    for video in VOOS:
        pose, tamanho, tiles, _ = carregar_voo(video)
        trechos = [(conjunto, clip) for conjunto, cache in CONJUNTOS.items()
                   for clip in load_clips(video, MOTOR, cache=cache)]
        for leste, norte in [(0., 0.), (LADO/2, 0.), (0., LADO/2), (LADO/2, LADO/2)]:
            registros = pd.DataFrame(passagens_do_voo(pose, tamanho, tiles, (leste, norte)))
            for plano, marcos in PLANOS.items():
                mask = mascara_eventos(registros, marcos, len(pose), len(tiles))
                for conjunto, clip in trechos:
                    selecao = mask[clip["frames"]]
                    boxes = selected_tiles_fronteiras(clip, selecao)
                    contagem = focus_counts(clip, projected(clip, boxes), METRICA["conf"],
                                            METRICA["raio_m"], METRICA["dmax_m"])
                    linhas.append(dict(conjunto=conjunto, video=video, trecho=clip["clip"], plano=plano,
                                       leste_m=leste, norte_m=norte, recortes=int(selecao.sum()),
                                       **ratios(contagem.sum(axis=0))))
    df = pd.DataFrame(linhas)
    campos = ["vp", "fp", "fn", "recortes"]
    origem = df[(df.leste_m == 0) & (df.norte_m == 0)]
    chaves = ["conjunto", "video", "trecho", "plano"]
    principal = pd.read_csv(DESTINO / "por_trecho_fronteiras.csv").query("estrategia == 'geografica'")
    pd.testing.assert_frame_equal(origem.set_index(chaves)[campos].sort_index(),
                                  principal.set_index(chaves)[campos].sort_index())
    salvar(df, "sensibilidade_origem_por_trecho.csv")
    resumo = []
    for (leste, norte, plano), grupo in df.groupby(["leste_m", "norte_m", "plano"]):
        resumo.append(dict(leste_m=leste, norte_m=norte, plano=plano, recortes=grupo.recortes.sum(),
                           **ratios(grupo[["vp","fp","fn"]].sum())))
    salvar(pd.DataFrame(resumo), "sensibilidade_origem.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=["preparar", "nms", "fronteiras", "origem"])
    etapa = parser.parse_args().etapa
    verificar_pesos()
    DESTINO.mkdir(exist_ok=True)
    if etapa == "preparar":
        protocolo = dict(lado_parcela_m=LADO, origem="EN local ao voo, centro da grade em (0,0).",
                         unidade="Passagem contínua do centro de uma parcela pela imagem. Nova entrada reinicia K.",
                         planos=PLANOS, fase_meio="Metade da distância horizontal acumulada da câmera, "
                         "com quadro intermediário distinto se existir; em repouso usa o meio temporal.",
                         bordas_voo="Primeiro e último quadros gravados delimitam passagens ainda abertas.",
                         recorte="Entre os recortes 640×640 que contêm o centro, escolher o de centro mais "
                         "próximo em pixels; empates seguem a ordem da grade. Unir pedidos no mesmo quadro e recorte.",
                         voo="Trajetória integral conhecida na reprodução, inclusive para meio e saída.",
                         sensibilidade_adicional="Após o resultado principal, transladar a origem por meia "
                         "parcela para leste, norte e ambas as direções, sem selecionar a melhor malha.",
                         teste="24 trechos de quatro segundos usados na comparação por cobertura, "
                         "quatro por voo. Referência (classe, trilha) em cada trecho.",
                         controle="Quatro fases da amostragem temporal com número idêntico de recortes por trecho e plano.",
                         pesos_sha256=METRICA["pesos_sha256"], motor=MOTOR, voos=VOOS,
                         criterio="Descrever recall, precisão, F1, recortes e interseção de referências "
                         "com o integral; não estabelecer equivalência ou otimizar K nos voos de teste.")
        (DESTINO / "protocolo.json").write_text(json.dumps(protocolo, ensure_ascii=False, indent=2)+"\n",
                                                encoding="utf-8")
        preparar()
    elif etapa == "origem":
        sensibilidade_origem()
    else:
        avaliar(etapa)
