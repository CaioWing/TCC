"""Seleção por detalhe da primeira vista. Compara os mesmos recortes orçados por trecho."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from dados import CACHE, GSD, OUT, SPLITS, WEIGHTS, folder
from memoria_sim import T, grade_projecao, load_pose
from seguimento import FASES, temporal

MOTOR = "trt_fp16"
PASSO = 15
LIMITE = 12  # Meia grade por oportunidade, no máximo uma grade inteira a cada 30 quadros.
ESCALA = 4
JANELA = 9  # Aproxima 0,5 m com GSD de 1,35 cm/pixel na imagem reduzida a 1/4.
TEMP = CACHE / "adaptativa"
RESULTADOS = OUT / "adaptativa"
VOOS = SPLITS["voos"]["test"]
METRICA = json.loads((OUT / "protocolo.json").read_text(encoding="utf-8"))


def verificar_pesos():
    motor = json.loads((OUT / f"motor_{MOTOR}.json").read_text(encoding="utf-8"))
    for path, esperado in [(WEIGHTS, METRICA["pesos_sha256"]),
                           (CACHE / f"{MOTOR}.opt1.engine", motor["engine_sha256"])]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != esperado:
            raise ValueError(f"Identidade divergente: {path}")


def gravar_npz(destino, **dados):
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".partial.npz")
    np.savez_compressed(temporario, **dados)
    temporario.replace(destino)


def preparar_novos(video):
    with np.load(CACHE / f"{video}.npz") as anterior:
        n, fps = int(anterior["total_frames"]), float(anterior["fps"])
        tamanho, tiles = anterior["size"], anterior["tiles"]
        duracao = round(4 * fps)
        starts = [round(f * n) - duracao // 2 for f in (.2, .8)]
        frames = np.concatenate([np.arange(s, s + duracao) for s in starts])
        assert not np.intersect1d(frames, anterior["frames"]).size
    with np.load(CACHE / "seguimento" / f"{video}.npz") as seguimento:
        assert not np.intersect1d(frames, seguimento["frames"]).size
    log, = (folder(video) / "log").glob("*.csv")
    pose = np.column_stack(load_pose(log, n, fps))[frames]
    gravar_npz(TEMP / f"{video}.npz", frames=frames, starts=starts, size=tamanho,
               fps=fps, pose=pose, gsd=GSD, tiles=tiles,
               masks=np.ones((len(frames), len(tiles), 1), dtype=bool))


def inferir():
    import torch
    from inferir import infer_video
    from motores import runner, setup

    verificar_pesos()
    setup()
    with torch.inference_mode():
        run = runner(MOTOR)
        for video in VOOS:
            preparar_novos(video)
            infer_video(video, MOTOR, run, cache=TEMP)


def amostras_video(video, quadros):
    caminho, = (folder(video) / "avi").glob("*.avi")
    leitor = cv2.VideoCapture(str(caminho))
    codec = int(leitor.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
    try:
        for quadro in quadros:
            atual = round(leitor.get(cv2.CAP_PROP_POS_FRAMES))
            if atual != quadro:
                if codec == b"MJPG" or quadro - atual > PASSO:
                    leitor.set(cv2.CAP_PROP_POS_FRAMES, int(quadro))
                else:
                    for _ in range(quadro - atual):
                        if not leitor.grab():
                            raise IOError(f"Quadro ausente em {video}: {quadro}")
            ok, imagem = leitor.read()
            if not ok or round(leitor.get(cv2.CAP_PROP_POS_FRAMES)) != quadro + 1:
                raise IOError(f"Falha de leitura em {video}, quadro {quadro}")
            altura, largura = imagem.shape[:2]
            reduzida = cv2.resize(imagem, (largura // ESCALA, altura // ESCALA),
                                  interpolation=cv2.INTER_AREA)
            cinza = cv2.cvtColor(reduzida, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(cinza, cv2.CV_32F)
            variancia = cv2.blur(lap * lap, (JANELA, JANELA)) - cv2.blur(lap, (JANELA, JANELA)) ** 2
            yield quadro, variancia
    finally:
        leitor.release()


def geometria(video):
    with np.load(CACHE / f"{video}.npz") as info:
        n, fps = int(info["total_frames"]), float(info["fps"])
        largura, altura = map(int, info["size"])
        tiles = info["tiles"]
    log, = (folder(video) / "log").glob("*.csv")
    shape, raster = grade_projecao(log, n, fps, largura, altura, GSD)
    return n, tiles, shape, raster


def detalhe_no_solo(detalhe, u, v):
    return detalhe[(v / ESCALA).astype(int), (u / ESCALA).astype(int)]


def calibrar():
    valores = []
    for video in SPLITS["voos"]["val"]:
        _, _, _, raster = geometria(video)
        with np.load(CACHE / f"{video}.npz") as info:
            frames = info["frames"]
        for quadro, detalhe in amostras_video(video, frames[frames % PASSO == 0]):
            _, _, _, _, dentro, u, v = raster(quadro)
            valores.append(detalhe_no_solo(detalhe, u[dentro], v[dentro]))
    amostras = np.concatenate(valores)
    return float(np.quantile(amostras, .25)), len(amostras)


def selecionar_voo(video, limiar):
    n, tiles, shape, raster = geometria(video)
    # Uma célula ajuda a escolher o recorte antes da primeira vista e, se esta
    # tiver pouco detalhe, até receber uma segunda vista em outro quadro.
    estado = np.zeros(shape, dtype=np.uint8)  # 0: nova, 1: pede outra vista, 2: atendida
    solicitou = np.zeros(shape, dtype=bool)
    repetiu = np.zeros(shape, dtype=bool)
    imageada = np.zeros(shape, dtype=bool)
    selecao = np.zeros((n, len(tiles)), dtype=bool)
    for quadro, detalhe in amostras_video(video, range(0, n, PASSO)):
        j0, j1, i0, i1, dentro, u, v = raster(quadro)
        regiao = np.s_[j0:j1, i0:i1]
        imageada[regiao] |= dentro
        local = estado[regiao]
        precisa = dentro & (local < 2)
        if not precisa.any():
            continue
        yy, xx = np.nonzero(precisa)
        pu, pv = u[precisa], v[precisa]
        cobre = np.stack([(pu >= x) & (pu < x + T) & (pv >= y) & (pv < y + T)
                          for x, y in tiles])
        atendida = np.zeros(len(pu), dtype=bool)
        for _ in range(LIMITE):
            ganhos = cobre.sum(axis=1)
            melhor = int(ganhos.argmax())  # empate: ordem das linhas e colunas da grade
            if ganhos[melhor] == 0:
                break
            selecao[quadro, melhor] = True
            novas = cobre[melhor].copy()
            atendida |= novas
            cobre[:, novas] = False
        y, x = yy[atendida], xx[atendida]
        primeira = local[y, x] == 0
        pouco_detalhe = detalhe_no_solo(detalhe, u[y, x], v[y, x]) <= limiar
        pedir = primeira & pouco_detalhe
        solicitou[regiao][y[pedir], x[pedir]] = True
        repetiu[regiao][y[~primeira], x[~primeira]] = True
        local[y, x] = np.where(pedir, 1, 2)
    assert np.all(selecao.sum(axis=1) <= LIMITE)
    assert np.array_equal(solicitou & ~repetiu, estado == 1)
    gravar_npz(TEMP / "mascaras" / f"{video}.npz", selecao=selecao)
    row = dict(video=video, quadros=n, recortes=int(selecao.sum()),
               celulas_nos_quadros_candidatos=int(imageada.sum()),
               celulas_com_primeira=int((estado > 0).sum()),
               celulas_pediram_segunda=int(solicitou.sum()),
               celulas_com_segunda_solicitada=int(repetiu.sum()))
    print(video, row, flush=True)
    return row


def gravar_csv(tabela, nome):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    destino = RESULTADOS / nome
    temporario = destino.with_suffix(".partial.csv")
    tabela.to_csv(temporario, index=False, float_format="%.8f")
    temporario.replace(destino)


def linhas_da_tabela(tabela):
    def numero(n):
        return f"{int(n):,}".replace(",", ".")

    def intervalo(valores):
        a, b = valores.min(), valores.max()
        return numero(a) if a == b else f"{numero(a)}--{numero(b)}"

    def linha(rotulo, grupo):
        totais = grupo.groupby(["metodo", "fase"])[["recortes", "referencias", "vp", "fp"]].sum()
        geo, integral = [totais.loc[(metodo, -1.)] for metodo in ("adaptativa", "integral")]
        tempo = totais.loc["temporal"]
        assert (tempo.recortes == geo.recortes).all()
        campos = [rotulo, numero(geo.referencias), numero(geo.recortes), numero(geo.vp),
                  intervalo(tempo.vp), numero(integral.vp), numero(geo.fp), intervalo(tempo.fp)]
        return " & ".join(campos) + r" \\"

    novas = tabela[tabela.conjunto == "novos"]
    linhas = [linha(video[5:7], grupo) for video, grupo in novas.groupby("video")]
    linhas += [r"\hline", linha("Novos", novas), linha("Iniciais", tabela[tabela.conjunto == "iniciais"])]
    return "\\newcommand{\\linhasAdaptativas}{%\n" + "\n".join(linhas) + "%\n}\n"


def selecionar():
    cv2.setNumThreads(1)
    limiar, n = calibrar()
    print("Limiar de detalhe na validação:", limiar, "em", n, "células-quadro", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        linhas = list(pool.map(selecionar_voo, VOOS, [limiar] * len(VOOS)))
    gravar_csv(pd.DataFrame(linhas), "geometria.csv")
    config = dict(voos=VOOS, validacao=SPLITS["voos"]["val"], motor=MOTOR,
                  pesos_sha256=METRICA["pesos_sha256"], quantil=.25, limiar=limiar,
                  celulas_quadro_validacao=n, passo=PASSO, limite_por_quadro=LIMITE,
                  reducao=ESCALA, janela_detalhe=JANELA, fases_temporais=FASES,
                  primeiro_ensaio=[1/3, 2/3], novos_trechos=[.2, .8], duracao_s=4)
    (RESULTADOS / "protocolo.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def avaliar():
    from avaliar import ratios
    from metricas import CLASSES, focus_counts, load_clips, projected, selected_tiles

    verificar_pesos()
    params = [METRICA[key] for key in ("conf", "raio_m", "dmax_m")]
    linhas = []
    referencias = []
    historico = pd.concat([pd.read_csv(OUT / f"pareamento_{MOTOR}.csv"),
                           pd.read_csv(OUT / "seguimento/pareamento.csv")])
    anteriores = set(zip(historico.video, historico.classe, historico.trilha))
    for video in VOOS:
        with np.load(TEMP / "mascaras" / f"{video}.npz") as dados:
            mask_voo = dados["selecao"]
        for conjunto, cache in [("iniciais", CACHE), ("novos", TEMP)]:
            for clip in load_clips(video, MOTOR, cache=cache):
                mask = mask_voo[clip["frames"]]
                orcamento = int(mask.sum())
                metodos = [("adaptativa", -1., mask),
                           ("integral", -1., np.ones_like(mask))]
                id_trecho = VOOS.index(video) * 2 + clip["clip"] - 1
                for fase in FASES:
                    controle = temporal(mask.shape, orcamento, fase, id_trecho)
                    assert int(controle.sum()) == orcamento
                    metodos.append(("temporal", fase, controle))
                por_recipiente = {
                    key: dict(conjunto=conjunto, video=video, trecho=clip["clip"],
                              classe=CLASSES[key[0]], trilha=key[1],
                              constava_nos_ensaios_anteriores=int((video, CLASSES[key[0]], key[1])
                                                                  in anteriores))
                    for key in clip["centers"]
                }
                for nome, fase, selecionados in metodos:
                    caixas = selected_tiles(clip, selecionados)
                    counts, acertos = focus_counts(clip, projected(clip, caixas), *params,
                                                   return_matches=True)
                    met = ratios(counts.sum(axis=0))
                    assert met["referencias"] == len(clip["centers"])
                    coluna = f"temporal_{fase:g}" if nome == "temporal" else nome
                    for chave, entry in por_recipiente.items():
                        entry[coluna] = int(chave in acertos)
                    linhas.append(dict(conjunto=conjunto, video=video, trecho=clip["clip"],
                                       primeiro=int(clip["frames"][0]), ultimo=int(clip["frames"][-1]),
                                       metodo=nome, fase=fase, recortes=int(selecionados.sum()), **met))
                referencias.extend(por_recipiente.values())
        print(video, "pontuado", flush=True)
    tabela = pd.DataFrame(linhas)
    chaves = ["conjunto", "video", "trecho"]
    assert len(tabela.groupby(chaves)) == 24
    assert tabela.groupby(chaves).referencias.nunique().eq(1).all()
    assert tabela[tabela.metodo != "integral"].groupby(chaves).recortes.nunique().eq(1).all()
    colunas = ["recortes", "vp", "fp", "fn"]
    original = pd.read_csv(OUT / f"resumo_{MOTOR}.csv").set_index("estrategia")
    integral_inicial = tabela[(tabela.conjunto == "iniciais") & (tabela.metodo == "integral")]
    np.testing.assert_array_equal(integral_inicial[colunas].sum(), original.loc["todos", colunas])
    gravar_csv(tabela, "por_trecho.csv")
    gravar_csv(pd.DataFrame(referencias), "pareamento.csv")
    for grupos, nome in [(["conjunto", "video", "metodo", "fase"], "por_voo.csv"),
                         (["conjunto", "metodo", "fase"], "resumo.csv")]:
        resumo = []
        for chave, grupo in tabela.groupby(grupos):
            resumo.append(dict(zip(grupos, chave)) | dict(recortes=grupo.recortes.sum())
                          | ratios(grupo[["vp", "fp", "fn"]].sum()))
        gravar_csv(pd.DataFrame(resumo), nome)
    (RESULTADOS / "tabela_voos.tex").write_text(linhas_da_tabela(tabela), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("etapa", choices=["inferir", "selecionar", "avaliar"])
    args = parser.parse_args()
    if args.etapa == "inferir":
        inferir()
    elif args.etapa == "selecionar":
        selecionar()
    else:
        avaliar()
