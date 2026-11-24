"""Escolhe poucos quadros e recortes pela trajetória e compara máscaras de mesmo custo."""
import json
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd

from adaptativa import METRICA, MOTOR, TEMP, VOOS, verificar_pesos
from avaliar import ratios
from dados import CACHE, OUT, SPLITS
from memoria_sim import CELL, T, tiles_grid
from metricas import focus_counts, load_clips, projected, selected_tiles
from seguimento import FASES, temporal

DESTINO = OUT / "trajetoria"
QUADROS = (3, 4, 6)
PESOS_SEGUNDA = (0., 1.5)
ORCAMENTO_VALIDACAO = 48
DISTANCIA_SEGUNDA = 2.  # m entre centros da câmera para uma segunda vista
VELOCIDADE_NOMINAL = 4.1  # m/s, velocidade de cruzeiro da campanha
PARAMS = [METRICA[k] for k in ("conf", "raio_m", "dmax_m")]


def agenda(pose, fps, quantidade, fase):
    """Uma oportunidade em cada intervalo, ao atingir o deslocamento-alvo ou no fim."""
    escolhidos = []
    for indices in np.array_split(np.arange(len(pose)), quantidade):
        alvo = fase * VELOCIDADE_NOMINAL * len(indices) / fps
        origem = pose[indices[0], :2]
        for i in indices:
            if np.linalg.norm(pose[i, :2] - origem) >= alvo or i == indices[-1]:
                escolhidos.append(i)
                break
    return escolhidos


def preparar_grade(clip):
    largura, altura = clip["size"]
    gsd, poses = clip["gsd"], clip["pose"]
    tiles = tiles_grid(largura, altura)
    # Os limites só dimensionam os vetores. A malha é ancorada em múltiplos de CELL.
    margem = np.hypot(largura, altura) * gsd / 2 + CELL
    minimo = np.floor((poses[:, :2].min(axis=0) - margem) / CELL).astype(int)
    maximo = np.ceil((poses[:, :2].max(axis=0) + margem) / CELL).astype(int)
    leste, norte = np.meshgrid(np.arange(minimo[0], maximo[0] + 1) * CELL,
                               np.arange(minimo[1], maximo[1] + 1) * CELL)

    @lru_cache(maxsize=None)
    def recortes(i):
        e, n, yaw = poses[i]
        de, dn = (leste - e) / gsd, (norte - n) / gsd
        u = largura / 2 + np.cos(yaw) * de - np.sin(yaw) * dn
        v = altura / 2 - np.sin(yaw) * de - np.cos(yaw) * dn
        return [np.flatnonzero((u >= x) & (u < x + T) & (v >= y) & (v < y + T))
                for x, y in tiles]

    return recortes, leste.size


def selecionar(poses, grade, indices, limite, id_trecho, peso):
    """Peso None usa a rotação da grade. As duas regras têm a mesma cota por quadro."""
    recortes, tamanho = grade
    mascara = np.zeros((len(poses), 24), dtype=bool)
    primeira = np.full(tamanho, -1)
    segunda = np.zeros(tamanho, dtype=bool)
    for passo, quadro in enumerate(indices):
        areas = recortes(quadro)
        enviados = passo * limite // len(indices)
        cota = (passo + 1) * limite // len(indices) - enviados
        assert 0 < cota <= len(areas) == 24
        # Controle: percorre os 24 índices antes de reutilizar uma posição da imagem.
        ordem = list((id_trecho + 7 * (np.arange(24) + enviados)) % 24)
        conhecida = primeira >= 0
        apta = conhecida & ~segunda
        apta[conhecida] &= np.linalg.norm(
            poses[primeira[conhecida], :2] - poses[quadro, :2], axis=1) >= DISTANCIA_SEGUNDA
        coberta = np.zeros(tamanho, dtype=bool)
        if peso is not None:
            ganho = (~conhecida).astype(float) + peso * apta
        for _ in range(cota):
            if peso is None:
                tile = ordem[0]
            else:
                tile = max(ordem, key=lambda j: ganho[areas[j]].sum())
                ganho[areas[tile]] = 0.  # Sobreposição no mesmo quadro conta uma vez.
            ordem.remove(tile)
            mascara[quadro, tile] = True
            coberta[areas[tile]] = True
        segunda |= coberta & apta
        primeira[coberta & ~conhecida] = quadro
    assert mascara.sum() == limite and mascara.any(axis=1).sum() == len(indices)
    return mascara, dict(celulas_observadas=int((primeira >= 0).sum()),
                         celulas_segunda_distinta=int(segunda.sum()))


def pontuar(clip, mascara):
    contagens = focus_counts(clip, projected(clip, selected_tiles(clip, mascara)), *PARAMS)
    return dict(recortes=int(mascara.sum()), quadros_usados=int(mascara.any(axis=1).sum()),
                **ratios(contagens.sum(axis=0)))


def agregar(tabela, chaves):
    linhas = []
    for valores, grupo in tabela.groupby(chaves):
        linhas.append(dict(zip(chaves, valores)) | ratios(grupo[["vp", "fp", "fn"]].sum())
                      | dict(recortes=grupo.recortes.sum(), quadros_usados=grupo.quadros_usados.sum()))
    return pd.DataFrame(linhas)


def gravar(df, nome):
    destino = DESTINO / nome
    tmp = destino.with_suffix(".partial.csv")
    df.to_csv(tmp, index=False, float_format="%.8f")
    tmp.replace(destino)


def tabela_tcc(resumo, tabela):
    novos = resumo[resumo.conjunto == "novos"]
    linhas = []
    def intervalo(serie, percentual=False):
        def formato(x):
            return f"{100 * x:.1f}".replace(".", ",") if percentual else str(int(x))
        a, b = formato(serie.min()), formato(serie.max())
        return a if a == b else f"{a}--{b}"

    for metodo, nome in [("temporal", "Temporal por trecho"),
                         ("trajetoria_minima", "Recortes por cobertura"),
                         ("trajetoria_quatro", "Recortes por cobertura"),
                         ("rotacao", "Rodízio de recortes"),
                         ("trajetoria", "Recortes por cobertura")]:
        grupo = novos[novos.metodo == metodo]
        assert len(grupo) == len(FASES)
        n = intervalo(tabela.loc[(tabela.conjunto == "novos") & (tabela.metodo == metodo),
                                 "quadros_usados"])
        campos = [nome, n, intervalo(grupo.vp), intervalo(grupo.fp),
                  *[intervalo(grupo[col], True) for col in ("precisao", "recall", "f1")]]
        linhas.append(" & ".join(campos) + r" \\")
    return "\\newcommand{\\linhasTrajetoria}{%\n" + "\n".join(linhas) + "%\n}\n"


def carregar(video, motor, cache):
    with np.load(cache / f"{video}.npz") as info:
        assert np.array_equal(info["tiles"], tiles_grid(*info["size"]))
        fps = float(info["fps"])
    for clip in load_clips(video, motor, cache=cache):
        yield clip, preparar_grade(clip), fps


def calibrar():
    linhas = []
    for v, video in enumerate(SPLITS["voos"]["val"]):
        for clip, grade, fps in carregar(video, "torchscript_fp32", CACHE):
            for n, peso, fase in product(QUADROS, PESOS_SEGUNDA, FASES):
                indices = agenda(clip["pose"], fps, n, fase)
                mascara, _ = selecionar(clip["pose"], grade, indices, ORCAMENTO_VALIDACAO,
                                        2 * v + clip["clip"] - 1, peso)
                linhas.append(dict(n=n, peso=peso, fase=fase, **pontuar(clip, mascara)))
    resumo = agregar(pd.DataFrame(linhas), ["n", "peso"])
    gravar(resumo, "validacao.csv")
    # A soma sobre as quatro fases serve para escolher a configuração, sem tratar
    # essas repetições como novos trechos independentes.
    melhor = resumo.sort_values(["f1", "recall", "n", "peso"],
                                ascending=[False, False, True, True]).iloc[0]
    return int(melhor.n), float(melhor.peso)


def avaliar(n, peso):
    anterior = pd.read_csv(OUT / "adaptativa/por_trecho.csv").set_index(
        ["conjunto", "video", "trecho", "metodo", "fase"])
    linhas = []
    for v, video in enumerate(VOOS):
        for conjunto, cache in [("iniciais", CACHE), ("novos", TEMP)]:
            for clip, grade, fps in carregar(video, MOTOR, cache):
                chave = (conjunto, video, clip["clip"])
                limite = int(anterior.loc[(*chave, "adaptativa", -1.), "recortes"])
                id_trecho = 2 * v + clip["clip"] - 1
                for fase in FASES:
                    indices = agenda(clip["pose"], fps, n, fase)
                    base = dict(conjunto=conjunto, video=video, trecho=clip["clip"], fase=fase)
                    for metodo, prioridade in [("trajetoria", peso), ("rotacao", None)]:
                        mask, cobertura = selecionar(clip["pose"], grade, indices, limite, id_trecho, prioridade)
                        linhas.append(dict(**base, metodo=metodo, **pontuar(clip, mask), **cobertura))
                    controle = temporal(mask.shape, limite, fase, id_trecho)
                    minimo = int(controle.any(axis=1).sum())
                    for metodo, qtd in [("trajetoria_minima", minimo), ("trajetoria_quatro", 4)]:
                        instantes = agenda(clip["pose"], fps, qtd, fase)
                        reduzida, cobertura = selecionar(clip["pose"], grade, instantes, limite,
                                                         id_trecho, peso)
                        linhas.append(dict(**base, metodo=metodo, **pontuar(clip, reduzida), **cobertura))
                    met = pontuar(clip, controle)
                    for campo in ("vp", "fp", "fn", "recortes"):
                        assert met[campo] == anterior.loc[(*chave, "temporal", fase), campo]
                    linhas.append(dict(**base, metodo="temporal", **met))
        print(video, flush=True)
    tabela = pd.DataFrame(linhas)
    por_caso = tabela.groupby(["conjunto", "video", "trecho", "fase"])
    assert len(por_caso) == 96 and por_caso.size().eq(5).all()
    assert por_caso.recortes.nunique().eq(1).all() and por_caso.referencias.nunique().eq(1).all()
    gravar(tabela, "por_trecho.csv")
    resumo = agregar(tabela, ["conjunto", "metodo", "fase"])
    gravar(resumo, "resumo.csv")
    (DESTINO / "tabela.tex").write_text(tabela_tcc(resumo, tabela), encoding="utf-8")


if __name__ == "__main__":
    verificar_pesos()
    DESTINO.mkdir(exist_ok=True)
    n, peso = calibrar()
    config = dict(quadros_candidatos=QUADROS, pesos_segunda_vista=PESOS_SEGUNDA,
                  quadros_escolhidos=n, peso_escolhido=peso, fases=FASES,
                  escolha="Maior F1 agregado em F/I, depois maior recall, menos quadros e menor peso.",
                  recortes_por_trecho_validacao=ORCAMENTO_VALIDACAO, duracao_s=4,
                  deslocamento_segunda_vista_m=DISTANCIA_SEGUNDA, celula_m=CELL,
                  velocidade_nominal_m_s=VELOCIDADE_NOMINAL, motor_validacao="torchscript_fp32",
                  motor_teste=MOTOR, pesos_sha256=METRICA["pesos_sha256"],
                  voos_validacao=SPLITS["voos"]["val"], voos_teste=VOOS,
                  origem_orcamentos_teste="../adaptativa/por_trecho.csv",
                  sensibilidade="Depois da leitura principal, manter o peso calibrado e usar quatro "
                                "quadros ou a mesma contagem da amostragem temporal em cada trecho.",
                  estado_inicial="Grade vazia no início de cada trecho, nas duas regras de recortes.")
    (DESTINO / "protocolo.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)
                                            + "\n", encoding="utf-8")
    print("Escolha na validação:", n, peso, flush=True)
    avaliar(n, peso)
