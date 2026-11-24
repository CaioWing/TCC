"""Cruza o primeiro alerta de cada trilha com o movimento nos 5 s após sua entrada.

Executar após a avaliação dos voos completos e o balanço de telemetria.
Cada linha do CSV agrupa contagens pareadas, por voo e no conjunto. Uma
trilha pode terminar antes dos 5 s e ainda assim faz parte do denominador.
O segundo CSV identifica todas as discordâncias no prazo do grupo de maior
variação e a direção e velocidade da aeronave na janela correspondente.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simular_potencia_missao import CRUZEIRO, OUT, samples

DESTINO = OUT / "qualidade_memoria/quadrantes"
SEM_JANELA = "sem_janela_valida"


def movimento(inicio, t, sinais):
    duracao = CRUZEIRO.duracao_s
    fim = inicio + duracao
    if fim > t[-1]:
        return {"regime": SEM_JANELA}
    esquerda = t.searchsorted(inicio, side="right") - 1
    direita = t.searchsorted(fim)
    tempos = t[esquerda:direita + 1]
    if np.diff(tempos).max() > CRUZEIRO.lacuna_s:
        return {"regime": SEM_JANELA}
    instantes = np.r_[inicio, tempos[(tempos > inicio) & (tempos < fim)], fim]
    velocidades = np.column_stack([np.interp(instantes, tempos, sinais[eixo][esquerda:direita + 1])
                                   for eixo in ("vx", "vy")])
    media = np.trapezoid(velocidades, instantes, axis=0) / duracao
    maior_desvio = np.linalg.norm(velocidades - media, axis=1).max()
    regime = "maior_variacao" if maior_desvio > CRUZEIRO.horizontal_ms else "menor_variacao"
    modulos = np.linalg.norm(velocidades, axis=1)
    # A direção de deslocamento fica indefinida quando a velocidade é nula.
    angulo = np.nan
    if modulos[0] > 0 and modulos[-1] > 0:
        rumo = np.rad2deg(np.arctan2(velocidades[:, 1], velocidades[:, 0]))
        angulo = abs((rumo[-1] - rumo[0] + 180) % 360 - 180)
    return dict(regime=regime, desvio_horizontal_ms=maior_desvio,
                diferenca_direcao_graus=angulo, velocidade_inicial_ms=modulos[0],
                velocidade_final_ms=modulos[-1], velocidade_minima_amostrada_ms=modulos.min(),
                velocidade_maxima_amostrada_ms=modulos.max())


def resumo(pares, grupo):
    linha = dict(trilhas=len(pares))
    if grupo != SEM_JANELA:
        c = pares.atraso_s_quadrantes.le(CRUZEIRO.duracao_s)
        i = pares.atraso_s_intervalo_pareado.le(CRUZEIRO.duracao_s)
        linha.update(ambas=int((c & i).sum()), somente_quadrantes=int((c & ~i).sum()),
                     somente_intervalo=int((~c & i).sum()), nenhuma=int((~c & ~i).sum()))
    return linha


def main():
    metodos = {"quadrantes", "intervalo_pareado"}
    alertas = pd.read_csv(DESTINO / "por_trilha.csv").loc[lambda df: df.metodo.isin(metodos)]
    voos = pd.read_csv(DESTINO / "por_voo.csv").query("metodo == 'quadrantes'").set_index("video")
    balanco = pd.read_csv(OUT / "balanco_telemetria.csv").set_index("voo")
    chaves = ["video", "trilha", "classe"]
    assert set(alertas.metodo) == metodos
    assert alertas.groupby(chaves).metodo.nunique().eq(2).all()
    assert alertas.groupby(chaves).quadro_entrada.nunique().eq(1).all()
    pares = alertas.pivot(index=chaves + ["quadro_entrada"], columns="metodo",
                          values=["atraso_s", "primeiro_alerta_quadro"])
    pares.columns = ["_".join(par) for par in pares.columns]
    grupos = []
    for video, grupo in pares.reset_index().groupby("video"):
        voo = balanco.loc[int(video[5:7])]
        t, sinais = samples(voo.name, voo.regiao, voo.trecho_s)
        fps = voos.loc[video, "fps"]
        janela = pd.DataFrame([dict(quadro_entrada=quadro, **movimento(quadro / fps, t, sinais))
                               for quadro in grupo.quadro_entrada.unique()])
        grupos.append(grupo.merge(janela, on="quadro_entrada", validate="many_to_one"))
    trilhas = pd.concat(grupos, ignore_index=True)
    resultados = []
    for video, grupo_voo in [*trilhas.groupby("video"), ("todos", trilhas)]:
        for nome, grupo in grupo_voo.groupby("regime"):
            resultados.append(dict(video=video, regime=nome, **resumo(grupo, nome)))
    pd.DataFrame(resultados).convert_dtypes().to_csv(DESTINO / "alertas_por_movimento.csv", index=False)
    maior = trilhas[trilhas.regime.eq("maior_variacao")]
    c = maior.atraso_s_quadrantes.le(CRUZEIRO.duracao_s)
    i = maior.atraso_s_intervalo_pareado.le(CRUZEIRO.duracao_s)
    maior[c ^ i].drop(columns="regime").sort_values(["video", "quadro_entrada"]).convert_dtypes().to_csv(
        DESTINO / "alertas_discordantes_maior_variacao.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
