"""Reproduz a análise de sensibilidade da Figura 16 a partir da telemetria.

Na raiz do TCC, executar este arquivo e depois code/figuras_selecao_cruzeiro.py.
Cada cenário altera um critério, mantendo a mesma divisão de voos da curva principal.
"""
from dataclasses import asdict, replace
import json

import numpy as np
import pandas as pd

from simular_potencia_missao import (BASE_KG, CRUZEIRO, DATA, OUT, eficiencia_ajustada,
                                    estatisticas_janelas, mechanical, samples)


ALTERNATIVAS = {
    "duracao_s": (3, 7),
    "lacuna_s": (0.5, 1.5),
    "altitude_m": (1, 3),
    "vertical_ms": (0.2, 0.4),
    "horizontal_ms": (0.3, 0.7),
    "campanha_ms": (0.4, 0.8),
}


def classificar_janelas(janelas, regiao):
    """Preserva todos os candidatos e identifica os aceitos para os boxplots."""
    na_faixa = CRUZEIRO.na_faixa_da_campanha(janelas.velocidade_ms, regiao)
    sem_limite_horizontal = replace(CRUZEIRO, horizontal_ms=np.inf)
    return janelas.assign(
        demais_criterios=(sem_limite_horizontal.regulares(janelas) & na_faixa).astype(int),
        aceita=(CRUZEIRO.regulares(janelas) & na_faixa).astype(int),
    )


def main():
    balanco = pd.read_csv(OUT / "balanco_telemetria.csv")
    regioes = json.loads((DATA / "splits/splits.json").read_text(encoding="utf-8"))["regioes"]
    cenarios = [("referencia", None, CRUZEIRO)]
    for criterio, valores in ALTERNATIVAS.items():
        for valor in valores:
            cenarios.append((criterio, valor, replace(CRUZEIRO, **{criterio: valor})))

    duracoes = sorted({limites.duracao_s for _, _, limites in cenarios})
    por_voo = []
    janelas_por_voo = []
    for voo in balanco.itertuples():
        tempos, sinais = samples(voo.voo, voo.regiao, voo.trecho_s)
        janelas = {d: estatisticas_janelas(tempos, sinais, d) for d in duracoes}
        classificadas = classificar_janelas(janelas[CRUZEIRO.duracao_s], voo.regiao)
        classificadas.insert(0, "regiao", voo.regiao)
        classificadas.insert(0, "voo", voo.voo)
        janelas_por_voo.append(classificadas)
        grupo = "ajuste" if voo.regiao in regioes["train"] else "comparacao"
        for cenario, (_, _, limites) in enumerate(cenarios):
            candidatas = janelas[limites.duracao_s]
            aceitas = candidatas.loc[limites.regulares(candidatas)
                                     & limites.na_faixa_da_campanha(candidatas.velocidade_ms, voo.regiao)]
            if aceitas.empty:
                raise ValueError(f"Sem janelas para comparar o voo {voo.voo} no cenário {limites}")
            velocidade, potencia = aceitas[["velocidade_ms", "potencia_W"]].median()
            por_voo.append(dict(cenario=cenario, voo=voo.voo, n_janelas=len(aceitas),
                                grupo="outra_campanha" if voo.regiao == "A" else grupo,
                                bateria_W=potencia, mecanica_W=mechanical(velocidade, BASE_KG)))

    pd.concat(janelas_por_voo, ignore_index=True).to_csv(
        OUT / "janelas_cruzeiro_por_voo.csv", index=False, float_format="%.17g")

    resultados = []
    for cenario, voos in pd.DataFrame(por_voo).set_index("voo").groupby("cenario"):
        criterio, valor, limites = cenarios[cenario]
        eta = eficiencia_ajustada(voos)
        erros = voos.mecanica_W / eta - voos.bateria_W
        erros_comparacao = erros[voos.grupo == "comparacao"]
        n_janelas = voos.n_janelas.sum()
        resultados.append(dict(criterio=criterio, valor=valor, **asdict(limites),
                               n_janelas=n_janelas, tempo_min=n_janelas * limites.duracao_s / 60,
                               n_voos=len(voos), eta=eta, mae_comparacao_W=erros_comparacao.abs().mean(),
                               erro_voo01_W=erros.loc[1], erro_voo02_W=erros.loc[2]))
    pd.DataFrame(resultados).to_csv(OUT / "sensibilidade_cruzeiro.csv", index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
