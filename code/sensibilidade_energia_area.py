"""Energia por área ao variar a altura nos seis percursos ou a sobreposição entre faixas."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

from balanco_telemetria import DATA, JETSON
from memoria_sim import CELL, grade_projecao
from paleta_tcc import BLUE
from plot_ajuste_gsd import fit_gsd, load_tracks
from plot_energia_por_area import cenarios
from simular_potencia_missao import BASE_KG, KIT_KG, FIG, OUT, eficiencia_ajustada, mechanical

ALTURA_BASE = 40
ALTURAS = np.arange(20, 81, 10)


def area_do_percurso(log, n, fps, largura, altura, gsd):
    dimensoes, raster = grade_projecao(log, n, fps, largura, altura, gsd)
    terreno = np.zeros(dimensoes, dtype=bool)
    for quadro in range(n):
        j0, j1, i0, i1, dentro, _, _ = raster(quadro)
        terreno[j0:j1, i0:i1] |= dentro
    return terreno.sum() * CELL**2 / 1e6


def percursos():
    ativos = cenarios().query("carga_g == 174").set_index("voo")
    balanco = pd.read_csv(OUT / "balanco_telemetria.csv").set_index("voo")
    voos = pd.read_csv(OUT / "qualidade_memoria/quadrantes/por_voo.csv")
    classes = json.loads((DATA / "splits/splits.json").read_text(encoding="utf-8"))["classes_avaliadas"]
    objetos = pd.read_csv(DATA / "objetos_por_trilha.csv")
    linhas = []
    for voo in voos.query("metodo == 'quadrantes'").itertuples():
        numero = int(voo.video[5:7])
        referencia = balanco.loc[numero]
        pasta = DATA / "raw" / f"flight-mbg-v2-{numero:02d}-region{referencia.regiao}"
        log, = (pasta / "log").glob("*.csv")
        trilhas, meta = load_tracks(flight=pasta)
        _, _, escalas, _, melhor = fit_gsd(trilhas)
        gsd_base = escalas[melhor] / 100
        lados = objetos.loc[objetos.video.eq(voo.video) & objetos.label.isin(classes), "side"]
        assert len(lados) == voo.referencias and lados.gt(0).all()
        energia = ativos.loc[numero, "ativa"] * referencia.area_sim_km2
        for h in ALTURAS:
            gsd = gsd_base * h / ALTURA_BASE
            area = area_do_percurso(log, voo.quadros_voo, meta["fps"], meta["width"], meta["height"], gsd)
            if h == ALTURA_BASE:
                np.testing.assert_allclose(area, referencia.area_sim_km2, atol=CELL**2/1e6, rtol=0)
            linhas.append(dict(voo=numero, altura_m=h, gsd_cm_px=100*gsd, largura_faixa_m=meta["width"]*gsd,
                               area_km2=area, energia_Wh=energia, energia_Wh_km2=energia/area,
                               trilhas=len(lados), trilhas_abaixo_32px=int((lados*ALTURA_BASE/h < 32).sum())))
    return pd.DataFrame(linhas)


def faixas_paralelas(dados):
    velocidade = 4.1
    eta = eficiencia_ajustada(pd.read_csv(OUT / "curva_potencia_por_voo.csv"))
    repouso = json.loads((JETSON / "results.json").read_text(encoding="utf-8"))["P_idle_W"]
    bancada = pd.read_csv(JETSON / "summary.csv").query("config == 'trt_fp16' and mode == 'e2e'").iloc[0]
    envios = pd.read_csv(OUT / "qualidade_memoria/quadrantes/por_voo.csv").query("metodo == 'quadrantes'")
    taxa = envios.recortes.sum()/envios.duracao_s.sum()
    potencia = mechanical(velocidade, BASE_KG + KIT_KG)/eta + repouso + taxa*bancada.E_dyn_mJ_per_inf/1000
    largura_base = dados.loc[dados.altura_m == ALTURA_BASE, "largura_faixa_m"].median()
    linhas = []
    for h in [20, 30, 40, 60, 80]:
        largura = largura_base*h/ALTURA_BASE
        for sobreposicao in [0, .25, .5]:
            distancia = largura*(1-sobreposicao)
            energia = potencia*1e6/(3600*velocidade*distancia)
            linhas.append(dict(altura_m=h, sobreposicao_pct=100*sobreposicao, largura_faixa_m=largura,
                               espacamento_m=distancia, velocidade_ms=velocidade, potencia_W=potencia,
                               energia_Wh_km2=energia))
    return pd.DataFrame(linhas)


def figura_altura(dados):
    fig, ax = plt.subplots(figsize=(5.5, 2.5), layout="constrained")
    energia = dados.groupby("altura_m").energia_Wh_km2
    mediana = energia.median()
    ax.fill_between(mediana.index, energia.min(), energia.max(), color=BLUE, alpha=.18, lw=0,
                    label="Mínimo a máximo dos seis voos")
    ax.plot(mediana.index, mediana, "o-", ms=4, lw=1.6, color=BLUE, label="Mediana")
    ax.set(xlabel="Altura sobre o terreno (m)", ylabel="Energia por área (Wh/km²)",
           xticks=mediana.index)
    ax.axvline(ALTURA_BASE, color=".45", ls=":", lw=.9, zorder=0)
    ax.grid(axis="y", color=".9", lw=.6)
    ax.legend(frameon=False, fontsize=8)
    return fig


def figura_faixas(dados):
    tabela = dados.pivot(index="sobreposicao_pct", columns="altura_m", values="energia_Wh_km2")
    fig, ax = plt.subplots(figsize=(5.6, 2.08), layout="constrained")
    gradiente = LinearSegmentedColormap.from_list("energia", ["#f1f5f7", "#93b1c0", "#294e67"])
    valores = tabela.to_numpy()
    escala = Normalize(valores.min(), valores.max())
    ax.imshow(valores, cmap=gradiente, norm=escala, aspect="auto")
    for (i, j), valor in np.ndenumerate(valores):
        ax.text(j, i, f"{valor:.0f}", va="center", ha="center", fontsize=9.6,
                color="white" if escala(valor) > .65 else "#182a35")
    ax.set(xticks=range(len(tabela.columns)), xticklabels=tabela.columns, xlabel="Altura sobre o terreno (m)",
           yticks=range(len(tabela)), yticklabels=[f"{p:.0f}%" for p in tabela.index],
           ylabel="Sobreposição lateral", title="Energia por área em faixas paralelas (Wh/km²)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return fig


if __name__ == "__main__":
    rotas = percursos()
    faixas = faixas_paralelas(rotas)
    rotas.to_csv(OUT / "energia_altura_percursos.csv", index=False, float_format="%.8f")
    faixas.to_csv(OUT / "energia_altura_sobreposicao.csv", index=False, float_format="%.8f")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
                         "axes.titlepad": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "pdf.fonttype": 42}):
        for nome, fig in [("fig_altura_percursos", figura_altura(rotas)),
                          ("fig_altura_sobreposicao", figura_faixas(faixas))]:
            fig.savefig(FIG / f"{nome}.pdf", bbox_inches="tight", pad_inches=.055)
            plt.close(fig)
