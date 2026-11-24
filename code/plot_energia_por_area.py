"""Energia por área nos seis voos completos, para as massas e a cota do ensaio."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from balanco_telemetria import JETSON, ROOT
from paleta_tcc import GREEN


def cenarios(config="trt_fp16"):
    dados = ROOT / "doc/dados"
    selecao = pd.read_csv(dados / "qualidade_memoria/quadrantes/por_voo.csv")
    selecao = selecao[selecao.metodo.isin(["quadrantes", "intervalo_pareado"])].copy()
    selecao["voo"] = selecao.video.str.extract(r"video(\d+)_", expand=False).astype(int)
    cotas = selecao.pivot(index="voo", columns="metodo", values="recortes")
    if len(cotas) != 6 or not cotas.quadrantes.equals(cotas.intervalo_pareado) or cotas.isna().any().any():
        raise ValueError("São esperados seis voos com a mesma cota por estratégia.")

    balanco = pd.read_csv(dados / "balanco_telemetria.csv").set_index("voo").loc[cotas.index]
    duracao = selecao.loc[selecao.metodo == "quadrantes"].set_index("voo").loc[cotas.index, "duracao_s"]
    np.testing.assert_allclose(balanco.trecho_s, duracao, rtol=0, atol=1e-6)
    bancada = pd.read_csv(JETSON / "summary.csv").query("config == @config and mode == 'e2e'")
    energia_mJ = bancada.E_dyn_mJ_per_inf.item()
    dinamica = cotas.quadrantes * energia_mJ / (3.6e6 * balanco.area_sim_km2)

    # Massa e repouso já estão integrados sobre o percurso em simular_potencia_missao.py.
    campos = ["voo", "carga_g", "bateria_Wh_km2", "massa_Wh_km2", "ocioso_Wh_km2"]
    cen = pd.read_csv(dados / "massa_inferencia_por_voo.csv", usecols=campos)
    cen = cen.merge(dinamica.rename("inferencia_Wh_km2"), on="voo", validate="many_to_one")
    cen["desligada"] = cen.bateria_Wh_km2 + cen.massa_Wh_km2
    cen["repouso"] = cen.desligada + cen.ocioso_Wh_km2
    cen["ativa"] = cen.repouso + cen.inferencia_Wh_km2
    return cen


def figura(cen):
    fig, voos = plt.subplots(figsize=(5.65, 2.32))
    fig.subplots_adjust(left=.11, right=.98, top=.83, bottom=.22)
    base = cen.loc[cen.carga_g == 174].sort_values("voo")
    pos = np.arange(len(base))
    voos.hlines(pos, base.bateria_Wh_km2, base.ativa, color=".69", linewidth=1.4)
    voos.scatter(base.bateria_Wh_km2, pos, s=28, facecolor="white", edgecolor=".20",
                 linewidth=1.2, label="Sem kit (medição)", zorder=3)
    voos.scatter(base.ativa, pos, s=28, marker="s", color=GREEN,
                 label="Com kit de 174 g (estimativa)", zorder=3)
    voos.set_yticks(pos, [f"Voo {n:02d}" for n in base.voo])
    voos.set(xlim=(218, 302), ylim=(5.55, -.55), xlabel="Energia total por área imageada (Wh/km²)",
             xticks=range(220, 301, 20))
    voos.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=2, borderaxespad=0,
                columnspacing=1.05, handletextpad=.35)
    voos.tick_params(axis="y", length=0)
    voos.grid(axis="x", color=".9", linewidth=.65)
    return fig


if __name__ == "__main__":
    resultados = cenarios()
    print(resultados.groupby("carga_g")[["bateria_Wh_km2", "desligada", "repouso", "ativa"]]
          .median().round(2).to_string())
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.3, "axes.titlesize": 9.2,
                         "legend.fontsize": 7.6, "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "pdf.fonttype": 42}):
        fig = figura(resultados)
        fig.savefig(ROOT / "doc/figuras/fig_energia_por_area.pdf", bbox_inches="tight", pad_inches=.05)
        plt.close(fig)
