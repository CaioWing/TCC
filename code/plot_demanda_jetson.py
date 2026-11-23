"""Compara a demanda por voo com a bancada TensorRT FP16 nas duas coortes."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd

from paleta_tcc import BLUE, GREEN, ORANGE, RED

ROOT = Path(__file__).resolve().parents[1]


def intervalo(taxas):
    return np.percentile(taxas, [0, 50, 100])


def ponto(ax, y, taxas, cor):
    minimo, mediana, maximo = taxas
    ax.plot([minimo, maximo], [y, y], color=cor, lw=2.2, solid_capstyle="round", zorder=2)
    ax.scatter(mediana, y, s=25, color=cor, edgecolor="#343b40", linewidth=.55, zorder=3)


def painel_fila(ax, base):
    fila = pd.read_csv(base / "cobertura_continua/fila_jetson_projetada.csv")
    media = fila[fila.cenario == "media"]
    tempo_ms, = media.tempo_recorte_ms.unique()
    quadros = pd.read_csv(base / "cobertura_continua/por_voo.csv")
    fps, = quadros.fps.unique()
    for y, (metodo, passo, cor) in enumerate([("cobertura", 20, BLUE), ("intervalo", 30, ORANGE)]):
        grupo = media[media.metodo == metodo]
        assert len(grupo) == 6 and grupo.quadros_com_espera.eq(0).all()
        recortes, = grupo.max_recortes_quadro.unique()
        intervalo = passo / fps
        ocupado = recortes * tempo_ms / 1000
        folga = intervalo - ocupado
        np.testing.assert_allclose(grupo.margem_min_ate_proximo_lote_ms, folga*1000)
        for quadro in range(0, 60, passo):
            inicio = quadro/fps
            ax.barh(y, ocupado, left=inicio, color=cor, height=.64)
            ax.barh(y, folga, left=inicio+ocupado, color="#e9e9e9", height=.64)
        ax.text(ocupado/2, y, f"{ocupado*1000:.1f} ms".replace(".", ","), color="white", ha="center",
                va="center", weight="semibold", fontsize=7.4)
        ax.text(ocupado+folga/2, y, f"{folga*1000:.1f} ms".replace(".", ","), color=".20", ha="center",
                va="center", fontsize=7.4)
    ax.set_yticks([0, 1], ["Cobertura · 16 recortes", "Intervalo · 24 recortes"])
    ax.set(xlim=(0,60/fps), ylim=(1.55,-.51), xlabel="Tempo desde o início do bloco (s)")
    ax.set_xticks([0,.5,1,1.5,2], ["0","0,5","1","1,5","2"])
    ax.tick_params(axis="y", length=0, labelsize=7.6)
    ax.tick_params(axis="x", length=2.5, labelsize=7.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)


def figura_demanda():
    base = ROOT / "doc/dados/qualidade_memoria"
    campanha = pd.read_csv(ROOT / "data/mbg_v2/splits/memoria_sim_26voos.csv")
    bancada = pd.read_csv(ROOT / "data/yolo11n-jetson24-20261116/summary.csv")
    fp16, = bancada.query("config == 'trt_fp16' and mode == 'e2e'").itertuples()
    joules = fp16.E_dyn_mJ_per_inf / 1000
    bm = campanha[campanha.region != "A"]
    a = campanha[campanha.region == "A"]
    assert len(bm) == 24 and len(a) == 2
    linhas_campanha = [
        ("Saída da célula (B–M)", intervalo(bm.adiada_recortes_s), BLUE),
        ("Saída da célula (A)", intervalo(a.adiada_recortes_s), BLUE),
        ("Um quadro a cada 30 (B–M)", np.full(3, 24), ORANGE),
        ("Entrada da célula (B–M)", intervalo(bm.imediata_recortes_s), GREEN),
        ("Integral (B–M)", np.full(3, 719), RED),
    ]

    passagens = pd.read_csv(base / "passagens/geometria.csv")
    faixas = pd.read_csv(base / "multivistas/custo_e_cobertura_por_voo.csv")
    voos = set(passagens.video)
    assert voos == set(faixas.video) and len(voos) == 6
    np.testing.assert_allclose(passagens.recortes_s, passagens.recortes_unicos * passagens.fps / passagens.quadros_voo)
    np.testing.assert_allclose(faixas.taxa_geo, faixas.recortes_geo * faixas.fps / faixas.quadros)
    np.testing.assert_allclose(faixas.potencia_incremental_fp16_w, faixas.taxa_geo * joules)
    linhas_teste = []
    for nome, plano in [(r"$k_{\mathrm{int}}=0$", "extremos"), (r"$k_{\mathrm{int}}=1$", "tres"),
                        ("Só entrada", "entrada"), ("Só meio", "meio")]:
        dados = passagens[passagens.plano == plano].set_index("video")
        assert dados.index.is_unique and set(dados.index) == voos
        linhas_teste.append((nome, intervalo(dados.recortes_s), BLUE))
    for k, nome in [(1, "Uma faixa"), (2, "Duas faixas")]:
        dados = faixas[faixas.k == k].set_index("video")
        assert dados.index.is_unique and set(dados.index) == voos
        linhas_teste.append((nome, intervalo(dados.taxa_geo), GREEN))

    pt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig = plt.figure(figsize=(7.32, 4.48))
        ax = fig.add_axes([.202, .455, .256, .415])
        teste = fig.add_axes([.713, .455, .272, .415])
        fig.text(.002, .995, "(a) Campanha completa · 26 voos", size=9.3, va="top")
        fig.text(.503, .995, "(b) Teste · seis voos de E, H e L", size=9.3, va="top")
        fig.text(.002, .291, "(c) Fila estimada em um bloco de 60 quadros", size=9.3, va="top")
        fig.text(.984, .286, "Processamento em cor · folga em cinza", ha="right",
                 va="top", fontsize=7.5, color=".24")
        painel_fila(fig.add_axes([.202, .105, .783, .151]), base)
        for y, (_, taxas, cor) in enumerate(linhas_campanha):
            ponto(ax, y, taxas / fp16.fps, cor)
        ax.set_yticks(np.arange(5), [nome for nome, _, _ in linhas_campanha])
        ax.set_ylim(4.45, -.45)
        ax.set_xscale("log")
        ax.set_xlim(.09, 25)
        ax.axvline(1, color="#343b40", lw=1, ls="--")
        ax.set_xlabel("Demanda / vazão FP16 (vezes)", fontsize=8.0)
        ax.xaxis.set_major_locator(FixedLocator([.1, .2, .5, 1, 2, 5, 10, 20]))
        ax.xaxis.set_major_formatter(pt)
        ax.xaxis.set_minor_locator(NullLocator())

        alturas = [0, 1, 3.2, 4.2, 6.4, 7.4]
        for y, (_, taxas, cor) in zip(alturas, linhas_teste):
            ponto(teste, y, taxas, cor)
        teste.set_yticks(alturas, [nome for nome, _, _ in linhas_teste])
        teste.set(xlim=(0, 16), ylim=(8.0, -1.4))
        teste.set_xticks([0, 4, 8, 12, 16])
        teste.set_xlabel("Demanda (recortes/s)", fontsize=8.1)
        for y, texto in [(-.98, "Geográfica por passagem"), (2.25, "Controles de uma vista"),
                         (5.45, "Saída de cada faixa")]:
            teste.text(-.76, y, texto, transform=teste.get_yaxis_transform(), va="center",
                       fontsize=7.8, weight="semibold")
        watts = teste.secondary_xaxis("top", functions=(lambda taxa: taxa * joules, lambda w: w / joules))
        watts.set_xlabel("Potência acima do repouso (W)", fontsize=8.0, labelpad=3)
        watts.set_xticks([0, .2, .4, .6])
        watts.xaxis.set_major_formatter(pt)
        watts.tick_params(labelsize=7, length=2.5, pad=1)
        for eixo in (ax, teste):
            eixo.tick_params(axis="both", labelsize=7.4, length=2.5)
            eixo.tick_params(axis="y", length=0, pad=3)
            eixo.grid(axis="x", alpha=.21)
            eixo.set_axisbelow(True)
    return fig


if __name__ == "__main__":
    with plt.rc_context({"pdf.fonttype": 42}):
        figura = figura_demanda()
        figura.savefig(ROOT / "doc/figuras/fig_jetson_demanda.pdf", bbox_inches="tight", pad_inches=.055)
        plt.close(figura)
