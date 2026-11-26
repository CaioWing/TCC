"""Potência adicional em voo com uma Jetson nas três estratégias de inferência."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avaliacao_mbg.projetar_fila_quadrantes import METODOS, PLANOS, simular
from paleta_tcc import BLUE, GRAY
from plot_bancada_motores import DADOS as BANCADA, ROOT


DADOS = ROOT / "doc/dados"
TABELA = DADOS / "potencia_estrategias_motores.csv"
ESTRATEGIAS = dict(quadrantes="Quadrantes", intervalo_pareado="Intervalo",
                   integral="Integral (saturada)")
MOTORES = {
    "trt_fp32": "TensorRT FP32", "trt_fp16": "TensorRT FP16", "trt_int8": "TensorRT INT8",
    "ort_trt_fp32": "ORT · TensorRT FP32", "ort_trt_fp16": "ORT · TensorRT FP16",
    "ort_cuda_fp32": "ORT · CUDA FP32", "ort_cpu_fp32": "ORT · CPU FP32",
    "torchscript_fp32": "PyTorch FP32", "torchscript_fp16": "PyTorch FP16",
}


def planos_voos():
    pasta = DADOS / "qualidade_memoria"
    selecao = pd.read_csv(pasta / "quadrantes/por_voo.csv").set_index(["video", "metodo"])
    integral = pd.read_csv(pasta / "altitude_integral/por_voo.csv").query("metodo == 'integral'")
    voos = []
    for ref in integral.itertuples():
        with np.load(PLANOS / f"{ref.video}.npz") as plano:
            fps = float(plano["fps"])
            n = int(plano["total_frames"])
            quadros = plano["frames"]
            contagens = plano["masks"].sum(axis=1)
            tiles = plano["masks"].shape[1]
        assert ref.quadros_voo == n and ref.recortes == n * tiles
        envios = {"integral": (np.arange(n), np.full(n, tiles))}
        for j, metodo in enumerate(METODOS):
            quant = contagens[:, j]
            usados = quant > 0
            publicado = selecao.loc[(ref.video, metodo)]
            assert quant.sum() == publicado.recortes and usados.sum() == publicado.quadros_enviados
            assert np.isclose(n / fps, publicado.duracao_s, rtol=0, atol=1e-6)
            envios[metodo] = quadros[usados], quant[usados]
        voos.append((int(ref.video[5:7]), fps, n / fps, envios))
    assert len({v[0] for v in voos}) == len(voos) == 6
    return voos


def recalcular():
    voos = planos_voos()
    ids = [v[0] for v in voos]
    total_s = sum(v[2] for v in voos)
    balanco = pd.read_csv(DADOS / "balanco_telemetria.csv").set_index("voo").loc[ids]
    np.testing.assert_allclose(balanco.trecho_s, [v[2] for v in voos], rtol=0, atol=1e-6)
    massa = pd.read_csv(DADOS / "massa_inferencia_por_voo.csv").query("carga_g == 174")
    massa = massa.set_index("voo").loc[ids]
    transporte_w = (massa.massa_Wh_km2 * balanco.area_sim_km2).sum() * 3600 / total_s
    repouso_w = (massa.ocioso_Wh_km2 * balanco.area_sim_km2).sum() * 3600 / total_s

    ensaios = pd.read_csv(BANCADA / "summary.csv").query("mode == 'e2e'").set_index("config")
    linhas = []
    for config, ensaio in ensaios.loc[list(MOTORES)].iterrows():
        for metodo in ESTRATEGIAS:
            ocupado_s = 0.
            solicitados = concluidos = 0
            for _, fps, duracao, envios in voos:
                quadros, tamanhos = envios[metodo]
                lotes = simular(quadros, tamanhos, fps, 1000 / ensaio.fps)
                # Só atribui potência a bordo ao trabalho executado até o encerramento do vídeo.
                trabalho_ms = (np.minimum(lotes.fim_ms, 1000 * duracao) - lotes.inicio_ms).clip(lower=0)
                ocupado_s += trabalho_ms.sum() / 1000
                concluidos += int(np.floor(trabalho_ms * ensaio.fps / 1000 + 1e-8).sum())
                solicitados += int(tamanhos.sum())
            ocupacao = ocupado_s / total_s
            assert ocupacao <= 1 + 1e-9 and concluidos <= solicitados
            dinamica_w = ocupacao * ensaio.P_dyn_W
            linhas.append(dict(metodo=metodo, config=config, voos=len(voos), duracao_s=total_s,
                               solicitados=solicitados, concluidos_em_voo=concluidos,
                               conclusao_pct=100 * concluidos / solicitados, ocupacao_pct=100 * ocupacao,
                               transporte_W=transporte_w, repouso_W=repouso_w, inferencia_W=dinamica_w,
                               acrescimo_total_W=transporte_w + repouso_w + dinamica_w))
    return pd.DataFrame(linhas)


def figura(dados):
    totais = dados.pivot(index="config", columns="metodo", values="acrescimo_total_W")
    paineis = ESTRATEGIAS
    if totais.quadrantes.round(1).equals(totais.intervalo_pareado.round(1)):
        paineis = {"quadrantes": "Quadrantes e intervalo", "integral": ESTRATEGIAS["integral"]}
    fig, eixos = plt.subplots(1, len(paineis), sharey=True, figsize=(7.35, 3.17))
    fig.subplots_adjust(left=.215, right=.987, top=.87, bottom=.25, wspace=.14)
    y = np.array([0, 1, 2, 3.45, 4.45, 5.45, 6.45, 7.9, 8.9])
    for ax, (metodo, titulo) in zip(eixos, paineis.items()):
        medidas = dados.loc[dados.metodo == metodo].set_index("config").loc[list(MOTORES)]
        comum = medidas.transporte_W + medidas.repouso_W
        ax.barh(y, comum, height=.65, color=GRAY, label="Transporte do kit e placa em repouso")
        ax.barh(y, medidas.inferencia_W, left=comum, height=.65, color=BLUE, label="Inferência")
        for pos, total in zip(y, medidas.acrescimo_total_W):
            ax.text(total + .55, pos, f"{total:.1f}".replace(".", ","), va="center", fontsize=7.6)
        ax.set(xlim=(0, 38), xticks=[0, 10, 20, 30], ylim=(9.6, -.75))
        ax.set_title(titulo, fontsize=9.5, fontweight="semibold", pad=6)
        ax.grid(axis="x", color=".9", linewidth=.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    eixos[0].set_yticks(y, MOTORES.values(), fontsize=8.1)
    fig.text(.60, .115, "Acréscimo estimado de potência (W)", ha="center")
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               handlelength=1.35, columnspacing=2, bbox_to_anchor=(.58, 0))
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recalcular", action="store_true", help="Reproduz os envios dos planos de voo.")
    if parser.parse_args().recalcular or not TABELA.exists():
        recalcular().to_csv(TABELA, index=False, float_format="%.8f")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.axisbelow": True,
                         "pdf.fonttype": 42}):
        fig = figura(pd.read_csv(TABELA))
        fig.savefig(ROOT / "doc/figuras/fig_potencia_estrategias_motores.pdf",
                    bbox_inches="tight", pad_inches=.055)
        plt.close(fig)
