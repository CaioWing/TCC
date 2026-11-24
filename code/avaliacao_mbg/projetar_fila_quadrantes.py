"""Reproduz na fila do detector os envios dos quadrantes e do intervalo pareado."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "doc/dados/qualidade_memoria/quadrantes"
PLANOS = ROOT / "tmp/avaliacao_mbg/quadrantes_v1"
METODOS = ("quadrantes", "intervalo_pareado")


def simular(quadros, tamanhos, fps, tempo_recorte_ms):
    chegada = quadros * 1000. / fps
    inicios = np.empty(len(quadros))
    fins = np.empty(len(quadros))
    anterior = 0.
    for i, (agora, tamanho) in enumerate(zip(chegada, tamanhos)):
        inicios[i] = max(agora, anterior)
        fins[i] = inicios[i] + tamanho * tempo_recorte_ms
        anterior = fins[i]
    return pd.DataFrame(dict(quadro=quadros, recortes=tamanhos, chegada_ms=chegada,
                             inicio_ms=inicios, fim_ms=fins, espera_ms=inicios-chegada,
                             resposta_ms=fins-chegada))


def resumir(lotes, video, metodo, cenario, tempo_recorte_ms, duracao_s):
    return dict(video=video, metodo=metodo, cenario=cenario,
                tempo_recorte_ms=tempo_recorte_ms, duracao_s=duracao_s,
                recortes=lotes.recortes.sum(), lotes=len(lotes),
                lotes_unitarios=lotes.recortes.eq(1).sum(),
                lotes_com_espera=lotes.espera_ms.gt(1.e-6).sum(),
                espera_max_ms=lotes.espera_ms.max(),
                resposta_mediana_ms=lotes.resposta_ms.median(),
                resposta_p99_ms=lotes.resposta_ms.quantile(.99),
                resposta_max_ms=lotes.resposta_ms.max(),
                folga_final_ms=duracao_s * 1000. - lotes.fim_ms.iloc[-1])


def main():
    ensaios = pd.read_csv(ROOT / "data/yolo11n-jetson24-20261116/summary.csv")
    fp16, = ensaios.query("config == 'trt_fp16' and mode == 'e2e'").itertuples()
    cenarios = {"media": fp16.lat_mean_ms, "p99": fp16.lat_p99_ms}
    avaliacao = pd.read_csv(RESULTADOS / "por_voo.csv").set_index(["video", "metodo"])
    voos = sorted(avaliacao.index.get_level_values("video").unique())
    assert len(voos) == 6
    resultados, exemplo = [], []
    for video in voos:
        with np.load(PLANOS / f"{video}.npz") as plano:
            frames = plano["frames"]
            contagens = plano["masks"].sum(axis=1)
            fps, n = float(plano["fps"]), int(plano["total_frames"])
        np.testing.assert_array_equal(frames, np.unique(frames))
        np.testing.assert_equal(contagens.sum(axis=0)[0], contagens.sum(axis=0)[1])
        for j, metodo in enumerate(METODOS):
            counts = contagens[:, j]
            usados = counts > 0
            ref = avaliacao.loc[(video, metodo)]
            assert counts.sum() == ref.recortes and usados.sum() == ref.quadros_enviados
            assert n == ref.quadros_voo and np.isclose(n / fps, ref.duracao_s)
            for cenario, latencia in cenarios.items():
                lotes = simular(frames[usados], counts[usados], fps, latencia)
                resultados.append(resumir(lotes, video, metodo, cenario, latencia, n / fps))
                if video == voos[0] and cenario == "media":
                    primeiros = lotes[lotes.chegada_ms < 6000].copy()
                    primeiros.insert(0, "metodo", metodo)
                    primeiros.insert(0, "video", video)
                    exemplo.append(primeiros)
    pd.DataFrame(resultados).to_csv(RESULTADOS / "fila_jetson_por_voo.csv", index=False,
                                    float_format="%.8f")
    pd.concat(exemplo).to_csv(RESULTADOS / "fila_jetson_inicio_voo09.csv", index=False,
                              float_format="%.8f")


if __name__ == "__main__":
    main()
