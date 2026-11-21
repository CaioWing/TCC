"""Potência equivalente para atender toda a demanda, com a capacidade medida em bancada."""
import pandas as pd

from paleta_tcc import BLUE, GREEN, ORANGE


FORMATS = [("fp32", BLUE, "o"), ("fp16", ORANGE, "s"), ("int8", GREEN, "^")]


def theoretical_board_power(rates, benchmark, idle_w):
    """P = P_ociosa + r E_inf, com r em recortes/s e E_inf incremental em mJ."""
    return pd.DataFrame({
        name: idle_w + rates * (run.E_dyn_mJ_per_inf / 1000)
        for name, run in benchmark.iterrows()
    })
