"""Reproduz as curvas, os erros por voo e os cenários de carga do capítulo 4.

Executar após balanco_telemetria.py. Requer NumPy, pandas e Matplotlib.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from balanco_telemetria import DATA, JETSON, ROOT
from paleta_tcc import BLUE, GREEN, ORANGE
from plot_potencia_inferencia import FORMATS, theoretical_board_power

FIG = ROOT / "doc/figuras"
OUT = ROOT / "doc/dados"
BASE_KG = 1.4
KIT_KG = 0.174  # NVIDIA, Jetson Orin Nano HW FAQ, pergunta 10.
TEST_MASSES = (0.100, KIT_KG, 0.250)
NOMINAL_ETA = 0.80


def mechanical(speed, mass, tip=110.0):
    """Cruzeiro nivelado, empuxo inclinado e raiz positiva da relação de Glauert."""
    speed = np.asarray(speed, dtype=float)
    rho, gravity, rotors, radius = 1.225, 9.81, 4, 0.12
    area = np.pi * radius**2
    drag = 0.5 * rho * speed**2 * 0.02
    thrust = np.hypot(mass * gravity, drag)
    alpha = np.arctan2(drag, mass * gravity)
    parallel = speed * np.cos(alpha)
    normal = speed * np.sin(alpha)
    vh2 = thrust / (2 * rho * rotors * area)
    lo, hi = np.zeros_like(speed), np.sqrt(vh2)
    for _ in range(45):
        vi = (lo + hi) / 2
        below = vi * np.hypot(parallel, normal + vi) < vh2
        lo, hi = np.where(below, vi, lo), np.where(below, hi, vi)
    induced = 1.15 * thrust * (lo + hi) / 2
    profile = rotors * (0.10 * 0.02 / 8) * rho * area * tip**3 * (1 + 4.65 * (parallel/tip)**2)
    return induced + profile + drag * speed


def samples(flight, region, end):
    folder = DATA / "raw" / f"flight-mbg-v2-{flight:02d}-region{region}"
    path, = (folder / "log").glob("*.csv")
    cols = ["time(millisecond)", "currentVoltage", "currentCurrent", "remainPowerPercent",
            "velocityX(mps)", "velocityY(mps)", "velocityZ(mps)", "altitude(m)"]
    x = pd.read_csv(path, usecols=cols)
    x = x[x[["currentVoltage", "currentCurrent", "remainPowerPercent"]].ne(0).all(axis=1)]
    x = x.drop_duplicates("time(millisecond)").sort_values("time(millisecond)")
    t = (x["time(millisecond)"].to_numpy() - x["time(millisecond)"].iloc[0]) / 1000
    channels = dict(vx=x["velocityX(mps)"], vy=x["velocityY(mps)"], vz=x["velocityZ(mps)"],
                    altitude=x["altitude(m)"], power=-x.currentVoltage*x.currentCurrent/1e6)
    # Mesmo extremo do balanço da bateria, interpolado entre as amostras que o cercam.
    tt = np.append(t[t < end], end)
    return tt, {name: np.interp(tt, t, vals) for name, vals in channels.items()}


def steady_windows(t, data):
    """Janelas contíguas de 5 s. Testa a regularidade sobre todas as amostras originais."""
    windows = []
    for start in np.arange(0, t[-1]-5+1e-9, 5):
        stop = start + 5
        left = max(0, t.searchsorted(start, side="right")-1)
        right = min(len(t), t.searchsorted(stop) + 1)
        if np.diff(t[left:right]).max() > 1:
            continue
        inner = t[(t > start) & (t < stop)]
        times = np.r_[start, inner, stop]
        vals = {key: np.interp(times, t, col) for key, col in data.items()}
        vx, vy = vals["vx"], vals["vy"]
        # Médias no tempo, sem dar peso maior a trechos com telemetria mais frequente.
        mean_x = np.trapezoid(vx, times) / 5
        mean_y = np.trapezoid(vy, times) / 5
        if (np.hypot(vx-mean_x, vy-mean_y).max() > 0.5 or np.abs(vals["vz"]).max() > 0.3
                or np.abs(vals["altitude"]-40).max() > 2):
            continue
        windows.append((np.trapezoid(np.hypot(vx, vy), times)/5, np.trapezoid(vals["power"], times)/5))
    return np.array(windows).reshape(-1, 2)


def plot_curve(flights, eta):
    v = np.linspace(flights.velocidade_ms.min(), flights.velocidade_ms.max(), 200)
    fig, (ax, errors) = plt.subplots(1, 2, figsize=(9, 3.7), gridspec_kw={"width_ratios":[1.45, 1]})
    ax.plot(v, mechanical(v, BASE_KG)/eta, color="0.2", lw=1.7, label="Modelo ajustado")
    held_out = flights.loc[flights.grupo != "ajuste"].sort_values("voo")
    flight_positions = pd.Series(np.arange(len(held_out)), index=held_out.voo)
    groups = [("ajuste", "Ajuste · 14 voos", BLUE, "o"),
              ("comparacao", "Comparação · 10 voos", GREEN, "s"),
              ("outra_campanha", "Região A · 2 voos", ORANGE, "^")]
    for name, label, color, marker in groups:
        group = flights.loc[flights.grupo == name]
        ax.scatter(group.velocidade_ms, group.bateria_W, s=27, facecolor="white",
                   edgecolor=color, marker=marker, label=label, zorder=3)
        if name != "ajuste":
            x = flight_positions.loc[group.voo]
            errors.vlines(x, 0, group.erro_ajustado_W, colors=color, lw=1.1)
            errors.scatter(x, group.erro_ajustado_W, s=26, color=color, marker=marker, zorder=3)
    ax.set(xlabel="Velocidade em cruzeiro (m/s)", ylabel="Potência elétrica (W)",
           xlim=(v.min()-.35, v.max()+.35), xticks=range(4,9))
    ax.set_title("(a) Previsão e potência da bateria", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    errors.axhline(0, color="0.35", lw=1)
    errors.set(xlabel="Número do voo", ylabel="Prevista − medida (W)",
               xticks=flight_positions.to_numpy(), xticklabels=held_out.voo.tolist(), ylim=(-9, 8))
    errors.tick_params(axis="x", labelsize=8)
    errors.set_title("(b) Erro nos voos fora do ajuste", fontsize=10)
    for panel in (ax, errors):
        panel.grid(axis="y", color="0.90", linewidth=0.6)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG/"fig_potencia_telemetria.pdf", bbox_inches="tight")
    plt.close(fig)


def campaign_rates(balance):
    """Mediana das taxas por voo em B–M, com amostragem a partir do quadro zero."""
    duration = balance.loc[balance.regiao != "A"].set_index("voo").video_s
    memory = pd.read_csv(DATA / "splits/memoria_sim_26voos.csv").set_index("vid").loc[duration.index]
    frames = pd.read_csv(OUT / "objetos_por_quadro/resumo_voos.csv").set_index("flight").loc[duration.index, "frames"]
    all_crops = frames * memory.n_tiles
    matched_step = (all_crops / memory.adiada_recortes).round().astype(int)
    rates = pd.DataFrame({
        "Vídeo completo": all_crops / duration,
        "Memória imediata": memory.imediata_recortes / duration,
        "1 em 30 quadros": ((frames - 1) // 30 + 1) * memory.n_tiles / duration,
        "Amostragem pareada": ((frames - 1) // matched_step + 1) * memory.n_tiles / duration,
        "Memória adiada": memory.adiada_recortes / duration,
    })
    return rates.median()


ENGINES = {"trt": "TensorRT", "ort_trt": "ORT + TensorRT", "ort_cuda": "ORT + CUDA",
           "ort_cpu": "ORT + CPU", "torchscript": "TorchScript"}
STRATEGIES = ["Memória adiada", "Amostragem pareada", "1 em 30 quadros", "Memória imediata", "Vídeo completo"]


def decimal(value):
    return f"{value:.2f}".replace(".", ",")


def plot_payload(eta, idle, benchmark, rates):
    rates = rates.loc[STRATEGIES]
    board = theoretical_board_power(rates, benchmark, idle)
    feasible = board.apply(lambda column: rates <= benchmark.loc[column.name, "fps"])
    supported = board.where(feasible)
    v = np.linspace(0, 14, 281)
    loaded = mechanical(v, BASE_KG + KIT_KG) / eta
    cruise = 4.1
    off = mechanical(cruise, BASE_KG + KIT_KG) / eta
    mass_w = off - mechanical(cruise, BASE_KG) / eta
    fig = plt.figure(figsize=(9.5, 5.9))
    rows = fig.add_gridspec(2, 1, left=.083, right=.98, top=.94, bottom=.20,
                            hspace=.85, height_ratios=[.9, 1.12])

    top = rows[0].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=.57)
    ax = fig.add_subplot(top[0])
    ax.plot(v, mechanical(v, BASE_KG) / eta, color=".35", ls="--", label="Sem kit")
    ax.plot(v, loaded, color=BLUE, label="174 g · desligado")
    ax.fill_between(v, mechanical(v, BASE_KG + min(TEST_MASSES)) / eta,
                    mechanical(v, BASE_KG + max(TEST_MASSES)) / eta,
                    color=BLUE, alpha=.14, label="100 a 250 g · desligado")
    ax.fill_between(v, loaded + supported.min().min(), loaded + supported.max().max(),
                    color=GREEN, alpha=.85, label="174 g · taxas atendidas")
    ax.set(xlabel="Velocidade (m/s)", ylabel="Potência total (W)", xlim=(0, 14),
           xticks=range(0, 15, 2), ylim=(155, 232))
    ax.set_title("(a) Efeito da massa e da velocidade", loc="left", fontsize=11)
    ax.grid(axis="y", color=".89", lw=.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", labelspacing=.2)
    fixed = fig.add_subplot(top[1])
    fixed.set_title("(b) Parcelas comuns a 4,1 m/s", loc="left", fontsize=11)
    bars = fixed.barh([0, 1], [mass_w, idle], color=[".48", ".68"], height=.44)
    fixed.bar_label(bars, labels=[decimal(mass_w), decimal(idle)], padding=3, fontsize=9)
    fixed.set(yticks=[0, 1], yticklabels=["Transportar o kit", "Placa em repouso"],
              xlabel="Potência adicional (W)", xlim=(0, 24), xticks=[0, 10, 20], ylim=(1.8, -.75))
    fixed.tick_params(axis="y", length=0, labelsize=9)
    fixed.grid(axis="x", color=".89", lw=.6)

    curves = rows[1].subgridspec(1, len(ENGINES), wspace=.26)
    delta_w = board - idle
    for i, (engine, label) in enumerate(ENGINES.items()):
        ax = fig.add_subplot(curves[i])
        for fmt, color, marker in FORMATS:
            key = f"{engine}_{fmt}"
            if key not in board:
                continue
            run = benchmark.loc[key]
            energy = run.E_dyn_mJ_per_inf / 1000
            for start, end, style in [(rates.min(), run.fps, "-"), (run.fps, rates.max(), "--")]:
                grid = np.geomspace(start, end, 85)
                ax.plot(grid, grid * energy, color=color, ls=style, lw=1.5)
            for supported_rate in (True, False):
                scenario = feasible[key] == supported_rate
                ax.scatter(rates[scenario], delta_w.loc[scenario, key], marker=marker, s=24,
                           facecolor=color if supported_rate else "white", edgecolor=color, zorder=3)
            ax.plot([], [], marker=marker, color=color, ls="none", markersize=4,
                    label=decimal(run.fps))
        ax.legend(title="Limite (rec./s)", loc="upper left", frameon=False, fontsize=8.4,
                  title_fontsize=8.4, labelspacing=.18, handlelength=.65, handletextpad=.4, borderpad=.12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set(xlim=(5, 980), ylim=(.14, 510), xticks=[7, 24, 160, 719],
               xticklabels=["7", "24", "160", "719"], yticks=[.2, 1, 5, 20, 100, 400],
               yticklabels=["0,2", "1", "5", "20", "100", "400"])
        ax.set_title(label, fontsize=9.5)
        ax.minorticks_off()
        ax.tick_params(labelleft=(i == 0), labelsize=8.2)
        ax.grid(axis="y", color=".89", lw=.6)
        if i == 0:
            ax.set_ylabel("Potência adicional (W)")
    bounds = rows[1].get_position(fig)
    fig.text(bounds.x0, bounds.y1 + .069, "(c) Acréscimo de potência em relação à placa em repouso",
             ha="left", va="bottom", fontsize=11)
    fig.text((bounds.x0 + bounds.x1)/2, bounds.y0 - .065, "Recortes solicitados por segundo (escala logarítmica)",
             ha="center", va="top", fontsize=9.5)
    legend = [plt.Line2D([], [], marker=marker, color=color, ls="none", label=fmt.upper())
              for fmt, color, marker in FORMATS]
    legend.extend([plt.Line2D([], [], color=".3", label="Até a taxa medida"),
                   plt.Line2D([], [], color=".3", ls="--", label="Extrapolação")])
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9,
               handletextpad=.4, columnspacing=1.4, bbox_to_anchor=(.51, .005))
    fig.savefig(FIG / "fig_massa_inferencia.pdf", bbox_inches="tight", pad_inches=.07)
    plt.close(fig)


def main():
    balance = pd.read_csv(OUT/"balanco_telemetria.csv")
    split = json.loads((DATA/"splits/splits.json").read_text(encoding="utf-8"))["regioes"]
    benchmark = pd.read_csv(JETSON/"summary.csv").query("mode == 'e2e'").set_index("config")
    jetson = benchmark.loc["trt_fp16"]
    idle = json.loads((JETSON/"results.json").read_text(encoding="utf-8"))["P_idle_W"]
    joules = jetson.E_dyn_mJ_per_inf/1000
    records, routes = [], {}
    for row in balance.itertuples():
        t, data = samples(row.voo, row.regiao, row.trecho_s)
        routes[row.voo] = (t, np.hypot(data["vx"], data["vy"]))
        windows = steady_windows(t, data)
        target = 7.9 if row.regiao == "A" else 4.1
        cruise = windows[np.abs(windows[:,0]-target) <= 0.6]
        if not len(cruise):
            raise ValueError(f"Sem janela de cruzeiro no voo {row.voo}")
        v, p = np.median(cruise, axis=0)
        group = "ajuste" if row.regiao in split["train"] else "comparacao"
        records.append(dict(voo=row.voo, regiao=row.regiao, grupo="outra_campanha" if row.regiao=="A" else group,
                            janelas_regulares=len(windows), janelas_cruzeiro=len(cruise),
                            velocidade_ms=v, bateria_W=p, mecanica_W=mechanical(v,BASE_KG)))
    flights = pd.DataFrame(records)
    calibration = flights.loc[flights.grupo=="ajuste"]
    eta = np.median(calibration.mecanica_W/calibration.bateria_W)
    flights["referencia_W"] = flights.mecanica_W/NOMINAL_ETA
    flights["ajustada_W"] = flights.mecanica_W/eta
    flights["erro_ajustado_W"] = flights.ajustada_W-flights.bateria_W
    flights.to_csv(OUT/"curva_potencia_por_voo.csv", index=False, float_format="%.7f")
    scenarios = []
    for row in balance.itertuples():
        t, v = routes[row.voo]
        no_payload = mechanical(v, BASE_KG)
        hours_area = row.trecho_s / (3600*row.area_sim_km2)
        for mass in TEST_MASSES:
            extra = (mechanical(v, BASE_KG+mass)-no_payload)/eta
            cost_mass = np.trapezoid(extra,t)/(3600*row.area_sim_km2)
            cost_idle = idle*hours_area
            cost_inference = row.adiada_recortes_s*joules*hours_area
            off = row.bateria_Wh_km2+cost_mass
            scenarios.append(dict(voo=row.voo, regiao=row.regiao, carga_g=round(1000*mass),
                                  bateria_Wh_km2=row.bateria_Wh_km2, massa_Wh_km2=cost_mass,
                                  ocioso_Wh_km2=cost_idle, inferencia_adiada_Wh_km2=cost_inference,
                                  total_desligada_Wh_km2=off, total_ociosa_Wh_km2=off+cost_idle,
                                  total_adiada_Wh_km2=off+cost_idle+cost_inference))
    scenarios = pd.DataFrame(scenarios)
    scenarios.to_csv(OUT/"massa_inferencia_por_voo.csv", index=False, float_format="%.7f")
    plt.rcParams.update({"font.size":9.5, "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "axes.axisbelow":True})
    rates = campaign_rates(balance)
    median_rate = rates["Memória adiada"]
    plot_curve(flights,eta)
    plot_payload(eta, idle, benchmark, rates)
    print(f"eta={eta:.8f}; P_ocioso={idle:.5f}; P_inferencia_adiada={median_rate*joules:.5f}")
    for group, df in flights.groupby("grupo"):
        err = df.erro_ajustado_W
        print(group, "n =",len(df),"janelas =",df.janelas_cruzeiro.sum(),
              "V, P, modelo =",df[["velocidade_ms","bateria_W","ajustada_W"]].median().round(3).to_list(),
              "MAE nominal, ajustado; viés =",round((df.referencia_W-df.bateria_W).abs().mean(),3),
              round(err.abs().mean(),3),round(err.mean(),3))
    print(flights.loc[flights.regiao=="A"].round(3).to_string(index=False))
    print(scenarios.query("regiao != 'A'").groupby("carga_g").median(numeric_only=True).round(3).to_string())
    for v in [0,4.1,7.9]:
        print(v,"m/s: base =",round(mechanical(v,BASE_KG)/eta,3),"massa 100/174/250 g =",
              [round((mechanical(v,BASE_KG+m)-mechanical(v,BASE_KG))/eta,3) for m in TEST_MASSES])
    drag = 0.5 * 1.225 * 4.1**2 * 0.02
    thrust_ratio = np.hypot((BASE_KG+KIT_KG)*9.81, drag)/np.hypot(BASE_KG*9.81, drag)
    tip = 110*np.sqrt(thrust_ratio)  # Cenário alternativo a coeficiente de tração constante.
    print("4.1 m/s, CT constante: delta de propulsao (W) =",
          (mechanical(4.1,BASE_KG+KIT_KG,tip)-mechanical(4.1,BASE_KG))/eta)

    # Exemplo da equação de energia por área. O voo 17 tem GSD de 1,40 cm/pixel.
    flight17 = flights.set_index("voo").loc[17]
    track17 = balance.set_index("voo").loc[17]
    width_m = 3840 * 0.014
    ideal_area_rate = flight17.velocidade_ms * width_m
    loaded_w = mechanical(flight17.velocidade_ms, BASE_KG + KIT_KG) / eta
    active_w = loaded_w + idle + track17.adiada_recortes_s * joules
    ideal_wh_km2 = active_w * 1e6 / (3600 * ideal_area_rate)
    route_wh_km2 = scenarios.set_index(["voo", "carga_g"]).loc[
        (17, round(1000 * KIT_KG)), "total_adiada_Wh_km2"
    ]
    print(f"Voo 17, kit ativo (Wh/km²): faixa ideal {ideal_wh_km2:.2f}, "
          f"trajeto registrado {route_wh_km2:.2f}")


if __name__ == "__main__":
    main()
