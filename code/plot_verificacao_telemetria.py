"""Reproduz a comparação entre os modelos de faixa e os 26 voos da MBG-V2."""
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from paleta_tcc import BLUE, GREEN, ORANGE
from plot_ajuste_gsd import ROOT, fit_gsd, load_tracks


DATA = ROOT / "data/mbg_v2"
FIG = ROOT / "doc/figuras/fig_verificacao_telemetria.pdf"


def visible_span(path, fps):
    """Duração inclusiva entre a primeira e a última caixa visível de cada trilha."""
    spans = []
    for _, track in ET.iterparse(path, events=["end"]):
        if track.tag == "track":
            frames = [int(b.get("frame")) for b in track.findall("box") if b.get("outside") != "1"]
            if frames:
                spans.append(max(frames) - min(frames) + 1)
            track.clear()
    return np.median(spans) / fps


def tilt_in_both_directions(log, nominal_speed, end):
    """Média das medianas por sentido: tangente da inclinação e velocidade horizontal."""
    valid = log[["currentVoltage", "currentCurrent", "remainPowerPercent"]].ne(0).all(axis=1)
    samples = log.loc[valid].drop_duplicates("time(millisecond)")
    time = (samples["time(millisecond)"] - samples["time(millisecond)"].min()) / 1000
    speed = np.hypot(samples["velocityX(mps)"], samples["velocityY(mps)"])
    cruise = ((time <= end) & (speed.sub(nominal_speed).abs() <= .6)
              & (samples["velocityZ(mps)"].abs() <= .3) & (samples["altitude(m)"].sub(40).abs() <= 2))
    samples = samples.loc[cruise]
    speed = speed.loc[cruise].to_numpy()
    velocity = samples[["velocityX(mps)", "velocityY(mps)"]].to_numpy()
    # O eixo que maximiza a projeção quadrática das velocidades representa ambos os sentidos.
    _, axes = np.linalg.eigh(velocity.T @ velocity)
    along = (velocity @ axes[:, -1]) / speed
    direction = np.arctan2(velocity[:, 1], velocity[:, 0])
    pitch, roll, yaw = np.deg2rad(samples[["pitch(deg)", "roll(deg)", "yaw(deg)"]].to_numpy().T)
    # Componente horizontal do eixo de sustentação na direção do movimento, dividida pela vertical.
    relative_heading = direction - yaw
    tilt = (-np.tan(pitch) * np.cos(relative_heading)
            + np.tan(roll) / np.cos(pitch) * np.sin(relative_heading))
    threshold = np.cos(np.deg2rad(15))
    legs = [along >= threshold, along <= -threshold]
    if not all(mask.any() for mask in legs):
        return np.nan, np.nan
    return tuple(np.mean([np.median(values[mask]) for mask in legs]) for values in (tilt, speed))


def observed_errors():
    balance = pd.read_csv(ROOT / "doc/dados/balanco_telemetria.csv").set_index("voo")
    rows = []
    for flight, record in balance.iterrows():
        folder = DATA / "raw" / f"flight-mbg-v2-{flight:02d}-region{record.regiao}"
        tracks, meta = load_tracks(flight=folder)
        _, _, scale, _, best = fit_gsd(tracks)
        gsd = scale[best] / 100
        columns = ["speed(mps)", "remainPowerPercent", "currentVoltage", "currentCurrent",
                   "time(millisecond)", "velocityX(mps)", "velocityY(mps)", "velocityZ(mps)",
                   "altitude(m)", "pitch(deg)", "roll(deg)", "yaw(deg)"]
        log = pd.read_csv(next((folder / "log").glob("*.csv")), usecols=columns)
        v = log.loc[log.remainPowerPercent > 0, "speed(mps)"].median()
        energy = record.potencia_mediana_W * 1e6 / (3600 * v * meta["width"] * gsd)
        duration = meta["height"] * gsd / v
        observed = visible_span(next((folder / "ann").glob("*.xml")), meta["fps"])
        tilt, cruise_speed = tilt_in_both_directions(log, 7.9 if record.regiao == "A" else 4.1, record.trecho_s)
        rows.append((flight, record.regiao, 100 * (record.bateria_Wh_km2 / energy - 1),
                     100 * (observed / duration - 1), tilt, cruise_speed))
    return pd.DataFrame(rows, columns=["flight", "region", "energy", "duration", "tilt", "cruise_speed"])


def paineis_fisicos(errors, energy, duration, tilt):
    flights = errors.sort_values("flight")
    regions = flights.groupby("region", sort=False).flight.agg(["min", "max"])
    for ax, column, title in [(energy, "energy", "(a) Energia por área"),
                              (duration, "duration", "(b) Tempo de visibilidade")]:
        ax.set_title(title, loc="left")
        ax.axhspan(-10, 10, facecolor="0.94")
        ax.axhline(0, color="0.28", lw=1)
        for mask, marker, color, label in [
            (errors.region.eq("A"), "s", ORANGE, "Região A · 7,9 m/s"),
            (errors.region.ne("A"), "o", BLUE, "Regiões B a M · 4,1 m/s"),
        ]:
            ax.scatter(errors.loc[mask, "flight"], errors.loc[mask, column], c=color, s=25,
                       marker=marker, edgecolors="white", lw=.45, zorder=3, label=label)
        ax.set(xlim=(flights.flight.min()-.55, flights.flight.max()+.55), ylim=(-15, 36),
               xticks=flights.flight, yticks=[-10, 0, 10, 20, 30])
        for boundary in regions["min"].iloc[1:]:
            ax.axvline(boundary-.5, color=".82", lw=.55, ls=":", zorder=1)
        ax.grid(axis="y", color=".89", lw=.6)
    energy.set_ylabel("Diferença (%)")
    energy.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: "0" if y == 0 else f"{y:+g}".replace("-", "−")))
    energy.text(.5, -.11, "Gasto por área menor (−) ou maior (+) que o previsto para o cruzeiro",
                transform=energy.transAxes, ha="center", va="top", fontsize=8.5)
    energy.tick_params(axis="x", length=0, labelbottom=False)
    energy.legend(loc="lower right", bbox_to_anchor=(1, 1.035), ncol=2, fontsize=8.3,
                  frameon=False, borderaxespad=0, columnspacing=1.4, handletextpad=.25)
    duration.set_ylabel("Desvio (%)")
    duration.tick_params(axis="x", labelsize=8.3, length=0, pad=3)
    for region, bounds in regions.iterrows():
        center = (bounds["min"] + bounds["max"]) / 2
        duration.annotate(region, (center, 0), xycoords=duration.get_xaxis_transform(),
                          xytext=(0, -17), textcoords="offset points", ha="center",
                          va="top", fontsize=9, fontweight="bold")
        duration.plot([bounds["min"]-.34, bounds["max"]+.34], [-.19, -.19],
                      transform=duration.get_xaxis_transform(), clip_on=False, color=".58", lw=.65)
    for label, offset in [("Voo", -3), ("Região", -17)]:
        duration.annotate(label, (0, 0), xycoords="axes fraction", xytext=(-5, offset),
                          textcoords="offset points", ha="right", va="top", fontsize=8.3)

    campaign = errors.groupby(errors.region.eq("A"))[["tilt", "cruise_speed"]].median()
    ratio = campaign.loc[True] / campaign.loc[False]
    ratios = [ratio.tilt, ratio.cruise_speed, ratio.cruise_speed**2]
    tilt.set_title("(c) Inclinação entre as campanhas", loc="left")
    tilt.bar([0, 1, 2], ratios, width=.52, color=[BLUE, ".70", ".70"])
    for x, y in enumerate(ratios):
        tilt.text(x, y+.10, f"{y:.2f}".replace(".", ","), ha="center")
    tilt.set(xticks=[0, 1, 2], xticklabels=["Observada", "Se crescer\ncom V", "Se crescer\ncom V²"],
             ylabel="Razão entre 7,9 e 4,1 m/s", ylim=(0, 4.35), yticks=[0, 1, 2, 3, 4])
    tilt.grid(axis="y", color=".89", lw=.6)


def render(errors):
    sampling = pd.read_csv(DATA / "splits/amostragem_sim_26voos.csv").query("region != 'A'")
    memory = pd.read_csv(DATA / "splits/memoria_sim_26voos.csv").query("region != 'A'")
    steps = np.arange(30, 241, 30)
    missed = 100 * sampling[[f"nv_k{k}" for k in steps]].to_numpy()
    with plt.rc_context({"font.size": 9.5, "axes.titlesize": 10.5, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.axisbelow": True, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.65, 6.0))
        grid = fig.add_gridspec(2, 1, height_ratios=[1.65, 1], left=.09, right=.98,
                                bottom=.10, top=.95, hspace=.54)
        flights_grid = grid[0].subgridspec(2, 1, hspace=.48)
        energy = fig.add_subplot(flights_grid[0])
        duration = fig.add_subplot(flights_grid[1], sharex=energy)
        comparison_grid = grid[1].subgridspec(1, 2, wspace=.35)
        tilt = fig.add_subplot(comparison_grid[0])
        coverage = fig.add_subplot(comparison_grid[1])

        paineis_fisicos(errors, energy, duration, tilt)

        coverage.set_title("(d) Área que fica sem análise", loc="left")
        lo, mid, hi = np.min(missed, axis=0), np.median(missed, axis=0), np.max(missed, axis=0)
        coverage.fill_between(steps, lo, hi, facecolor=BLUE, alpha=.15)
        coverage.plot(steps, mid, color=BLUE, lw=1.7, marker="o", ms=3.7, label="Quadros a intervalos fixos")
        zero = 100 * memory.adiada_frac_nao_analisada.max()
        coverage.axhline(zero, color=GREEN, lw=1.5, ls="--", label=f"Saída da célula ({zero:g}%)")
        coverage.set(xlim=(24, 246), xticks=[30, 60, 120, 180, 240], ylim=(-1, np.ceil(hi.max())+1),
                     xlabel="Intervalo da amostragem (quadros)", ylabel="Área sem análise (%)")
        coverage.grid(axis="y", color=".89", lw=.6)
        coverage.legend(loc="upper left", frameon=False, fontsize=8.5)
        coverage.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))
        fig.savefig(FIG, bbox_inches="tight", pad_inches=.07)
        plt.close(fig)


def main():
    errors = observed_errors()
    render(errors)
    print(errors.groupby(errors.region.eq("A"))[["energy", "duration"]].agg(["min", "median", "max"]))


if __name__ == "__main__":
    main()
