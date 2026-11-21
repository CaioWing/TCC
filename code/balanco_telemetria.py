"""Reproduz o balanço por voo do capítulo 4 com os CSV da MBG-V2 e da Jetson.

Uso na raiz do projeto: python code/balanco_telemetria.py (requer ffprobe e NumPy).
As áreas e as taxas por voo são as da simulação geométrica em células de 0,5 m.
"""
import csv
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mbg_v2"
JETSON = ROOT / "data/yolo11n-jetson24-20261116"
OUTPUT = ROOT / "doc/dados/balanco_telemetria.csv"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def flight_energy(folder):
    video, = (folder / "avi").glob("*.avi")
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate", "-of", "json", str(video)
    ]))["streams"][0]
    video_s = int(stream["nb_frames"]) / float(Fraction(stream["avg_frame_rate"]))
    log, = (folder / "log").glob("*.csv")
    rows = read_csv(log)
    samples = {}
    invalid = 0
    for row in rows:
        voltage = float(row["currentVoltage"])
        current = float(row["currentCurrent"])
        if voltage == 0 or current == 0 or float(row["remainPowerPercent"]) == 0:
            invalid += 1
            continue
        time_ms = float(row["time(millisecond)"])
        samples.setdefault(time_ms, -voltage * current / 1e6)
    times_ms = sorted(samples)
    times = (np.array(times_ms) - times_ms[0]) / 1000
    power = np.array([samples[t] for t in times_ms])
    end = min(video_s, times[-1])
    inside = times < end
    t = np.append(times[inside], end)
    p = np.append(power[inside], np.interp(end, times, power))
    energy_Wh = np.sum(np.diff(t) * (p[:-1] + p[1:]) / 2) / 3600
    return dict(linhas=len(rows), invalidas=invalid, duplicadas=len(rows)-invalid-len(times),
                video_s=video_s, trecho_s=end, potencia_mediana_W=np.median(power[times <= end]),
                energia_bateria_Wh=energy_Wh)


def main():
    memory = read_csv(DATA / "splits/memoria_sim_26voos.csv")
    refs = read_csv(JETSON / "summary.csv")
    fp16, = [r for r in refs if r["config"] == "trt_fp16" and r["mode"] == "e2e"]
    idle = json.loads((JETSON / "results.json").read_text(encoding="utf-8"))["P_idle_W"]
    joules = float(fp16["E_dyn_mJ_per_inf"]) / 1000
    capacity = float(fp16["fps"])
    rows = []
    for sim in sorted(memory, key=lambda r: int(r["vid"])):
        flight, region = int(sim["vid"]), sim["region"]
        folder = DATA / "raw" / f"flight-mbg-v2-{flight:02d}-region{region}"
        battery = flight_energy(folder)
        area = float(sim["area_img_sim"]) / 1e6
        rate = float(sim["adiada_recortes_s"])
        nominal = int(sim["n_tiles"])  # Um quadro completo por segundo.
        if max(rate, nominal) > capacity:
            raise ValueError(f"Demanda excede a referência de bancada no voo {flight}")
        hours_per_km2 = battery["trecho_s"] / (3600 * area)
        flight_cost = battery["energia_bateria_Wh"] / area
        memory_cost = (idle + joules * rate) * hours_per_km2
        sample_cost = (idle + joules * nominal) * hours_per_km2
        rows.append(dict(voo=flight, regiao=region, **battery, area_sim_km2=area,
                         bateria_Wh_km2=flight_cost, adiada_recortes_s=rate,
                         temporal_recortes_s=nominal, placa_adiada_Wh_km2=memory_cost,
                         placa_temporal_Wh_km2=sample_cost,
                         diferenca_pareada_Wh_km2=sample_cost-memory_cost))
    if {row["voo"] for row in rows} != set(range(1, 27)) or len(rows) != 26:
        raise ValueError("A comparação requer exatamente um registro para cada um dos 26 voos")
    with OUTPUT.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
