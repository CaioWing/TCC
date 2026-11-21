"""Conta trilhas anotadas que aparecem nos quadros amostrados na validação."""
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mbg_v2"


def main():
    splits = json.loads((DATA / "splits/splits.json").read_text(encoding="utf-8"))
    classes = set(splits["classes_avaliadas"])
    with (DATA / "splits/memoria_sim_26voos.csv").open(newline="") as source:
        flights = {int(row["vid"]): row for row in csv.DictReader(source)}
    rows = []
    for video in splits["voos"]["val"]:
        flight = int(video.removeprefix("video").split("_")[0])
        sim = flights[flight]
        root = ET.parse(DATA / "ann" / f"{video}.xml").getroot()
        nframes = int(root.findtext("meta/task/stop_frame")) + 1
        ntiles = int(sim["n_tiles"])
        budget = int(sim["adiada_recortes"])
        matched_step = round(ntiles * nframes / budget)
        tracks = [
            [int(b.get("frame")) for b in tr.iter("box") if b.get("outside") != "1"]
            for tr in root.iter("track") if tr.get("label") in classes
        ]
        if not all(tracks):
            raise ValueError(f"Trilha sem caixa visível em {video}")
        for k in (30, matched_step):
            hits = np.zeros(k, dtype=int)
            for frames in tracks:
                # Cada resíduo é uma fase que retém ao menos uma caixa da trilha.
                hits[np.unique(np.asarray(frames) % k)] += 1
            rows.append(dict(voo=flight, regiao=sim["region"], trilhas=len(tracks), passo=k,
                             recortes_adiada=budget, recortes_temporais_medios=ntiles*nframes/k,
                             trilhas_com_vista_min=int(min(hits)), trilhas_com_vista_max=int(max(hits)),
                             pct_com_vista_medio=100*float(np.mean(hits))/len(tracks)))
    with (ROOT / "doc/dados/oportunidades_validacao.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
