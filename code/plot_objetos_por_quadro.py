"""Conta objetos simultâneos em todos os quadros anotados da MBG-V2.

Usa os XML originais dos 26 voos, incluindo quadros sem caixas e excluindo
marcadores outside=1. Cada identificador de trilha conta uma vez por quadro;
não se exige telemetria válida, pois esta estatística usa apenas anotações.
"""
import csv
import hashlib
import json
from pathlib import Path
from paleta_tcc import PALETTE
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/dados/objetos_por_quadro"
FIGURE = ROOT / "doc/figuras/fig_objetos_por_quadro.pdf"


def load_counts(path):
    counts = None
    tracks = boxes = outside = occluded = 0
    track_ids = set()
    for _, elem in ET.iterparse(path, events=["end"]):
        if elem.tag == "meta":
            start = int(elem.findtext("task/start_frame"))
            stop = int(elem.findtext("task/stop_frame"))
            assert start == 0
            counts = np.zeros(stop + 1, dtype=np.int32)
            elem.clear()
        elif elem.tag == "track":
            assert counts is not None
            track_id = elem.get("id")
            assert track_id not in track_ids
            track_ids.add(track_id)
            frames = []
            for box in elem.findall("box"):
                boxes += 1
                if box.get("outside", "0") == "1":
                    outside += 1
                    continue
                frame = int(box.get("frame"))
                assert 0 <= frame < len(counts)
                frames.append(frame)
                occluded += int(box.get("occluded", "0"))
            assert len(frames) == len(set(frames)), "Trilha repetida no mesmo quadro"
            np.add.at(counts, frames, 1)
            tracks += 1
            elem.clear()
    assert int(counts.sum()) == boxes - outside
    return counts, dict(tracks=tracks, raw_boxes=boxes, outside=outside,
                       included_boxes=int(counts.sum()), occluded_included=occluded)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "data/mbg_v2/raw").glob("flight-*/ann/*.xml"))
    assert len(paths) == 26
    flights, all_counts, sources = [], [], {}
    for path in paths:
        counts, info = load_counts(path)
        flight, region = re.search(r"video(\d+)_region([A-M])", path.stem).groups()
        row = dict(flight=int(flight), region=region, frames=len(counts),
                   zero=int(np.count_nonzero(counts == 0)),
                   one=int(np.count_nonzero(counts == 1)),
                   multiple=int(np.count_nonzero(counts >= 2)),
                   at_least_one_pct=float(100*np.mean(counts >= 1)),
                   at_least_two_pct=float(100*np.mean(counts >= 2)),
                   mean_objects=float(np.mean(counts)),
                   median_objects=float(np.median(counts)),
                   max_objects=int(counts.max()), **info)
        assert row["zero"] + row["one"] + row["multiple"] == row["frames"]
        flights.append(row)
        all_counts.append(counts)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with (OUT / f"voo{int(flight):02d}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "objects"])
            writer.writerows(enumerate(counts.tolist()))
    pooled = np.concatenate(all_counts)
    with (OUT / "resumo_voos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flights[0].keys())
        writer.writeheader()
        writer.writerows(flights)

    bins = [(0, 0), (1, 1), (2, 5), (6, 10), (11, 20), (21, int(pooled.max()))]
    bin_labels = ["0", "1", "2–5", "6–10", "11–20", "21 ou mais"]
    bin_counts = [int(np.count_nonzero((pooled >= low) & (pooled <= high)))
                  for low, high in bins]
    assert sum(bin_counts) == len(pooled)
    per_flight_bins = np.array([
        [np.count_nonzero((counts >= low) & (counts <= high)) for low, high in bins]
        for counts in all_counts])
    assert np.array_equal(per_flight_bins.sum(axis=0), bin_counts)
    assert np.array_equal(per_flight_bins.sum(axis=1), [row["frames"] for row in flights])
    with (OUT / "faixas_por_voo.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flight", "region", "objects_range", "frames", "percent"])
        for row, counts in zip(flights, per_flight_bins):
            writer.writerows((row["flight"], row["region"], label, int(count),
                              100*count/row["frames"])
                             for label, count in zip(bin_labels, counts))
    present = [row["at_least_one_pct"] for row in flights]
    multiple = [row["at_least_two_pct"] for row in flights]
    metadata = dict(
        sources_sha256=sources,
        counting="One annotated track per frame; all classes; outside=1 excluded; occluded=1 retained; all video frames including zero objects; no telemetry filtering",
        flights=26, frames=len(pooled), tracks=sum(row["tracks"] for row in flights),
        raw_boxes=sum(row["raw_boxes"] for row in flights),
        outside=sum(row["outside"] for row in flights), included_boxes=int(pooled.sum()),
        at_least_one_pct_median_across_flights=float(np.median(present)),
        at_least_one_pct_min_across_flights=float(np.min(present)),
        at_least_two_pct_median_across_flights=float(np.median(multiple)),
        at_least_two_pct_min_across_flights=float(np.min(multiple)),
        pooled_at_least_one_pct=float(100*np.mean(pooled >= 1)),
        pooled_at_least_two_pct=float(100*np.mean(pooled >= 2)),
        pooled_mean_objects=float(np.mean(pooled)),
        pooled_median_objects=float(np.median(pooled)),
        pooled_max_objects=int(pooled.max()),
        distribution=[dict(label=label, frames=count, percent=100*count/len(pooled))
                      for label, count in zip(bin_labels, bin_counts)])
    (OUT / "metadados.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k != "sources_sha256"}, indent=2, ensure_ascii=False))

    # Cinza, azul e laranja seguem as figuras do TCC. As faixas adicionais
    # usam verde, violeta e vermelho suaves; todas mantêm a cor entre painéis.
    distribution_colors = PALETTE
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.4),
                                 gridspec_kw={"height_ratios": [1.25, 1]})
        fig.subplots_adjust(left=.09, right=.98, top=.91, bottom=.09, hspace=.62)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=".9", linewidth=.6)
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))

        ax = axes[0]
        x = np.arange(26)
        bottom = np.zeros(26)
        for counts, label, color in zip(per_flight_bins.T, bin_labels, distribution_colors):
            values = 100*counts/np.array([row["frames"] for row in flights])
            ax.bar(x, values, bottom=bottom, width=.8, color=color,
                   edgecolor="white", linewidth=.35, label=label)
            bottom += values
        assert np.allclose(bottom, 100)
        ax.set_ylim(0, 100)
        ax.set_xlim(-.65, 25.65)
        ax.set_xticks(x, [f'{row["flight"]:02d}{"*" if row["region"] == "A" else ""}\n{row["region"]}' for row in flights], fontsize=8)
        ax.set_ylabel("Quadros de cada voo")
        ax.set_xlabel("Voo / região")
        ax.set_title("(a) Quantidade de objetos por quadro em cada voo", loc="left", pad=35)
        ax.legend(ncol=6, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0)

        ax = axes[1]
        values = 100*np.array(bin_counts)/len(pooled)
        ax.bar(bin_labels, values, width=.62, color=distribution_colors)
        for i, value in enumerate(values):
            ax.text(i, value+.65, f"{value:.1f}%".replace(".", ","), ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, np.ceil((max(values)+5)/10)*10)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.set_ylabel("Quadros da base")
        ax.set_xlabel("Objetos anotados simultaneamente no quadro")
        ax.set_title("(b) Quantidade de objetos por quadro no conjunto dos 26 voos", loc="left", pad=12)
        fig.savefig(FIGURE, bbox_inches="tight")
        fig.savefig(ROOT / "tmp/revisao/objetos-por-quadro.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
