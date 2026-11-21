"""Verifica se os mesmos recipientes tinham vistas elegíveis fora da janela avaliada."""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from dados import CACHE, DATA, OUT, STRATEGIES, folder
from memoria_sim import simulate


def ciclo_anotado(video, refs):
    with np.load(CACHE / f"{video}.npz") as manifest:
        n = int(manifest["total_frames"])
        fps, gsd = float(manifest["fps"]), float(manifest["gsd"])
        width, height = map(int, manifest["size"])
        tiles = manifest["tiles"]
        masks = np.zeros((n, len(tiles), len(STRATEGIES)), dtype=bool)
        masks[:, :, 0] = True
        masks[::30, :, 2] = True
        masks[::int(manifest["step"]), :, 3] = True
        tile_index = {tuple(tile): i for i, tile in enumerate(tiles)}

        def record(policy, frame, x, y):
            masks[frame, tile_index[x,y], 1 if policy == "imediata" else 4] = True

        log, = (folder(video) / "log").glob("*.csv")
        simulate(log, n, fps, width, height, gsd, on_selected=record)
        if not np.array_equal(masks[manifest["frames"]], manifest["masks"]):
            raise ValueError("A seleção reconstruída diverge da seleção usada na inferência.")

    selected_ids = set(zip(refs.classe, refs.trilha))
    rows = []
    for _, track in ET.iterparse(DATA / "ann" / f"{video}.xml", events=["end"]):
        if track.tag != "track":
            continue
        cls, ident = track.get("label"), int(track.get("id"))
        if (cls, ident) in selected_ids:
            visible = [box for box in track.iter("box") if box.get("outside") != "1"]
            frames = np.array([int(box.get("frame")) for box in visible])
            xyxy = np.array([[float(box.get(k)) for k in ("xtl","ytl","xbr","ybr")] for box in visible])
            lo = np.maximum(xyxy[:,None,:2], tiles)
            hi = np.minimum(xyxy[:,None,2:], tiles + 640)
            area = (xyxy[:,2:] - xyxy[:,:2]).prod(axis=1)
            offers = np.maximum(hi-lo, 0).prod(axis=2) >= .4 * area[:,None]
            selected = (masks[frames] & offers[:,:,None]).any(axis=(0,1))
            rows.append(dict(video=video, classe=cls, trilha=ident,
                             **{s+"_voo": int(v) for s,v in zip(STRATEGIES,selected)}))
        track.clear()
    paired = refs.merge(pd.DataFrame(rows), on=["video","classe","trilha"], validate="one_to_one")
    assert len(paired) == len(refs)
    for s in STRATEGIES:
        assert (paired[s+"_voo"] >= paired[s]).all()
    return paired


if __name__ == "__main__":
    references = pd.read_csv(OUT / "oportunidades_trechos.csv")
    results = []
    for video, refs in references.groupby("video"):
        results.append(ciclo_anotado(video, refs))
        print(video, results[-1].filter(like="_voo").sum().to_dict(), flush=True)
    result = pd.concat(results, ignore_index=True)
    result.to_csv(OUT / "oportunidades_trilhas_completas.csv", index=False)
    print(result.filter(like="_voo").sum().to_string(), flush=True)
