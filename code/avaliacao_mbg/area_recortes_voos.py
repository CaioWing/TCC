"""Conta o terreno que apareceu na imagem e nos recortes enviados em cada voo completo."""
import numpy as np
import pandas as pd

from cobertura_continua import DESTINO, METODOS, TEMP
from dados import SPLITS, folder
from memoria_sim import T, grade_projecao


def contar_celulas(video, *, cache=TEMP, metodos=METODOS):
    with np.load(cache / f"{video}.npz") as dados:
        n, fps, gsd = int(dados["total_frames"]), float(dados["fps"]), float(dados["gsd"])
        largura, altura = dados["size"]
        tiles, masks = dados["tiles"], dados["masks"]
        envios = dict(zip(dados["frames"], masks))

    assert masks[:, :, 0].sum() == masks[:, :, 1].sum()
    log, = (folder(video) / "log").glob("*.csv")
    shape, raster = grade_projecao(log, n, fps, largura, altura, gsd, centros_na_origem=True)
    imageadas = np.zeros(shape, bool)
    processadas = {metodo: np.zeros(shape, bool) for metodo in metodos}
    for quadro in range(n):
        j0, j1, i0, i1, visivel, u, v = raster(quadro)
        janela = np.s_[j0:j1, i0:i1]
        imageadas[janela] |= visivel
        if quadro not in envios:
            continue
        for metodo, selecionados in zip(metodos, envios[quadro].T, strict=True):
            if selecionados.all():
                processadas[metodo][janela] |= visivel
            else:
                for x, y in tiles[selecionados]:
                    processadas[metodo][janela] |= (u >= x) & (u < x+T) & (v >= y) & (v < y+T)

    assert all(not np.any(mascara & ~imageadas) for mascara in processadas.values())
    return {"video": video, "celulas_imageadas": int(imageadas.sum()),
            **{f"celulas_{metodo}": int(mascara.sum()) for metodo, mascara in processadas.items()}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quadrantes", action="store_true")
    args = ap.parse_args()
    if args.quadrantes:
        from quadrantes_voos import DESTINO, METODOS, TEMP
    contagens = pd.DataFrame([contar_celulas(video, cache=TEMP, metodos=METODOS)
                              for video in SPLITS["voos"]["test"]])
    contagens.to_csv(DESTINO / "area_recortes_por_voo.csv", index=False)
    print(contagens.to_string(index=False))
