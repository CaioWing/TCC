"""Dois achados distintos no mesmo trecho, com a mesma cota de inferências."""
import matplotlib.patheffects as effects
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from dados import CACHE
from figura_exemplo_temporal import fotos_do_video
from figura_passagem import pontos_no_quadro
from figuras_malha import BORDA, ENVIO, VISIVEL
from fronteiras import selected_tiles_fronteiras
from memoria_sim import T
from metricas import CLASSES, box_ious, focus_counts, load_clips, projected
from passagens import DESTINO, MASCARAS, METRICA, MOTOR, VOOS, carregar_voo, melhor_recorte
from seguimento import temporal


def desenhar_exemplo(eixos):
    video = "video09_regionE"
    trecho = 2
    clip = next(c for c in load_clips(video, MOTOR, cache=CACHE) if c["clip"] == trecho)
    frames = clip["frames"]
    pose, tamanho, tiles, _ = carregar_voo(video)
    with np.load(MASCARAS / f"{video}.npz") as dados:
        geo = dados["extremos"][frames]
    cota = int(geo.sum())
    fase = 0.
    tempo = temporal(geo.shape, cota, fase, 2*VOOS.index(video) + trecho - 1)
    assert tempo.sum() == cota
    resultados = pd.read_csv(DESTINO / "por_trecho_fronteiras.csv").query(
        "conjunto == 'iniciais' and video == @video and trecho == @trecho and plano == 'extremos'")
    eventos = pd.read_csv(DESTINO / "eventos.csv").query("video == @video")
    ids = {}
    quadros = (3130, 3200)
    fotos = fotos_do_video(quadros, video)
    exemplos = [("temporal", tempo, 60, "(c) Temporal"),
                ("geografica", geo, 63, r"(d) Geográfica ($k_{\mathrm{int}}=0$)")]
    piscina = CLASSES.index("pool")
    for ax, foto, q, (metodo, mask, trilha, titulo) in zip(eixos, fotos, quadros, exemplos):
        pred = selected_tiles_fronteiras(clip, mask)
        contagens, ids[metodo] = focus_counts(
            clip, projected(clip, pred), METRICA["conf"], METRICA["raio_m"],
            METRICA["dmax_m"], return_matches=True)
        linha, = resultados.query("estrategia == @metodo and (fase == -1 or fase == @fase)").itertuples()
        np.testing.assert_array_equal(contagens.sum(axis=0), (linha.vp, linha.fp, linha.fn))
        assert (linha.recortes, linha.quadros_usados) == (cota, mask.any(axis=1).sum())
        t, = np.flatnonzero(frames == q)
        anotada, = [b for b in clip["gt"][t] if int(b[4]) == trilha and int(b[5]) == piscina]
        candidatas = pred[t][(pred[t][:, 4] >= METRICA["conf"]) & (pred[t][:, 5] == piscina)]
        ious = box_ious(candidatas, anotada[None, :]).ravel()
        assert ious.max() >= .5
        x1, y1, x2, y2 = candidatas[ious.argmax(), :4]
        visiveis = pontos_no_quadro(eventos, pose, tamanho, q)
        pedido = (visiveis.quadro_entrada == q) | (visiveis.quadro_saida == q)
        tiles_pedidos = [melhor_recorte(p.u, p.v, tiles) for p in visiveis[pedido].itertuples()]
        np.testing.assert_array_equal(np.unique(tiles_pedidos), np.flatnonzero(geo[t]))
        ativos = pedido & (metodo == "geografica")
        w, h = tamanho
        ax.imshow(foto, extent=(0, w, h, 0))
        for origem in tiles[mask[t]]:
            ax.add_patch(Rectangle(origem, T, T, fill=False, ec=ENVIO, lw=1,
                                   path_effects=[effects.withStroke(linewidth=1.7, foreground=BORDA)]))
        for grupo, forma, cor in [(visiveis[~ativos], "o", VISIVEL), (visiveis[ativos], "D", ENVIO)]:
            ax.scatter(grupo.u, grupo.v, marker=forma, s=14 if forma == "o" else 28,
                       fc=cor, ec=BORDA, lw=.7, zorder=3, clip_on=False)
        caixa = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec="#7cff85", lw=1.7,
                          path_effects=[effects.withStroke(linewidth=2.8, foreground=BORDA)], zorder=4)
        ax.add_patch(caixa)
        nome = "Piscina A" if metodo == "temporal" else "Piscina B"
        ax.annotate(nome, (x1+(x2-x1)/2, y1), xytext=(0, 11), textcoords="offset points",
                    ha="center", va="bottom", color="white", fontsize=7.5, zorder=5,
                    bbox=dict(boxstyle="round,pad=.15", facecolor=BORDA, edgecolor="white", lw=.5),
                    arrowprops=dict(arrowstyle="-", shrinkB=0, color="white", lw=.75))
        n = mask[t].sum()
        ax.set_title(f"{titulo} · quadro {q} · {n} recorte" + ("s" if n != 1 else ""),
                     fontsize=8.4, loc="left", pad=4)
        ax.set(xlim=(0, w), ylim=(h, 0))
        ax.set_axis_off()
    assert (piscina, 60) in ids["temporal"] - ids["geografica"]
    assert (piscina, 63) in ids["geografica"] - ids["temporal"]
    print(f"Voo 09, trecho {trecho}: {cota} recortes, {len(ids['temporal'])} e "
          f"{len(ids['geografica'])} acertos, {len(ids['temporal'] & ids['geografica'])} comuns.")
