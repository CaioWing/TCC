"""Reconstrói as posições usadas na avaliação do mapa final da memória por quadrantes."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

from adaptativa import METRICA, MOTOR, VOOS
from alertas_cobertura import referencias
from dados import OUT, ROOT
from metricas import CLASSES, focus_assignments, load_clips, projected, selected_tiles
from quadrantes_voos import DESTINO, LADO, METODOS, TEMP


CSV = DESTINO / "mapa_final_focos.csv"
FIGURA = ROOT / "doc/figuras/fig_mapa_final_quadrantes.pdf"
METODO = "quadrantes"
CHAVES = ["video", "classe", "trilha"]


def reconstruir(video):
    with np.load(TEMP / f"{video}.npz") as info:
        _, ref = referencias(video, int(info["total_frames"]), float(info["fps"]), info["size"], info["gsd"])
    clip, = load_clips(video, MOTOR, cache=TEMP)
    clip["centers"] = ref
    selecionados = selected_tiles(clip, clip["masks"][:, :, METODOS.index(METODO)])
    projecoes = projected(clip, selecionados)
    linhas = []
    for cls, centros, observacoes, ids, pares in focus_assignments(
        clip, projecoes, METRICA["conf"], METRICA["raio_m"], METRICA["dmax_m"]
    ):
        por_foco = dict(pares)
        refs_pareadas = set(por_foco.values())
        for i, (x, y) in enumerate(centros):
            linha = dict(video=video, classe=CLASSES[cls], foco=i, observacoes=observacoes[i],
                         foco_leste_m=x, foco_norte_m=y)
            if i in por_foco:
                chave = ids[por_foco[i]]
                xr, yr = ref[chave]
                linha.update(resultado="acerto", trilha=chave[1], ref_leste_m=xr, ref_norte_m=yr,
                             distancia_m=np.hypot(x-xr, y-yr))
            else:
                linha["resultado"] = "falso_alarme"
            linhas.append(linha)
        for j, chave in enumerate(ids):
            if j not in refs_pareadas:
                xr, yr = ref[chave]
                linhas.append(dict(video=video, classe=CLASSES[cls], resultado="omissao", trilha=chave[1],
                                   ref_leste_m=xr, ref_norte_m=yr))
    return pd.DataFrame(linhas)


def validar(mapa):
    publicado = pd.read_csv(DESTINO / "por_voo.csv").query("metodo == @METODO").set_index("video")
    contagens = pd.crosstab(mapa.video, mapa.resultado).reindex(publicado.index)
    np.testing.assert_array_equal(contagens[["acerto", "falso_alarme", "omissao"]], publicado[["vp", "fp", "fn"]])

    refs = mapa[mapa.trilha.notna()].astype({"trilha": "int64"}).set_index(CHAVES).sort_index()
    publicado = pd.read_csv(DESTINO / "por_trilha.csv").query("metodo == @METODO").set_index(CHAVES).sort_index()
    assert refs.index.is_unique and mapa[mapa.foco.notna()].set_index(["video", "classe", "foco"]).index.is_unique
    pd.testing.assert_index_equal(refs.index, publicado.index)
    np.testing.assert_array_equal(refs.resultado.eq("acerto"), publicado.acerto_foco)
    assert mapa.loc[mapa.resultado.eq("acerto"), "distancia_m"].between(0, METRICA["dmax_m"]).all()


def calcular_mapa():
    mapa = pd.concat([reconstruir(video) for video in VOOS], ignore_index=True)
    mapa[["foco", "trilha", "observacoes"]] = mapa[["foco", "trilha", "observacoes"]].astype("Int64")
    validar(mapa)
    mapa.to_csv(CSV, index=False, float_format="%.8f", columns=[
        "video", "classe", "resultado", "trilha", "ref_leste_m", "ref_norte_m",
        "foco", "foco_leste_m", "foco_norte_m", "observacoes", "distancia_m"])
    return mapa


ACERTO = "#205967"
ALARME = "#b55220"
OMISSAO = "#725187"
MALHA = "#babfc1"


def posicoes(dados, origem):
    return dados[[f"{origem}_leste_m", f"{origem}_norte_m"]].to_numpy(dtype=float)


def setor_detalhe(dados, malha):
    """Setor central com os três resultados, pares inteiros e símbolos afastados da borda."""
    refs, focos = posicoes(dados, "ref"), posicoes(dados, "foco")
    centro = np.nanmedian(refs, axis=0)
    meia_largura = 1.25 * LADO
    for i in np.argsort(np.sum((malha-centro)**2, axis=1), kind="stable"):
        canto = malha[i] - meia_largura
        fim = malha[i] + meia_largura
        ref_dentro = ((refs >= canto) & (refs <= fim)).all(axis=1)
        foco_dentro = ((focos >= canto) & (focos <= fim)).all(axis=1)
        if ((ref_dentro != foco_dentro) & dados.resultado.eq("acerto")).any():
            continue
        pontos = np.vstack((refs[ref_dentro], focos[foco_dentro]))
        if np.any(pontos-canto < 1) or np.any(fim-pontos < 1):
            continue
        local = dados[ref_dentro | foco_dentro]
        if local.resultado.nunique() == 3:
            return canto, fim, local
    raise ValueError("Nenhum setor contém os três resultados sem cortar um par.")


def marcar(eixo, xy, marca, cor, area, *, vazado=False, lw=.7, z=2):
    if marca == "x":
        eixo.scatter(*xy.T, marker=marca, color=cor, s=area, linewidths=lw, zorder=z)
    else:
        eixo.scatter(*xy.T, marker=marca, edgecolors=cor, facecolors="none" if vazado else cor,
                     s=area, linewidths=lw, zorder=z)


def mostrar(eixo, dados, malha, *, detalhe=False, referencias=True, inferencias=True):
    acertos = dados[dados.resultado == "acerto"]
    marcar(eixo, malha, ".", MALHA, 13 if detalhe else 1.05, lw=0, z=0)
    if referencias:
        if detalhe:
            marcar(eixo, posicoes(acertos, "ref"), "o", ACERTO, 48, vazado=True, lw=1.05, z=3)
            if inferencias:
                for xy in posicoes(dados[dados.trilha.notna()], "ref"):
                    eixo.add_patch(Circle(xy, METRICA["dmax_m"], facecolor="none", edgecolor="#788083",
                                          lw=.7, ls=":", zorder=1))
        marcar(eixo, posicoes(dados[dados.resultado == "omissao"], "ref"), "^", OMISSAO,
               37 if detalhe else 5, vazado=True, lw=1.1 if detalhe else .54, z=2)
    if inferencias:
        marcar(eixo, posicoes(acertos, "foco"), "o", ACERTO, 12 if detalhe else 4.6, lw=0, z=4)
        marcar(eixo, posicoes(dados[dados.resultado == "falso_alarme"], "foco"), "x", ALARME,
               37 if detalhe else 6, lw=1.35 if detalhe else .67, z=2)
    eixo.set_aspect("equal")
    eixo.set_xticks([])
    eixo.set_yticks([])
    for spine in eixo.spines.values():
        spine.set_color("#bec3c4")
        spine.set_linewidth(.55)


def barra(eixo, inicio, y, metros):
    eixo.plot([inicio, inicio+metros], [y, y], lw=1.7, color="#374347", solid_capstyle="butt")
    eixo.annotate(f"{metros:g} m".replace(".", ","), (inicio+metros, y), xytext=(3, 0),
                  textcoords="offset points", va="center", fontsize=7, color="#374347")


def desenhar(mapa):
    eventos = pd.read_csv(OUT / "passagens/eventos.csv",
                          usecols=["video", "i", "j", "leste_m", "norte_m"])
    eventos = eventos.drop_duplicates(["video", "i", "j"])
    malhas = {v: g[["leste_m", "norte_m"]].to_numpy() for v, g in eventos.groupby("video")}
    janela, fim, detalhe = setor_detalhe(mapa[mapa.video == VOOS[0]], malhas[VOOS[0]])
    fig = plt.figure(figsize=(7.25, 6.75))
    grade = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.85, 1.65], left=.03, right=.98,
                             top=.96, bottom=.115, hspace=.31)
    topo = grade[0].subgridspec(1, 2, wspace=.09)
    meio = grade[1].subgridspec(1, 4, wspace=.11)
    for i, (video, espaco) in enumerate(zip(VOOS, [*topo, *meio], strict=True)):
        eixo = fig.add_subplot(espaco)
        dados, malha = mapa[mapa.video == video], malhas[video]
        mostrar(eixo, dados, malha)
        pontos = np.vstack([malha, posicoes(dados, "foco"), posicoes(dados, "ref")])
        minimo, maximo = np.nanmin(pontos, axis=0), np.nanmax(pontos, axis=0)
        x0, y0 = minimo - [8, 25]
        eixo.set(xlim=(x0, maximo[0]+8), ylim=(y0, maximo[1]+8))
        num, regiao = video.removeprefix("video").split("_region")
        eixo.set_title(f"({chr(97+i)}) Voo {num} · {regiao}", pad=3, loc="left")
        barra(eixo, x0+7, y0+10, 50)
        if i == 0:
            eixo.add_patch(Rectangle(janela, *(fim-janela), fc="none", ec="#202b2e", lw=.85, zorder=5))
            eixo.text(.985, .045, "N ↑", transform=eixo.transAxes, ha="right", color="#374347", fontsize=7.5)

    parte = grade[2].subgridspec(1, 3, wspace=.12)
    for j, (titulo, refs, inf) in enumerate([("Referência", True, False), ("Memória", False, True),
                                            ("Sobreposição", True, True)]):
        eixo = fig.add_subplot(parte[j])
        mostrar(eixo, detalhe, malhas[VOOS[0]], detalhe=True, referencias=refs, inferencias=inf)
        eixo.set(xlim=(janela[0], fim[0]), ylim=(janela[1], fim[1]))
        eixo.set_title(f"({chr(103+j)}) {titulo}", loc="left", pad=3)
        barra(eixo, janela[0]+.8, janela[1]+.85, 5)
    leg = [
        Line2D([], [], ls="", marker="o", mec=ACERTO, mfc="none", ms=6, label="Referência pareada"),
        Line2D([], [], ls="", marker="o", color=ACERTO, ms=4, label="Foco pareado"),
        Line2D([], [], ls="", marker="x", color=ALARME, ms=6, label="Falso alarme"),
        Line2D([], [], ls="", marker="^", mec=OMISSAO, mfc="none", ms=6, label="Referência omitida"),
        Line2D([], [], ls="", marker=".", color=MALHA, ms=6, label="Ponto da malha"),
    ]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(.5,.005), ncol=3,
               frameon=False, handletextpad=.22, columnspacing=1.25, fontsize=8.6)
    fig.savefig(FIGURA, bbox_inches="tight", pad_inches=.055, metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__)
    argumentos.add_argument("--reconstruir", action="store_true",
                            help="Recalcula o CSV pelas anotações e predições do ensaio original.")
    if argumentos.parse_args().reconstruir or not CSV.is_file():
        mapa = calcular_mapa()
    else:
        mapa = pd.read_csv(CSV)
        validar(mapa)
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9.3, "pdf.fonttype": 42}):
        desenhar(mapa)
