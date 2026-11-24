"""Prepara as tabelas e a figura somente após completar a fila dos seis voos."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from dados import DATA, OUT
from paleta_tcc import BLUE, GREEN, ORANGE, PURPLE, RED

DESTINO = OUT / "altitude_integral"
NOMES = {"cobertura":"Memória por cobertura", "intervalo":"Um quadro a cada 30", "integral":"Integral"}


def dados_completos():
    r = pd.read_csv(DESTINO/"resumo_seis_voos.csv")
    t = pd.read_csv(DESTINO/"por_trilha.csv")
    v = pd.read_csv(DESTINO/"por_voo.csv")
    protocolo = json.loads((DESTINO/"protocolo.json").read_text(encoding="utf-8"))
    alturas = [40, *protocolo["alturas_simuladas_m"]]
    esperado = {(metodo,h) for metodo in ("cobertura","intervalo") for h in alturas} | {("integral",40)}
    assert set(zip(r.metodo,r.altura_m))==esperado and len(r)==len(esperado)
    assert set(zip(v.metodo,v.altura_m))==esperado and set(zip(t.metodo,t.altura_m))==esperado
    assert r.referencias.eq(1924).all() and r.voos.eq(6).all()
    assert v.groupby(["metodo","altura_m"]).video.nunique().eq(6).all()
    assert not v.duplicated(["metodo","altura_m","video"]).any()
    assert not t.duplicated(["metodo","altura_m","video","classe","trilha"]).any()
    assert t.groupby(["metodo","altura_m"]).size().eq(1924).all()
    assert r.query("metodo != 'integral'").recortes.eq(37776).all()
    assert r.query("metodo == 'integral'").recortes.item()==47150*24
    return r,t


def perdas_pareadas(t):
    chaves = ["video","classe","trilha"]
    linhas = []
    for metodo in ("cobertura","intervalo"):
        grupos = t[t.metodo==metodo]
        base = grupos[grupos.altura_m==40].set_index(chaves).sort_index()
        for h, g in grupos[grupos.altura_m!=40].groupby("altura_m"):
            g = g.set_index(chaves).sort_index()
            pd.testing.assert_index_equal(g.index,base.index)
            pd.testing.assert_series_equal(g.quadro_entrada,base.quadro_entrada)
            for classe, indices in g.groupby(level="classe").groups.items():
                b, a = base.loc[indices], g.loc[indices]
                b_geo, a_geo = b.acerto_foco.astype(bool), a.acerto_foco.astype(bool)
                b_vis, a_vis = b.atraso_s.notna(), a.atraso_s.notna()
                linhas.append(dict(metodo=metodo,altura_m=h,classe=classe,referencias=len(a),
                                   localizados_40m=b_geo.sum(),localizados_altura=a_geo.sum(),
                                   perderam_localizacao=(b_geo & ~a_geo).sum(),ganharam_localizacao=(~b_geo & a_geo).sum(),
                                   perderam_alerta_visual=(b_vis & ~a_vis).sum(),ganharam_alerta_visual=(~b_vis & a_vis).sum()))
    return pd.DataFrame(linhas)


def figura(r, metodos=("cobertura", "intervalo"), nomes=NOMES):
    fig, paineis = plt.subplots(1,3,figsize=(8.2,2.85),layout="constrained")
    for metodo, cor, marca, estilo in zip(metodos, (BLUE, ORANGE), ("o", "s"), ("-", "--"), strict=True):
        grupo = r[r.metodo==metodo].sort_values("altura_m")
        for ax, valores in zip(paineis, [100*grupo.recall, grupo.fp, 100*grupo.confirmados_ate_1s/grupo.referencias]):
            ax.plot(grupo.altura_m, valores, marker=marca, color=cor, ls=estilo,
                    lw=1.5, ms=3.8, markerfacecolor="white", label=nomes[metodo])
            ax.set_xlabel("Altura equivalente (m)")
            ax.grid(axis="y",lw=.5,color=".85")
            ax.set_xticks(grupo.altura_m)
    for ax,titulo in zip(paineis,["Recall geográfico (%)","Falsos alarmes no mapa","Trilhas com alerta até 1 s (%)"]):
        ax.set_title(titulo, fontsize=9.5)
    paineis[0].legend(frameon=False,fontsize=7)
    return fig


def numero(x, casas):
    if pd.isna(x):
        return "—"
    return f"{x:,.{casas}f}".translate(str.maketrans({",":".", ".":","}))


def resultados_por_classe(trilhas):
    chaves = ["video", "classe", "trilha"]
    anotacoes = pd.read_csv(DATA / "objetos_por_trilha.csv").rename(
        columns={"label": "classe", "tid": "trilha", "side": "lado_original_px"})
    medidas = trilhas.merge(anotacoes[chaves + ["lado_original_px"]], on=chaves,
                            how="left", validate="many_to_one")
    assert medidas.lado_original_px.gt(0).all()
    medidas["lado_px"] = medidas.lado_original_px * 40 / medidas.altura_m
    chaves_metodo = [*chaves, "metodo"]
    base = trilhas.loc[trilhas.altura_m == 40, chaves_metodo + ["acerto_foco"]]
    medidas = medidas.merge(base.rename(columns={"acerto_foco": "acerto_40m"}),
                            on=chaves_metodo, how="left", validate="many_to_one")
    assert medidas.acerto_40m.notna().all()
    localizados_base = medidas.lado_original_px.where(medidas.acerto_40m.eq(1) & medidas.altura_m.gt(40))
    medidas["lado_dos_perdidos"] = localizados_base.where(medidas.acerto_foco.eq(0))
    medidas["lado_dos_retidos"] = localizados_base.where(medidas.acerto_foco.eq(1))
    resumo = medidas.groupby(["classe", "metodo", "altura_m"], as_index=False).agg(
        referencias=("trilha", "size"), localizados=("acerto_foco", "sum"),
        lado_medio_px=("lado_px", "mean"),
        mediana_lado_40m_perdidos_px=("lado_dos_perdidos", "median"),
        mediana_lado_40m_retidos_px=("lado_dos_retidos", "median"))
    resumo["recall_pct"] = 100 * resumo.localizados / resumo.referencias
    return resumo


def figura_por_classe(resumo, *, metodos=("cobertura", "intervalo"),
                      titulos=("Cobertura", "Intervalo"), alturas=None):
    estilos = {
        "bucket": ("Balde", RED, "o"), "tire": ("Pneu", ORANGE, "s"),
        "potted_plant": ("Vaso de planta", PURPLE, "^"),
        "storm_drain": ("Bueiro", GREEN, "D"),
        "watertank": ("Caixa d’água", BLUE, "v"), "pool": ("Piscina", "#404449", "P"),
    }
    if alturas is None:
        alturas = sorted(resumo.altura_m.unique())
    originais = resumo.loc[(resumo.metodo == metodos[0]) & resumo.altura_m.eq(40)]
    lado_original = originais.set_index("classe").lado_medio_px.sort_values()
    lados = pd.DataFrame({h: lado_original * 40 / h for h in alturas})
    fig, (tamanho, rotulos, memoria, intervalo, integral) = plt.subplots(
        1, 5, figsize=(8.4, 2.72), width_ratios=[2.55, 1.22, 1.78, 1.78, .6],
        gridspec_kw={"wspace": .16})
    fig.subplots_adjust(left=.062, right=.99, top=.79, bottom=.22)
    rotulos.set(xlim=(0, 1), ylim=(len(lados) - .5, -.5))
    rotulos.axis("off")
    for i, (classe, valores) in enumerate(lados.iterrows()):
        nome, cor, marca = estilos[classe]
        tamanho.plot(alturas, valores, color=cor, marker=marca, lw=1.3, ms=3.7,
                     markerfacecolor="white", markeredgewidth=1)
        rotulos.plot(.06, i, marker=marca, color=cor, markerfacecolor="white", ms=4)
        rotulos.text(.19, i, nome, va="center", fontsize=8.3)
    tamanho.set_yscale("log", base=2)
    tamanho.set(xticks=alturas, yticks=[16, 32, 64, 128, 256],
                yticklabels=["16", "32", "64", "128", "256"], ylim=(13, 270),
                ylabel="Lado médio (px)")
    tamanho.grid(axis="y", lw=.5, color=".87")
    fig.text(tamanho.get_position().x0 + tamanho.get_position().width/2, .93,
             "(a) Caixas anotadas", ha="center", fontsize=9.1)

    cores = LinearSegmentedColormap.from_list("recall", ["#f4f7f8", "#9bbbc9", "#264b69"])
    for ax, metodo, titulo in [(memoria, metodos[0], titulos[0]), (intervalo, metodos[1], titulos[1]),
                               (integral, "integral", "Integral")]:
        valores = resumo.loc[resumo.metodo == metodo].pivot(
            index="classe", columns="altura_m", values="recall_pct").loc[lados.index]
        ax.imshow(valores, cmap=cores, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        for (i, j), valor in np.ndenumerate(valores.to_numpy()):
            ax.text(j, i, numero(valor, 1), va="center", ha="center", fontsize=7.5,
                    color="white" if valor >= 72 else "#192933")
        ax.set(xticks=range(len(valores.columns)), xticklabels=valores.columns, yticks=[])
        ax.tick_params(axis="x", length=0, pad=4)
        ax.spines[["left", "bottom", "right", "top"]].set_visible(False)
        ax.set_title(titulo, fontsize=8.3, pad=4)
    x0, x1 = memoria.get_position().x0, integral.get_position().x1
    fig.text((x0+x1)/2, .93, "(b) Recall geográfico por classe (%)", ha="center", fontsize=9.1)
    fig.text(.5, .045, "Altura equivalente (m)", ha="center", fontsize=8)
    return fig


def main():
    r,t = dados_completos()
    perdas = perdas_pareadas(t)
    tmp = DESTINO/"perdas_pareadas_por_classe.partial.csv"
    perdas.to_csv(tmp,index=False)
    tmp.replace(DESTINO/"perdas_pareadas_por_classe.csv")
    classes = resultados_por_classe(t)
    tmp = DESTINO / "tamanho_recall_por_classe.partial.csv"
    classes.to_csv(tmp, index=False, float_format="%.6f")
    tmp.replace(DESTINO / "tamanho_recall_por_classe.csv")
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42}):
        fig = figura(r)
        tmp = DESTINO/"desempenho_por_altura.partial.pdf"
        fig.savefig(tmp,bbox_inches="tight",pad_inches=.06)
        plt.close(fig)
        tmp.replace(DESTINO/"desempenho_por_altura.pdf")
        fig = figura_por_classe(classes)
        tmp = DESTINO / "tamanho_recall_por_classe.partial.pdf"
        fig.savefig(tmp, bbox_inches="tight", pad_inches=.06)
        plt.close(fig)
        tmp.replace(DESTINO / "tamanho_recall_por_classe.pdf")
    texto = ["# Escala dos recipientes e inferência integral", "",
             "Resultados dos seis voos completos, com 1.924 referências das seis classes. "
             "Cobertura e intervalo usaram 37.776 recortes por método e condição. "
             "Os 40 m correspondem às imagens originais. As alturas maiores representam uma simulação do tamanho "
             "aparente nas mesmas regiões e nos mesmos instantes.", "",
             "| Método | Altura (m) | Localizados | Recall (%) | Δ recall vs. 40 m (p.p.) | Falsos alarmes | Alertas até 1 s | Mediana do atraso (s) | P90 (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for linha in r.sort_values(["metodo","altura_m"]).itertuples():
        texto.append("| " + " | ".join([NOMES[linha.metodo],str(linha.altura_m),numero(linha.vp,0),
            numero(100*linha.recall,2),numero(linha.delta_recall_pp,2),numero(linha.fp,0),
            numero(linha.confirmados_ate_1s,0),numero(linha.mediana_s,2),numero(linha.p90_s,2)])+" |")
    texto.extend(["", "A referência integral usa 1.131.600 recortes nos 47.150 quadros originais. "
                  "O atraso é contado desde a primeira aparição anotada até a primeira confirmação visual. "
                  "A mediana e o P90 usam as trilhas confirmadas por cada condição. A contagem até um segundo inclui "
                  "todas as trilhas no denominador. Os falsos alarmes e o recall são avaliados no mapa ao fim do voo.", "",
                  "A simulação conserva a área pontuada e a seleção. Ela mede o efeito de escala e do contexto "
                  "na imagem. O desempenho em um voo real a outra altura também depende da área nova que a câmera "
                  "passa a ver, da trajetória e das condições de aquisição.", ""])
    tmp = DESTINO/"resultado.partial.md"
    tmp.write_text("\n".join(texto),encoding="utf-8")
    tmp.replace(DESTINO/"resultado.md")
    print("Resultados dos seis voos conferidos e consolidados em",DESTINO,flush=True)


if __name__ == "__main__":
    main()
