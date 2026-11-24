"""Compara as regras por célula e por faixa na mesma passagem do voo 10."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dados import CACHE, ROOT, folder
from figura_exemplo_temporal import fotos_do_video
from figuras_malha import amostra_de_celulas, contagem, legenda, painel
from memoria_sim import T, grade_projecao, simulate
from multivistas import MemoriaFaixas, altura_seguinte


def quadro_da_celula(raster, t, celula):
    j0, _, i0, _, dentro, u, v = raster(t)
    j, i = celula[0] - j0, celula[1] - i0
    if 0 <= j < dentro.shape[0] and 0 <= i < dentro.shape[1] and dentro[j, i]:
        return np.array([u[j, i], v[j, i]])
    return None


def saidas(n, tiles, shape, raster, altura, exigir_saida_unica=False, selecoes=None):
    # Escolher uma célula que de fato disparou a saída nas duas faixas.
    memoria = MemoriaFaixas(n, tiles, altura)
    anterior = {}
    tocou_topo = np.zeros(shape, dtype=bool)
    tocou_base = np.zeros(shape, dtype=bool)
    atual = raster(0)
    for t in range(n - 1):
        seguinte = raster(t + 1)
        j0, j1, i0, i1, dentro, u, v = atual
        regiao = np.s_[j0:j1, i0:i1]
        tocou_topo[regiao] |= dentro & (v < 40)
        tocou_base[regiao] |= dentro & (v > altura - 40)
        faixa = v >= altura / 2
        prox = altura_seguinte(atual, seguinte)
        saindo = dentro & ((prox < 0) | (faixa != (prox >= altura / 2)))
        if memoria.maps:
            estado = memoria.maps[2][j0:j1, i0:i1]
            saindo &= (estado & np.where(faixa, 2, 1)) == 0
        # No exemplo comparativo a mesma célula precisa disparar a regra de uma faixa.
        pendente_unica = None
        if exigir_saida_unica and memoria.maps:
            pendente_unica = memoria.maps[1][regiao] == 0
        memoria.frame(t, atual, seguinte, shape)
        for j, i in zip(*np.nonzero(saindo)):
            if pendente_unica is not None and prox[j, i] < 0 and not pendente_unica[j, i]:
                continue
            chave = (j0 + j, i0 + i)
            if chave in anterior and tocou_topo[chave] and tocou_base[chave] and prox[j, i] < 0:
                inicio = anterior[chave]
                origem = quadro_da_celula(raster, inicio, chave)
                if abs(origem[1] - altura / 2) < 40 and min(v[j, i], altura-v[j, i]) < 40:
                    for quadro, (px, py) in [(inicio, origem), (t, (u[j, i], v[j, i]))]:
                        a, b = tiles.T
                        cobre = (px >= a) & (px < a+T) & (py >= b) & (py < b+T)
                        assert (memoria.selected[2][quadro] & cobre).any()
                    if selecoes is not None:
                        for q, pixel in ((inicio, origem), (t, (u[j, i], v[j, i]))):
                            px, py = pixel
                            cobre = (px >= a) & (px < a + T) & (py >= b) & (py < b + T)
                            for k in (1, 2):
                                selecoes[k, q] = [tuple(tile) for tile in tiles[memoria.selected[k][q] & cobre]]
                    return inicio, t, chave
            if prox[j, i] >= 0:
                anterior[chave] = t
        atual = seguinte
    raise ValueError("Nenhuma célula com saídas verificadas nas duas faixas.")


VIDEO = "video10_regionE"
# Célula da travessia do voo 10 usada nos dois exemplos (linha norte, coluna leste).
CELULA = (164, 421)


def dados_dos_paineis():
    with np.load(CACHE / f"{VIDEO}.npz") as dados:
        tamanho, grade = dados["size"], dados["tiles"]
        n, fps, gsd = int(dados["total_frames"]), float(dados["fps"]), float(dados["gsd"])
    w, h = tamanho
    log, = (folder(VIDEO) / "log").glob("*.csv")
    forma, raster = grade_projecao(log, n, fps, w, h, gsd)
    entrada, divisa, saida = 1286, 1391, 1495
    assert quadro_da_celula(raster, entrada-1, CELULA) is None
    assert quadro_da_celula(raster, saida+1, CELULA) is None
    assert quadro_da_celula(raster, divisa, CELULA)[1] < h/2 <= quadro_da_celula(raster, divisa+1, CELULA)[1]

    # As máscaras do simulador incluem também envios motivados por outras células.
    indices = {tuple(tile): i for i, tile in enumerate(grade)}
    selecionados = {m: np.zeros((saida+1, len(grade)), bool) for m in ("imediata", "adiada")}
    def registrar(metodo, t, x, y):
        if t <= saida:
            selecionados[metodo][t, indices[x, y]] = True
    simulate(log, n, fps, w, h, gsd, on_selected=registrar)

    fotos = dict(zip((entrada, divisa, saida), fotos_do_video((entrada, divisa, saida), VIDEO)))
    alvos = {("entrada", entrada), ("entrada", saida), ("uma", entrada), ("uma", divisa),
             ("uma", saida), ("duas", divisa), ("duas", saida)}
    memoria = MemoriaFaixas(n, grade, h)
    vistas_entrada = np.zeros(forma, dtype=bool)
    paineis = {}
    atual = raster(0)
    for t in range(saida + 1):
        seguinte = raster(t+1)
        j0, j1, i0, i1, dentro, u, v = atual
        regiao = np.s_[j0:j1, i0:i1]
        if t in fotos:
            prox_v = altura_seguinte(atual, seguinte)
            pedidos = {"entrada": dentro & ~vistas_entrada[regiao],
                       "uma": dentro & (prox_v < 0) & (memoria.maps[1][regiao] == 0)}
            bit = np.where(v < h/2, 1, 2)
            sai_da_faixa = (prox_v < 0) | ((prox_v < h/2) != (v < h/2))
            pedidos["duas"] = dentro & sai_da_faixa & ((memoria.maps[2][regiao] & bit) == 0)
            anteriores = {"entrada": int(vistas_entrada[CELULA]), "uma": int(memoria.maps[1][CELULA]),
                          "duas": int(memoria.maps[2][CELULA]).bit_count()}
        memoria.frame(t, atual, seguinte, forma)
        enviados_entrada = selecionados["imediata"][t]
        for x, y in grade[enviados_entrada]:
            vistas_entrada[regiao] |= dentro & (x <= u) & (u < x+T) & (y <= v) & (v < y+T)
        if t in fotos:
            for nome, enviados in [("entrada", enviados_entrada), ("uma", memoria.selected[1][t]),
                                   ("duas", memoria.selected[2][t])]:
                if (nome, t) in alvos:
                    outros, ativos, pixel, aciona = amostra_de_celulas(atual, pedidos[nome], grade, enviados, CELULA)
                    paineis[nome, t] = (grade[enviados], outros, ativos, pixel, aciona, anteriores[nome])
        atual = seguinte
    np.testing.assert_array_equal(memoria.selected[1][:saida+1], selecionados["adiada"])
    esperados = {("entrada", entrada): (True, 0), ("entrada", saida): (False, 1),
                 ("uma", entrada): (False, 0), ("uma", divisa): (False, 0), ("uma", saida): (True, 0),
                 ("duas", divisa): (True, 0), ("duas", saida): (True, 1)}
    for chave, estado in esperados.items():
        assert paineis[chave][-2:] == estado
    recortes, _, _, (u, v), _, _ = paineis["duas", divisa]
    x, y = recortes.T
    assert ((x <= u) & (u < x+T) & (y <= v) & (v < y+T)).sum() == 2
    np.testing.assert_array_equal(paineis["uma", saida][0], paineis["duas", saida][0])
    assert round((saida - entrada) / fps, 2) == 6.97
    return tamanho, fotos, paineis


def desenhar(por_faixa):
    tamanho, fotos, estados = dados_dos_paineis()
    if por_faixa:
        quadros = [(1391, "saída da metade superior"), (1495, "saída da imagem")]
        metodos = [("uma", "Uma faixa"), ("duas", "Duas faixas")]
        arquivo = "fig_saida_faixas"
    else:
        quadros = [(1286, "entrada de C"), (1495, "saída de C")]
        metodos = [("entrada", "Entrada da célula"), ("uma", "Saída da célula")]
        arquivo = "fig_entrada_saida_celula"
    fig, eixos = plt.subplots(2, 2, figsize=(7.15, 5.05))
    fig.subplots_adjust(left=.008, right=.992, top=.84, bottom=.055, wspace=.065, hspace=.40)
    legenda(fig, "Célula")
    for coluna, (q, evento) in enumerate(quadros):
        ax = eixos[0, coluna]
        ax.text(0, 1.19, f"Quadro {q} · {evento}", transform=ax.transAxes,
                fontsize=9.5, fontweight="semibold", va="bottom")
        for linha, (nome, titulo) in enumerate(metodos):
            ax = eixos[linha, coluna]
            recortes, outros, ativos, pixel, aciona, n_anteriores = estados[nome, q]
            painel(ax, fotos[q], tamanho, recortes, outros, ativos, pixel, "C", duas_faixas=nome == "duas")
            ax.set_title(f"{titulo} · {contagem(len(recortes))}", fontsize=8.7, loc="left", pad=4)
            if aciona:
                texto = f"C pede a {n_anteriores + 1}ª vista"
            elif n_anteriores:
                texto = "C já recebeu sua vista"
            else:
                texto = "C ainda aguarda a saída"
            ax.text(.5, -.035, texto, transform=ax.transAxes, fontsize=8.5, ha="center", va="top")
            print(f"{nome}, quadro {q}: {contagem(len(recortes))}; {texto}")
    fig.savefig(ROOT / f"doc/figuras/{arquivo}.pdf", bbox_inches="tight", pad_inches=.055, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        desenhar(por_faixa=True)
