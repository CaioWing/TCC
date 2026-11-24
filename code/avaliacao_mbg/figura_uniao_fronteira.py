"""Mostra as duas partes e sua caixa reunida no mesmo quadro real da MBG-V2."""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from dados import OUT, ROOT, folder
from figura_passagem import pontos_no_quadro
from figuras_malha import BORDA, VISIVEL
from memoria_sim import T, tiles_grid
from metricas import box_ious
from paleta_tcc import BLUE, GREEN, ORANGE
from passagens import DESTINO, MASCARAS, PLANOS, carregar_voo, melhor_recorte


def pontos_sem_envio(caso, tamanho, grade):
    video, quadro = caso["video"], caso["quadro"]
    pose, tamanho_voo, grade_voo, _ = carregar_voo(video)
    np.testing.assert_array_equal(tamanho_voo, tamanho)
    np.testing.assert_array_equal(grade_voo, grade)
    eventos = pd.read_csv(DESTINO / "eventos.csv").query("video == @video")
    pontos = pontos_no_quadro(eventos, pose, tamanho_voo, quadro)
    pedidos = pontos[[f"quadro_{marco}" for marco in PLANOS["tres"]]].eq(quadro).any(axis=1)
    recortes = [melhor_recorte(p.u, p.v, grade_voo) for p in pontos[pedidos].itertuples()]
    with np.load(MASCARAS / f"{video}.npz") as mascaras:
        enviados = np.flatnonzero(mascaras["tres"][quadro])
    np.testing.assert_array_equal(np.unique(recortes), enviados)
    # Este quadro ilustra a ausência dos dois recortes no método de três vistas.
    assert len(pontos) == 20 and not enviados.size
    return pontos


def desenhar():
    caso = json.loads((OUT / "fronteiras/exemplo.json").read_text(encoding="utf-8"))
    arquivo, = (folder(caso["video"]) / "avi").glob("*.avi")
    captura = cv2.VideoCapture(str(arquivo))
    try:
        captura.set(cv2.CAP_PROP_POS_FRAMES, caso["quadro"])
        ok, quadro = captura.read()
        if not ok or round(captura.get(cv2.CAP_PROP_POS_FRAMES)) != caso["quadro"] + 1:
            raise IOError(f"Falha ao ler {caso['video']}, quadro {caso['quadro']}")
    finally:
        captura.release()
    altura, largura = quadro.shape[:2]
    grade = tiles_grid(largura, altura)
    pontos = pontos_sem_envio(caso, (largura, altura), grade)
    eixo = 0 if caso["eixo"] == "vertical" else 1
    costura = grade[caso["recorte_b"]][eixo]
    uniao = np.array(caso["uniao"])
    borda = round(max(uniao[2:] - uniao[:2]) * .17)
    x1, y1 = np.maximum(np.floor(uniao[:2]).astype(int) - borda, 0)
    x2, y2 = np.minimum(np.ceil(uniao[2:]).astype(int) + borda, (largura, altura))
    detalhe = cv2.cvtColor(quadro[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    caixas = np.array([caso[chave] for chave in ("a", "b", "uniao")])
    ious = box_ious(caixas, np.array([caso["anotacao"]]))[:, 0]
    np.testing.assert_allclose(ious, [caso["iou_a"], caso["iou_b"], caso["iou_uniao"]], atol=1e-5)

    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        fig, axs = plt.subplots(1, 3, figsize=(7.3, 1.91), gridspec_kw={"width_ratios": [1.1, 1, 1]})
        fig.subplots_adjust(left=.005, right=.995, top=.89, bottom=.15, wspace=.075)
        contexto = axs[0]
        reduzido = cv2.resize(quadro, (largura//4, altura//4), interpolation=cv2.INTER_AREA)
        contexto.imshow(cv2.cvtColor(reduzido, cv2.COLOR_BGR2RGB), extent=(0, largura, altura, 0))
        contexto.scatter(pontos.u, pontos.v, marker="o", s=13, facecolor=VISIVEL,
                         edgecolor=BORDA, linewidth=.7, zorder=3)
        efeitos = [patheffects.withStroke(foreground=BORDA, linewidth=2.)]
        contexto.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec="white", lw=1.2,
                                     path_effects=efeitos, zorder=4))
        origem = grade[caso["recorte_b"]][1-eixo]
        limite = [origem, origem+T]
        if eixo == 0:
            contexto.plot([costura, costura], limite, color="white", ls="--", lw=.85, path_effects=efeitos)
        else:
            contexto.plot(limite, [costura, costura], color="white", ls="--", lw=.85, path_effects=efeitos)
        contexto.set(xlim=(0, largura), ylim=(altura, 0))
        contexto.set_title("(a) Geográfica · um intermediário", loc="left", fontsize=8.4, pad=4)
        contexto.set_axis_off()
        paineis = [("(b) Integral · caixas de cada parte", [("a", BLUE), ("b", ORANGE)]),
                   ("(c) Integral · caixa reunida", [("uniao", GREEN)])]
        for ax, (titulo, caixas) in zip(axs[1:], paineis):
            ax.imshow(detalhe, extent=(x1, x2, y2, y1), interpolation="none")
            for chave, cor in caixas:
                u, v, u2, v2 = caso[chave]
                box = Rectangle((u, v), u2 - u, v2 - v, fill=False, color=cor, linewidth=2.4)
                box.set_path_effects([patheffects.withStroke(foreground="white", linewidth=4.)])
                ax.add_patch(box)
            linha = ax.axvline(costura) if eixo == 0 else ax.axhline(costura)
            linha.set(color="white", linestyle=(0, (3.5, 3)), linewidth=1.1)
            linha.set_path_effects([patheffects.withStroke(foreground=".18", linewidth=2.)])
            ax.set(xlim=(x1, x2), ylim=(y2, y1))
            ax.set_title(titulo, loc="left", fontsize=8.4, pad=4)
            ax.set_axis_off()
        for ax, texto in zip(axs, [f"{len(pontos)} pontos visíveis · 0 recortes",
                                    f"IoU das partes · {ious[0]:.3f} e {ious[1]:.3f}",
                                    f"IoU após a união · {ious[2]:.3f}"]):
            ax.text(.5, -.055, texto.replace(".", ","), transform=ax.transAxes, ha="center", va="top",
                    color=".2", fontsize=8.2)
        fig.savefig(ROOT / "doc/figuras/fig_uniao_fronteira.pdf", bbox_inches="tight", pad_inches=.045, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    desenhar()
