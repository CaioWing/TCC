"""Converte o eixo de densidade da figura vetorial existente para trilhas/km².

O PDF original é preservado em doc/dados para permitir a reprodução. Como a
conversão é linear, multiplicar todos os valores do eixo por 100 preserva a
geometria exata das barras e dos demais painéis, sem reamostrar a figura.
Dependências: pymupdf, matplotlib, pypdf.
"""

from pathlib import Path
import shutil

import pymupdf as fitz
from matplotlib.font_manager import findfont
from padronizar_cores_figuras import standardize_pdf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "doc/figuras/fig_mbgv2_anotacoes.pdf"
SOURCE = ROOT / "doc/dados/fig_mbgv2_anotacoes_original.pdf"


def main():
    if not SOURCE.exists():
        shutil.copy2(OUTPUT, SOURCE)
    doc = fitz.open(SOURCE)
    page = doc[0]
    spans = [span for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", []) for span in line["spans"]]
    label = next(s for s in spans if s["text"] == "trilhas por hectare imageado")
    ticks = [s for s in spans if s["text"] in {"0", "20", "40", "60", "80", "100", "120"}
             and 60 < s["bbox"][0] < 75 and s["bbox"][1] < 155]
    assert len(ticks) == 7, "O eixo original não corresponde à figura esperada."

    for span in [label, *ticks]:
        page.add_redact_annot(fitz.Rect(span["bbox"]), fill=False)
    page.apply_redactions(images=0, graphics=0)

    font_path = str(findfont("DejaVu Sans"))
    font = fitz.Font(fontfile=font_path)
    page.insert_font(fontname="DensityUnits", fontfile=font_path, set_simple=True)
    for span in ticks:
        value = f'{int(span["text"]) * 100:,}'.replace(",", ".")
        width = font.text_length(value, fontsize=span["size"])
        page.insert_text((73.02 - width, span["origin"][1]), value,
                         fontname="DensityUnits", fontsize=span["size"])

    value = "trilhas por km² imageado"
    center_y = (label["bbox"][1] + label["bbox"][3]) / 2
    width = font.text_length(value, fontsize=label["size"])
    page.insert_text((43, center_y + width / 2), value,
                     fontname="DensityUnits", fontsize=label["size"], rotate=90)
    doc.subset_fonts()
    doc.save(OUTPUT, garbage=4, deflate=True)
    doc.close()
    standardize_pdf(OUTPUT)

    with fitz.open(OUTPUT) as result:
        text = result[0].get_text()
        assert "hectare" not in text
        assert "trilhas por km² imageado" in text and "12.000" in text
        result[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
            ROOT / "tmp/revisao/anotacoes-km2.png")
    print(OUTPUT)


if __name__ == "__main__":
    main()
