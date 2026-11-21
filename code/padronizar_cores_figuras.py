"""Padroniza cores vetoriais das figuras legadas, sem alterar as fotografias.

Substitui apenas operadores RGB de traço/preenchimento nos PDFs incluídos no
TCC. Não modifica coordenadas, texto, transparências ou pixels das imagens.
É idempotente e pode ser reaplicado depois de regenerar uma figura legada.
"""
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject

from paleta_tcc import BLUE, ORANGE, GREEN, PURPLE, RED, GRAY


ROOT = Path(__file__).resolve().parents[1]


def rgb(color):
    return tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5))


# Mantém a mesma correspondência de classes entre as figuras da base MBG-V2.
MAPPING = {rgb(old): rgb(new) for old, new in {
    "#1f77b4": BLUE, "#ff7f0e": ORANGE, "#2ca02c": GREEN,
    "#d62728": RED, "#9467bd": PURPLE, "#8c564b": GRAY,
    "#3a6ea5": BLUE, "#c07a2c": ORANGE,
    "#2b6cb0": BLUE, "#7fb3d5": BLUE, "#7b4ea3": PURPLE,
    "#c0692a": ORANGE, "#ffd400": ORANGE,
}.items()}


def standardize_pdf(path):
    reader = PdfReader(path)
    writer = PdfWriter(clone_from=reader)
    visited = set()
    replacements = 0

    def transform(stream, resources):
        nonlocal replacements
        content = ContentStream(stream, writer)
        for operands, operator in content.operations:
            if operator not in (b"rg", b"RG"):
                continue
            values = tuple(float(v) for v in operands)
            for old, new in MAPPING.items():
                if max(abs(a-b) for a, b in zip(values, old)) < 1e-6:
                    operands[:] = [FloatObject(v) for v in new]
                    replacements += 1
                    break
        for reference in resources.get("/XObject", {}).values():
            obj = reference.get_object()
            if obj.get("/Subtype") == "/Form" and id(obj) not in visited:
                visited.add(id(obj))
                updated = transform(obj, obj.get("/Resources", resources))
                obj.set_data(updated.get_data())
        return content

    for page in writer.pages:
        page.replace_contents(transform(page.get_contents(), page["/Resources"]))
    if replacements:
        temporary = path.with_suffix(".paleta.pdf")
        writer.write(temporary)
        writer.close()
        reader.close()
        temporary.replace(path)
    else:
        writer.close()
        reader.close()
    return replacements


def main():
    sources = "\n".join(p.read_text(encoding="utf-8")
                        for p in (ROOT / "doc/USPSC-TA-Textual").glob("*.tex"))
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", sources)))
    for figure in figures:
        path = ROOT / "doc" / figure
        count = standardize_pdf(path)
        print(f"{path.name}: {count} cores vetoriais atualizadas")


if __name__ == "__main__":
    main()
