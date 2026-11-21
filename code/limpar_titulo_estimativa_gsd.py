"""Retira a fórmula do título da figura legada, preservando os vetores e imagens."""
from pathlib import Path
import pymupdf

path = Path(__file__).resolve().parents[1] / 'doc/figuras/fig_estimativa_gsd.pdf'
doc = pymupdf.open(path)
page = doc[0]
spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
         for line in block['lines'] for span in line['spans']
         if span['text'].startswith('Onde cada quadro')]
if spans:
    assert len(spans) == 1
    span = spans[0]
    rect = pymupdf.Rect(span['bbox'])
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0)
    page.insert_text((rect.x0, rect.y1-1.8), "Projeções da caixa d'água no solo",
                     fontname='helv', fontsize=span['size'], color=(0,0,0))
    output = path.with_name('fig_estimativa_gsd.limpo.pdf')
    doc.save(output, garbage=4, deflate=True)
    doc.close()
    output.replace(path)
    print(path)
else:
    doc.close()
    print('Título já simplificado.')
