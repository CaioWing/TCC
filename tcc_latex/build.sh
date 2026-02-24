#!/usr/bin/env bash
# build.sh - Compila o TCC completo
# Uso: ./build.sh [clean]

MAIN="USPSC-TCC-modelo-EESC"
LATEX="pdflatex -interaction=nonstopmode"
BIBTEX="bibtex"

if [ "$1" = "clean" ]; then
    rm -f *.aux *.bbl *.blg *.idx *.lof *.log *.lot *.loq *.toc \
          USPSC-TA-PreTextual/*.aux \
          USPSC-TA-Textual/*.aux \
          USPSC-TA-PosTextual/*.aux \
          "$MAIN.pdf"
    echo "Arquivos temporários removidos."
    exit 0
fi

echo "[1/4] Primeira passagem pdflatex..."
$LATEX "$MAIN.tex"

echo "[2/4] BibTeX..."
$BIBTEX "$MAIN"

echo "[3/4] Segunda passagem pdflatex..."
$LATEX "$MAIN.tex"

echo "[4/4] Terceira passagem pdflatex..."
$LATEX "$MAIN.tex"

if [ -f "$MAIN.pdf" ]; then
    echo ""
    echo "PDF gerado com sucesso: $MAIN.pdf"
else
    echo ""
    echo "Erro na compilação. Verifique $MAIN.log"
    exit 1
fi
