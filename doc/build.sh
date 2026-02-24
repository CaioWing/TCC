#!/usr/bin/env bash
# build.sh - Compila o TCC completo
# Uso: ./build.sh [clean]

MAIN="USPSC-TCC-modelo-EESC"
BUILDDIR="build"
LATEX="pdflatex -interaction=nonstopmode -output-directory=$BUILDDIR"
BIBTEX="bibtex"

if [ "$1" = "clean" ]; then
    rm -rf "$BUILDDIR"
    find . -name "*.aux" -not -path "./.git/*" -delete
    echo "Arquivos temporários removidos."
    exit 0
fi

mkdir -p "$BUILDDIR"

echo "[1/4] Primeira passagem pdflatex..."
$LATEX "$MAIN.tex"

echo "[2/4] BibTeX..."
$BIBTEX "$BUILDDIR/$MAIN"

echo "[3/4] Segunda passagem pdflatex..."
$LATEX "$MAIN.tex"

echo "[4/4] Terceira passagem pdflatex..."
$LATEX "$MAIN.tex"

if [ -f "$BUILDDIR/$MAIN.pdf" ]; then
    echo ""
    echo "PDF gerado com sucesso: $BUILDDIR/$MAIN.pdf"
else
    echo ""
    echo "Erro na compilação. Verifique $BUILDDIR/$MAIN.log"
    exit 1
fi
