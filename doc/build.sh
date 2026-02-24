#!/usr/bin/env bash
# build.sh - Compila o TCC completo
# Uso: ./build.sh [clean]

MAIN="USPSC-TCC-modelo-EESC"
BUILDDIR="build"

if [ "$1" = "clean" ]; then
    rm -rf "$BUILDDIR"
    find . -name "*.aux" -not -path "./.git/*" -delete
    echo "Arquivos temporários removidos."
    exit 0
fi

mkdir -p "$BUILDDIR"

latexmk -pdf -auxdir="$BUILDDIR" -outdir="$BUILDDIR" -interaction=nonstopmode "$MAIN.tex"

if [ -f "$BUILDDIR/$MAIN.pdf" ]; then
    echo ""
    echo "PDF gerado com sucesso: $BUILDDIR/$MAIN.pdf"
else
    echo ""
    echo "Erro na compilação. Verifique $BUILDDIR/$MAIN.log"
    exit 1
fi
