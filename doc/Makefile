MAIN     = USPSC-TCC-modelo-EESC
BUILDDIR = build

.PHONY: all clean

all:
	mkdir -p $(BUILDDIR)
	latexmk -pdf -auxdir=$(BUILDDIR) -outdir=$(BUILDDIR) -interaction=nonstopmode -f $(MAIN).tex
	@echo ""
	@echo "PDF gerado: $(BUILDDIR)/$(MAIN).pdf"

clean:
	rm -rf $(BUILDDIR)
	find . -name "*.aux" -not -path "./.git/*" -delete
