MAIN     = USPSC-TCC-modelo-EESC
BUILDDIR = build
LATEX    = pdflatex -interaction=nonstopmode -output-directory=$(BUILDDIR)
BIBTEX   = bibtex

.PHONY: all clean

all: $(BUILDDIR)
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(BUILDDIR)/$(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "PDF gerado: $(BUILDDIR)/$(MAIN).pdf"

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
	find . -name "*.aux" -not -path "./.git/*" -delete
