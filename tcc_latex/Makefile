MAIN = USPSC-TCC-modelo-EESC
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

.PHONY: all clean

all:
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.idx *.lof *.log *.lot *.loq *.toc \
	      USPSC-TA-PreTextual/*.aux \
	      USPSC-TA-Textual/*.aux \
	      USPSC-TA-PosTextual/*.aux \
	      $(MAIN).pdf
