PYTHON ?= python3
PDFTOPPM ?= pdftoppm
PREVIEW_DPI ?= 120
LATEXRUN = ../.build/latexrun.py
SOURCES = main.tex references.bib $(wildcard figures/*.png figures/*.pdf figures/*.jpg)

.PHONY: all previews view clean

all: main.pdf

# latexrun runs pdflatex and BibTeX as many times as needed for references.
main.pdf: $(SOURCES) $(LATEXRUN)
	$(PYTHON) $(LATEXRUN) -O . --latex-args="-interaction=nonstopmode -halt-on-error" main.tex

# Render every PDF page, replacing old previews only after a successful render.
previews: main.pdf
	@command -v $(PDFTOPPM) >/dev/null || { echo "Install Poppler for pdftoppm (macOS: brew install poppler)."; exit 1; }
	@mkdir -p previews
	@set -e; preview_tmp=$$(mktemp -d); trap 'rm -rf "$$preview_tmp"' EXIT HUP INT TERM; \
		$(PDFTOPPM) -r $(PREVIEW_DPI) -png main.pdf "$$preview_tmp/page"; \
		rm -f previews/page-*.png; \
		mv "$$preview_tmp"/page-*.png previews/

view: main.pdf
	open main.pdf

# Keep the published PDF and page previews; remove only intermediate files.
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.rel *.synctex.gz .latexrun.db*
