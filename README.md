## Building the Paper

### Prerequisites
- A TeX distribution ([TeX Live](https://tug.org/texlive/), [MacTeX](https://tug.org/mactex/), or [MiKTeX](https://miktex.org/))

### Compile
cd Graph_ML_Project
make view        # build and open PDF
# or manually:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
