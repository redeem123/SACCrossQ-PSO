# Manuscript Bundle

This folder contains the standalone manuscript package for journal compilation.

Included:
- `main.tex`
- `references.bib`
- `elsarticle.cls`
- `elsarticle-num.bst`
- only the figure assets referenced by `main.tex`
- `main.pdf` as the latest local build

Compile:

```sh
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
