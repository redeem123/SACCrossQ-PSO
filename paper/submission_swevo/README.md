# Full Submission Bundle

This folder contains the full journal submission package.

Included:
- the standalone manuscript bundle
- `COVER_LETTER.md`
- `HIGHLIGHTS.md`

Compile with:

```sh
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
