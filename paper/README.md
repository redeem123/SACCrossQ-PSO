# paper/

Active manuscript and submission workspace for the AFSACPSO paper.

## Current Status

- `main.tex` is the active Elsevier submission draft.
- `main.pdf` is the latest local build of that draft.
- `COVER_LETTER.md` and `HIGHLIGHTS.md` are the current submission-side documents.
- `afsacpso_uav/` stores outline and paper-design notes.
- `results/paper_artifacts/afsacpso_uav/manifest.json` is the reproducibility manifest referenced by the manuscript.

## Supporting Files

| Path | Purpose |
|---|---|
| `main_backup.tex` | Earlier single-author manuscript snapshot |
| `main_backup2.tex` | Additional manuscript backup |
| `Background_RelatedWork.tex` | Supporting text fragment |
| `paper_review.md` | Internal review notes |
| `data/` | Figures and table-side source assets used by `main.tex` |

## Build

```sh
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

If the bibliography has not changed, the `bibtex` step can be skipped for quick rebuilds.
