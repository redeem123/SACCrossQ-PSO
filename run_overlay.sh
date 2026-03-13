#!/bin/bash
set -euo pipefail

echo "Restoring originals..."
cp results/final/rlbased/*.pdf paper/data/rlbased/
cp results/final/others3chrismast/*.pdf paper/data/others3chrismast/

echo "Clipping defaults..."
python3 tools/clip3D.py --inplace --dir paper/data/rlbased/
python3 tools/clip3D.py --inplace --dir paper/data/others3chrismast/
python3 tools/clip_topview.py --inplace --dir paper/data/rlbased/
python3 tools/clip_topview.py --inplace --dir paper/data/others3chrismast/

echo "Applying custom Scenario 4 clips..."
python3 -c "
import os
from pypdf import PdfReader, PdfWriter

def clip(path, l, r):
    reader = PdfReader(path)
    writer = PdfWriter()
    for page in reader.pages:
        mb = page.mediabox
        page.mediabox.left = float(mb.left) + l
        page.mediabox.right = float(mb.right) - r
        writer.add_page(page)
    with open(path, 'wb') as f:
        writer.write(f)

for d in ['rlbased', 'others3chrismast']:
    f_3d = f'paper/data/{d}/Scenario4_3D.pdf'
    if os.path.exists(f_3d):
        clip(f_3d, -80, -90)

    f_tv = f'paper/data/{d}/Scenario4_TopView.pdf'
    if os.path.exists(f_tv):
        clip(f_tv, 15, 5)
"

echo "Overlaying legends..."
python3 tools/overlay_legend.py --scale 2.0 --position top-right --x-offset 10 --y-offset 10 --legend paper/data/rlbased/legend.pdf paper/data/rlbased/Scenario1_3D.pdf paper/data/rlbased/Scenario2_3D.pdf paper/data/rlbased/Scenario3_3D.pdf paper/data/rlbased/Scenario4_3D.pdf
python3 tools/overlay_legend.py --scale 2.0 --position top-right --x-offset 10 --y-offset 10 --legend paper/data/others3chrismast/legend.pdf paper/data/others3chrismast/Scenario1_3D.pdf paper/data/others3chrismast/Scenario2_3D.pdf paper/data/others3chrismast/Scenario3_3D.pdf paper/data/others3chrismast/Scenario4_3D.pdf
python3 tools/overlay_legend.py --scale 2.0 --position top-right --x-offset 10 --y-offset 10 --legend paper/data/rlbased/legend.pdf paper/data/rlbased/Scenario1_TopView.pdf paper/data/rlbased/Scenario2_TopView.pdf paper/data/rlbased/Scenario3_TopView.pdf
python3 tools/overlay_legend.py --scale 2.0 --position top-right --x-offset 10 --y-offset 10 --legend paper/data/others3chrismast/legend.pdf paper/data/others3chrismast/Scenario3_TopView.pdf
python3 tools/overlay_legend.py --scale 2.0 --position bottom-left --x-offset 80 --y-offset 200 --legend paper/data/others3chrismast/legend.pdf paper/data/others3chrismast/Scenario1_TopView.pdf
python3 tools/overlay_legend.py --scale 2.0 --position bottom-left --x-offset 100 --y-offset 100 --legend paper/data/others3chrismast/legend.pdf paper/data/others3chrismast/Scenario2_TopView.pdf
python3 tools/overlay_legend.py --scale 2.0 --position top-left --x-offset 10 --y-offset 10 --legend paper/data/rlbased/legend.pdf paper/data/rlbased/Scenario4_TopView.pdf
python3 tools/overlay_legend.py --scale 2.0 --position bottom-right --x-offset 10 --y-offset 10 --legend paper/data/others3chrismast/legend.pdf paper/data/others3chrismast/Scenario4_TopView.pdf

echo "Compiling LaTeX..."
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
