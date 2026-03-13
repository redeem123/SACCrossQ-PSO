#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Cleaning transient caches and LaTeX build artifacts under: $repo_root"

rm -rf "$repo_root/scripts/__pycache__"
rm -rf "$repo_root/tools/__pycache__"

rm -f "$repo_root/sage_latex_template_4_unzipped/main.aux"
rm -f "$repo_root/sage_latex_template_4_unzipped/main.bbl"
rm -f "$repo_root/sage_latex_template_4_unzipped/main.blg"
rm -f "$repo_root/sage_latex_template_4_unzipped/main.log"
rm -f "$repo_root/sage_latex_template_4_unzipped/main.out"

rm -f "$repo_root/paper/main.aux"
rm -f "$repo_root/paper/main.bbl"
rm -f "$repo_root/paper/main.blg"
rm -f "$repo_root/paper/main.log"
rm -f "$repo_root/paper/main.out"
rm -f "$repo_root/paper/main.spl"

echo "Done."
