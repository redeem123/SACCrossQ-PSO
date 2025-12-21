#!/usr/bin/env python3
"""Extract citation keys in order of first appearance from LaTeX file."""

import re
import sys

def extract_citations(tex_file):
    """Extract all citation keys in order from LaTeX file."""
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all \cite{...} commands
    cite_pattern = r'\\cite\{([^}]+)\}'
    matches = re.findall(cite_pattern, content)

    # Split multiple citations and track order
    seen = set()
    ordered_keys = []

    for match in matches:
        # Split by comma for multiple citations in one \cite{}
        keys = [k.strip() for k in match.split(',')]
        for key in keys:
            if key and key not in seen:
                seen.add(key)
                ordered_keys.append(key)

    return ordered_keys

if __name__ == '__main__':
    tex_file = 'main.tex'
    citation_order = extract_citations(tex_file)

    print(f"Total unique citations: {len(citation_order)}\n")
    for i, key in enumerate(citation_order, 1):
        print(f"{i:3d}. {key}")
