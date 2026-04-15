#!/usr/bin/env python3
"""
Replace literal strings inside PDF content streams (e.g. plot legends) without
regenerating figures in MATLAB. Works on the merged PDFs under paper/data/.

Uses pypdf (already used elsewhere in this repo). Mutates each stream object
referenced by /Contents (array or single indirect).

Default: RRSACPSO -> SACPSO in paper/data/rlbased and paper/data/others3chrismast.

  python3 tools/rename_pdf_legend_strings.py --inplace
  python3 tools/rename_pdf_legend_strings.py -d paper/data/rlbased --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, IndirectObject

# Longer keys first if you add multi-token replacements later.
DEFAULT_REPLACEMENTS: tuple[tuple[bytes, bytes], ...] = (
    (b"RRSACPSO", b"SACPSO"),
    (b"(RRSACPSO)", b"(SACPSO)"),
)


def _content_stream_objects(page) -> list:
    raw = page.get("/Contents")
    if raw is None:
        return []
    if isinstance(raw, IndirectObject):
        raw = raw.get_object()
    if isinstance(raw, ArrayObject):
        return [
            ref.get_object() if isinstance(ref, IndirectObject) else ref
            for ref in raw
        ]
    if hasattr(raw, "get_data") and hasattr(raw, "set_data"):
        return [raw]
    return []


def replace_in_pdf(input_path: str, output_path: str, replacements: tuple[tuple[bytes, bytes], ...]) -> int:
    """Return number of content streams whose bytes were changed."""
    reader = PdfReader(input_path)
    writer = PdfWriter()
    modified = 0
    for page in reader.pages:
        for stream in _content_stream_objects(page):
            if not hasattr(stream, "get_data") or not hasattr(stream, "set_data"):
                continue
            try:
                data = stream.get_data()
            except Exception:
                continue
            new_data = data
            for old_b, new_b in replacements:
                if old_b:
                    new_data = new_data.replace(old_b, new_b)
            if new_data != data:
                stream.set_data(new_data)
                modified += 1
        writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)
    return modified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-replace legend bytes inside PDF content streams (pypdf)."
    )
    parser.add_argument(
        "-d",
        "--dir",
        action="append",
        default=[],
        help="Directory of PDFs (repeatable). Default: paper/data/rlbased + others3chrismast.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite each PDF (writes via a temp file).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write to a temp output and report changes; do not modify originals.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    default_dirs = [
        os.path.join(repo_root, "paper", "data", "rlbased"),
        os.path.join(repo_root, "paper", "data", "others3chrismast"),
    ]
    dirs = [os.path.abspath(x) for x in (args.dir or default_dirs)]

    pdf_paths: list[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"Skip (not a directory): {d}", file=sys.stderr)
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(d, name))

    if not pdf_paths:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    total_streams = 0
    for path in pdf_paths:
        if args.dry_run:
            fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix="dry_")
            os.close(fd)
            try:
                n = replace_in_pdf(path, tmp, DEFAULT_REPLACEMENTS)
                if n:
                    print(f"would modify: {path} ({n} stream(s))")
                    total_streams += n
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        elif args.inplace:
            fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix="legend_")
            os.close(fd)
            try:
                n = replace_in_pdf(path, tmp, DEFAULT_REPLACEMENTS)
                if n:
                    os.replace(tmp, path)
                    print(f"updated: {path} ({n} stream(s))")
                    total_streams += n
                else:
                    os.remove(tmp)
            except Exception:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        else:
            base, ext = os.path.splitext(path)
            out = f"{base}_renamed{ext}"
            n = replace_in_pdf(path, out, DEFAULT_REPLACEMENTS)
            if n:
                print(f"wrote: {out} ({n} stream(s))")
            total_streams += n

    print(f"Done. Streams touched: {total_streams}")


if __name__ == "__main__":
    main()
