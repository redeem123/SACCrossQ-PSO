# tools/

Local third-party binaries and configuration — **not part of the algorithm code**.

| Item | Purpose |
|---|---|
| `poppler/` | PDF command-line utilities (pdftotext, pdfimages). Used by paper build scripts on Windows. Git-ignored due to size. |
| `pdftotext.bat` | Windows batch wrapper for Poppler's `pdftotext`. |
| `fonts.conf` | Fontconfig configuration file for Poppler on Windows. |

These tools are only needed for PDF processing on Windows. macOS/Linux users can install Poppler via Homebrew (`brew install poppler`) or apt.
