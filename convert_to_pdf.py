"""Convert the academic paper markdown to PDF using pandoc + xelatex.

Requires: pandoc, xelatex (MiKTeX or TeX Live), Georgia/Consolas/Cambria Math fonts.
Uses a custom LaTeX preamble (pandoc_preamble.tex) for styling.
"""

import subprocess
import os
import sys

INPUT_FILE = "mmda-dynamic-beta-academic-paper-v2.md"
OUTPUT_FILE = "mmda-dynamic-beta-academic-paper-v2.pdf"
PREAMBLE = "pandoc_preamble.tex"

print(f"Converting {INPUT_FILE} -> {OUTPUT_FILE} via pandoc + xelatex ...")

cmd = [
    "pandoc", INPUT_FILE,
    "-o", OUTPUT_FILE,
    "--pdf-engine=xelatex",
    "-V", "geometry:margin=1in",
    "-V", "fontsize=11pt",
    "-V", 'mainfont=Georgia',
    "-V", 'monofont=Consolas',
    "-V", 'mathfont=Cambria Math',
    "-V", "linestretch=1.3",
    "-V", "colorlinks=true",
    "-V", "linkcolor=NavyBlue",
    f"--include-in-header={PREAMBLE}",
    "--resource-path=.",
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("STDERR:", result.stderr)
    sys.exit(1)

fsize = os.path.getsize(OUTPUT_FILE)
print(f"Done. {OUTPUT_FILE} ({fsize:,} bytes)")
