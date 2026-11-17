"""Method diagram for the paper (Figure fig:method in paper/sections/method.tex).

The drawing is TikZ source in paper/figures/fig_method_src.tex, so that text and mathematics use the paper's own fonts
(TeX Gyre Pagella text, Computer Modern math). This script checks that the colours in that file match the muted palette
of scripts/paper_style.py, compiles it with pdflatex in a temporary folder, copies the PDF to
paper/figures/fig_method.pdf, renders paper/figures/fig_method.png with PyMuPDF and checks the size and that every font
is embedded. The diagram contains no measured data: its numbers (d+ = 700, d- = 68, 2,000 images, 780 steps) are those
stated in paper/sections/method.tex.

    .venv/Scripts/python.exe scripts/make_paper_fig_method.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_style import C  # noqa: E402

FIG = ROOT / "paper" / "figures"
SRC = FIG / "fig_method_src.tex"
PDF = FIG / "fig_method.pdf"
PNG = FIG / "fig_method.png"
WIDTH_IN, MAX_HEIGHT_IN = 6.0, 2.1


def check_palette(tex: str) -> None:
    """Every \\definecolor{p<name>}{HTML}{...} (and ink, ink2, grid) must equal paper_style.C[<name>]."""
    for name, hexval in re.findall(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}", tex):
        key = name[1:] if name.startswith("p") and name[1:] in C else name
        want = C[key].lstrip("#").upper()
        if hexval.upper() != want:
            raise SystemExit(f"colour {name} = #{hexval} does not match paper_style.C['{key}'] = #{want}")


def main() -> None:
    check_palette(SRC.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={tmp}",
                              str(SRC)], cwd=tmp, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-3000:])
            raise SystemExit("pdflatex failed")
        for line in res.stdout.splitlines():
            if "Overfull" in line or "Underfull" in line or "Warning" in line:
                print("pdflatex:", line)
        shutil.copyfile(Path(tmp) / "fig_method_src.pdf", PDF)

    doc = fitz.open(PDF)
    page = doc[0]
    w_in, h_in = page.rect.width / 72, page.rect.height / 72
    print(f"page: {w_in:.3f} x {h_in:.3f} in")
    if abs(w_in - WIDTH_IN) > 0.01 or h_in > MAX_HEIGHT_IN + 1e-6:
        raise SystemExit(f"size {w_in:.3f} x {h_in:.3f} in is outside {WIDTH_IN} x <= {MAX_HEIGHT_IN} in")
    fonts = page.get_fonts(full=True)
    for f in fonts:
        xref, ext, ftype, base = f[0], f[1], f[2], f[3]
        print(f"font: {base} ({ftype}, {'embedded' if ext and ext != 'n/a' else 'NOT embedded'})")
        if not ext or ext == "n/a":
            raise SystemExit(f"font {base} is not embedded")
    page.get_pixmap(dpi=300, alpha=False).save(PNG)
    doc.close()
    print("wrote", PDF, "and", PNG)


if __name__ == "__main__":
    main()
