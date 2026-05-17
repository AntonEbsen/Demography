"""
Export all project notebooks into a single combined PDF (and Markdown
backup), suitable for uploading to an AI assistant for review.

By default this:
  * concatenates every `notebooks/[0-9]*.ipynb` in lexicographic order;
  * **omits the code cells' source** but keeps the *outputs* (text,
    tables, figures);
  * keeps every markdown narrative cell;
  * writes the result to `outputs/for_ai/notebooks_combined.{md,html,pdf}`.

The script tries PDF backends in order and falls back gracefully:

  1. `nbconvert --to webpdf`  (uses Playwright/Chromium if installed)
  2. `nbconvert --to pdf`     (uses LaTeX; needs xelatex on PATH)
  3. `quarto render`          (uses Quarto if installed; very robust)
  4. HTML-only — the script then opens the HTML in your default browser
     so you can print-to-PDF manually (always works).

The Markdown file is *always* produced — many AI assistants accept .md
uploads and parse them more cleanly than PDFs.

Usage
-----
    python scripts/export_notebooks_for_ai.py
    python scripts/export_notebooks_for_ai.py --include-code
    python scripts/export_notebooks_for_ai.py --notebooks notebooks/03_*.ipynb
    python scripts/export_notebooks_for_ai.py --format md
    python scripts/export_notebooks_for_ai.py --format html

Designed to be one-file, no new pip installs assumed beyond `nbconvert`
(which the project already depends on).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Notebook merging
# ---------------------------------------------------------------------------

def merge_notebooks(nb_paths: list[Path]) -> "nbformat.NotebookNode":
    """Concatenate notebooks in `nb_paths` order, prepending an H1
    section header before each. Strips per-cell ``id`` collisions
    (nbformat 4.5+ requires unique cell IDs across the whole document).
    """
    merged = nbformat.v4.new_notebook()
    merged.metadata = {
        "kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    seen_ids: set[str] = set()
    for nb_path in nb_paths:
        nb = nbformat.read(nb_path, as_version=4)
        header_md = (
            f"\n\n# {nb_path.stem.replace('_', ' ').title()}\n\n"
            f"_(rendered from `{nb_path.name}`)_\n\n---\n"
        )
        header_cell = nbformat.v4.new_markdown_cell(header_md)
        # Page break before each notebook (only rendered by HTML/PDF).
        header_cell.metadata = {"raw_mimetype": "text/html"}
        merged.cells.append(header_cell)

        for cell in nb.cells:
            cid = cell.get("id")
            if cid and cid in seen_ids:
                cell["id"] = f"{nb_path.stem}_{cid}"
            seen_ids.add(cell.get("id", ""))
            merged.cells.append(cell)
    return merged


# ---------------------------------------------------------------------------
# 2. Per-format exporters
# ---------------------------------------------------------------------------

def export_html(nb, out_path: Path, include_code: bool) -> None:
    from nbconvert import HTMLExporter
    from traitlets.config import Config

    c = Config()
    c.HTMLExporter.exclude_input = not include_code
    c.HTMLExporter.exclude_input_prompt = True
    c.HTMLExporter.exclude_output_prompt = True
    body, _ = HTMLExporter(config=c).from_notebook_node(nb)
    out_path.write_text(body, encoding="utf-8")


def export_markdown(nb, out_path: Path, include_code: bool) -> None:
    from nbconvert import MarkdownExporter
    from traitlets.config import Config

    c = Config()
    c.MarkdownExporter.exclude_input = not include_code
    body, _ = MarkdownExporter(config=c).from_notebook_node(nb)
    out_path.write_text(body, encoding="utf-8")


# ---- PDF strategies ----

def _strip_inputs_to_tempnb(nb, include_code: bool) -> Path:
    """Write a tempfile copy with inputs hidden so the various
    command-line tools render only narrative + outputs."""
    nb_copy = nbformat.from_dict(nb)  # deep copy via dict round-trip
    if not include_code:
        for cell in nb_copy.cells:
            if cell.get("cell_type") == "code":
                cell["metadata"] = {**cell.get("metadata", {}), "jupyter": {"source_hidden": True}}
    fd, tmp = tempfile.mkstemp(suffix=".ipynb")
    os.close(fd)
    nbformat.write(nb_copy, tmp)
    return Path(tmp)


def _pdf_via_nbconvert_webpdf(nb, out_path: Path, include_code: bool) -> bool:
    """Headless-Chromium-based PDF via nbconvert (best fidelity)."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        try:
            import pyppeteer  # noqa: F401
        except ImportError:
            return False
    tmp = _strip_inputs_to_tempnb(nb, include_code)
    try:
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "webpdf",
            "--no-input" if not include_code else "--stdout",
            "--output", str(out_path.with_suffix("")),
            str(tmp),
        ]
        # Remove the dummy --stdout flag when including code
        cmd = [c for c in cmd if c != "--stdout"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and out_path.exists()
    finally:
        tmp.unlink(missing_ok=True)


def _pdf_via_nbconvert_latex(nb, out_path: Path, include_code: bool) -> bool:
    """LaTeX-based PDF via nbconvert. Needs xelatex on PATH."""
    if shutil.which("xelatex") is None and shutil.which("pdflatex") is None:
        return False
    tmp = _strip_inputs_to_tempnb(nb, include_code)
    try:
        cmd = [
            sys.executable, "-m", "nbconvert",
            "--to", "pdf",
            "--output", str(out_path.with_suffix("")),
        ]
        if not include_code:
            cmd.append("--no-input")
        cmd.append(str(tmp))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [latex stderr tail] ...{result.stderr[-400:]}", file=sys.stderr)
        return result.returncode == 0 and out_path.exists()
    finally:
        tmp.unlink(missing_ok=True)


def _pdf_via_quarto(nb, out_path: Path, include_code: bool) -> bool:
    """Quarto's notebook -> PDF pipeline. Very robust on Windows."""
    if shutil.which("quarto") is None:
        return False
    tmp = _strip_inputs_to_tempnb(nb, include_code)
    try:
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "quarto", "render", str(tmp),
            "--to", "pdf",
            "--output", out_path.name,
            "--output-dir", str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [quarto stderr tail] ...{result.stderr[-400:]}", file=sys.stderr)
        return result.returncode == 0 and out_path.exists()
    finally:
        tmp.unlink(missing_ok=True)


def export_pdf(nb, out_path: Path, include_code: bool) -> bool:
    """Try all PDF backends in order, return True on first success."""
    strategies = [
        ("nbconvert webpdf (Chromium)",   _pdf_via_nbconvert_webpdf),
        ("nbconvert pdf (LaTeX)",         _pdf_via_nbconvert_latex),
        ("quarto render",                 _pdf_via_quarto),
    ]
    for name, fn in strategies:
        print(f"  trying {name}...", flush=True)
        try:
            if fn(nb, out_path, include_code):
                print(f"  -> success via {name}")
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"  ({name} raised: {exc})")
    return False


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--notebooks", nargs="+", default=None,
        help="Specific notebook paths (default: notebooks/[0-9]*.ipynb).",
    )
    ap.add_argument(
        "--out-dir", default="outputs/for_ai",
        help="Output directory (default: outputs/for_ai).",
    )
    ap.add_argument(
        "--name", default="notebooks_combined",
        help="Output filename stem (default: notebooks_combined).",
    )
    ap.add_argument(
        "--include-code", action="store_true",
        help="Include code-cell source in addition to outputs.",
    )
    ap.add_argument(
        "--format", choices=["md", "html", "pdf", "all"], default="all",
        help="Which formats to write (default: all three).",
    )
    ap.add_argument(
        "--open-html", action="store_true",
        help="If PDF rendering fails, open the HTML so you can print-to-PDF.",
    )
    args = ap.parse_args(argv)

    # Resolve notebook list
    if args.notebooks:
        nb_paths = [Path(p).resolve() for p in args.notebooks]
    else:
        nb_paths = sorted((PROJECT_ROOT / "notebooks").glob("[0-9]*.ipynb"))
    if not nb_paths:
        print("ERROR: no notebooks found.", file=sys.stderr)
        return 1

    print(f"Merging {len(nb_paths)} notebook(s):")
    for p in nb_paths:
        try:
            print(f"  - {p.relative_to(PROJECT_ROOT)}")
        except ValueError:
            print(f"  - {p}")
    print(f"Code cells: {'INCLUDED' if args.include_code else 'omitted (outputs kept)'}")

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name

    merged = merge_notebooks(nb_paths)

    md_path = out_dir / f"{stem}.md"
    html_path = out_dir / f"{stem}.html"
    pdf_path = out_dir / f"{stem}.pdf"

    if args.format in ("md", "all"):
        print(f"\nWriting Markdown...")
        export_markdown(merged, md_path, args.include_code)
        print(f"  {md_path.relative_to(PROJECT_ROOT)}  ({md_path.stat().st_size // 1024} KB)")

    if args.format in ("html", "all"):
        print(f"\nWriting HTML...")
        export_html(merged, html_path, args.include_code)
        print(f"  {html_path.relative_to(PROJECT_ROOT)}  ({html_path.stat().st_size // 1024} KB)")

    if args.format in ("pdf", "all"):
        print(f"\nRendering PDF...")
        ok = export_pdf(merged, pdf_path, args.include_code)
        if ok:
            print(f"  {pdf_path.relative_to(PROJECT_ROOT)}  ({pdf_path.stat().st_size // 1024} KB)")
        else:
            print(
                "\nNo PDF backend succeeded. Options:\n"
                "  * Open the HTML and Ctrl-P -> Save as PDF (always works).\n"
                "  * Install one backend:\n"
                "      pip install playwright && playwright install chromium\n"
                "      (or)  install MiKTeX/TeX Live (you appear to have TeX Live\n"
                "      2022 on PATH, but xelatex must be reachable from this shell)\n",
                file=sys.stderr,
            )
            if args.open_html and html_path.exists():
                print(f"Opening {html_path} in your browser...")
                if sys.platform.startswith("win"):
                    os.startfile(str(html_path))  # type: ignore[attr-defined]
                else:
                    subprocess.run(["xdg-open", str(html_path)])
            return 2

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
