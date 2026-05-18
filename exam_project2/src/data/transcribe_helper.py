"""
transcribe_helper.py
====================
CLI helpers for the Preußische Statistik transcription workflow.

Typical usage from inside an AI-assist session:

    # 1. Convert a downloaded source PDF page range to PNGs (so the
    #    assistant can read the table image).
    python -m src.data.transcribe_helper render \\
        --pdf path/to/PS_Heft_121_1.pdf \\
        --first-page 130 --last-page 140 \\
        --outdir /tmp/ps121

    # 2. After the assistant returns transcribed numbers, validate and
    #    insert them into the per-year template.
    python -m src.data.transcribe_helper insert \\
        --year 1890 \\
        --code 1 --kreis MEMEL --mw15_49 7430 \\
        --code 2 --kreis FISCHHAUSEN --mw15_49 6512 \\
        --source-page "p. 134"

    # 3. List which Kreise are still unfilled.
    python -m src.data.transcribe_helper status --year 1890

Helper functions also exposed for import from notebooks.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_DIR = REPO_ROOT / "data" / "raw" / "transcribed_marital_status"


def render_pdf_pages(
    pdf_path: Path,
    first_page: int,
    last_page: int,
    outdir: Path,
    resolution: int = 200,
) -> list[Path]:
    """Render PDF pages to PNG via pdftoppm (poppler).

    Returns the list of generated PNG paths. The pdftoppm binary must be
    on PATH; on Windows it ships with TeX Live (texlive/bin/win32/) and
    on Linux with the poppler-utils package.
    """
    outdir = outdir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / pdf_path.stem
    cmd = [
        "pdftoppm",
        "-r", str(resolution),
        "-png",
        "-f", str(first_page),
        "-l", str(last_page),
        str(pdf_path),
        str(stem),
    ]
    subprocess.run(cmd, check=True)
    pages = sorted(outdir.glob(f"{pdf_path.stem}-*.png"))
    return pages


def insert_transcriptions(
    year: int,
    entries: Iterable[dict],
    template_dir: Path = TEMPLATE_DIR,
) -> pd.DataFrame:
    """Insert one or more {Code, mwf_15_49_direct, source_page, notes}
    records into the MAR{year} template CSV. Preserves any cells already
    filled in (only overwrites the columns specified per entry).

    Each entry is a dict with at minimum a ``Code`` and one of the
    transcription columns. Returns the updated DataFrame.
    """
    path = template_dir / f"MAR{year}_transcription_template.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    # Read with comment-aware parsing.
    with open(path, encoding="utf-8") as f:
        header_lines = []
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
            else:
                break
    df = pd.read_csv(path, comment="#")

    n_changed = 0
    for entry in entries:
        code = int(entry["Code"])
        row_mask = df["Code"] == code
        if not row_mask.any():
            logger.warning("Code %d not in MAR%d template -- skipped", code, year)
            continue
        for col, val in entry.items():
            if col == "Code":
                continue
            if col not in df.columns:
                logger.warning("Column %r not in template -- skipped", col)
                continue
            df.loc[row_mask, col] = val
        n_changed += 1

    # Write back, preserving the comment header.
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.writelines(header_lines)
        df.to_csv(f, index=False)
    logger.info("Inserted/updated %d row(s) in %s", n_changed, path.name)
    return df


def transcription_status(
    year: int,
    template_dir: Path = TEMPLATE_DIR,
) -> dict:
    """Summarise progress on a single year's template."""
    path = template_dir / f"MAR{year}_transcription_template.csv"
    if not path.exists():
        return {"year": year, "exists": False}
    df = pd.read_csv(path, comment="#")
    total = len(df)
    direct = int(df["mwf_15_49_direct"].notna().sum()) if "mwf_15_49_direct" in df.columns else 0
    band_cols = [
        "mwf_15_19", "mwf_20_24", "mwf_25_29", "mwf_30_34",
        "mwf_35_39", "mwf_40_44", "mwf_45_49",
    ]
    bands_complete = (
        int(df[band_cols].notna().all(axis=1).sum())
        if all(c in df.columns for c in band_cols) else 0
    )
    filled = int(((df["mwf_15_49_direct"].notna() if "mwf_15_49_direct" in df.columns else False)
                  | df[band_cols].notna().all(axis=1)).sum())
    return {
        "year": year,
        "exists": True,
        "total_kreise": total,
        "direct_cells_filled": direct,
        "band_complete": bands_complete,
        "any_filled": filled,
        "remaining": total - filled,
    }


def _cli() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(prog="transcribe_helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_render = sub.add_parser("render", help="Render PDF pages to PNG")
    p_render.add_argument("--pdf", type=Path, required=True)
    p_render.add_argument("--first-page", type=int, required=True)
    p_render.add_argument("--last-page", type=int, required=True)
    p_render.add_argument("--outdir", type=Path, required=True)
    p_render.add_argument("--resolution", type=int, default=200)

    p_insert = sub.add_parser("insert", help="Insert transcribed cells")
    p_insert.add_argument("--year", type=int, required=True)
    p_insert.add_argument("--code", type=int, action="append", required=True)
    p_insert.add_argument("--mw15_49", type=int, action="append", required=True)
    p_insert.add_argument("--kreis", type=str, action="append", default=None)
    p_insert.add_argument("--source-page", type=str, default=None)

    p_status = sub.add_parser("status", help="Show transcription progress")
    p_status.add_argument("--year", type=int, default=None,
                          help="Show one year; omit for all years")

    args = p.parse_args()

    if args.cmd == "render":
        paths = render_pdf_pages(args.pdf, args.first_page, args.last_page,
                                 args.outdir, args.resolution)
        for path in paths:
            print(path)
        return 0

    if args.cmd == "insert":
        if len(args.code) != len(args.mw15_49):
            print("--code and --mw15_49 must be specified the same number of times",
                  file=sys.stderr)
            return 2
        entries = []
        for i, code in enumerate(args.code):
            entry = {"Code": code, "mwf_15_49_direct": args.mw15_49[i]}
            if args.source_page:
                entry["source_page"] = args.source_page
            entries.append(entry)
        insert_transcriptions(args.year, entries)
        return 0

    if args.cmd == "status":
        years = [args.year] if args.year else [1880, 1885, 1890, 1895, 1900, 1905, 1910]
        rows = [transcription_status(y) for y in years]
        # Pretty-print as a table.
        print(f"{'Year':>6}  {'Total':>6}  {'Filled':>6}  {'Remaining':>9}")
        for r in rows:
            if not r.get("exists"):
                print(f"{r['year']:>6}  {'-':>6}  {'-':>6}  {'-':>9}  (template missing)")
                continue
            print(f"{r['year']:>6}  {r['total_kreise']:>6}  "
                  f"{r['any_filled']:>6}  {r['remaining']:>9}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(_cli())
