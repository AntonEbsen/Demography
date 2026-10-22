"""Reproduce the submitted paper's Figure 4.1 from read-only source workbooks.

Run from exam_project2: python -m src.validation.figure_4_1
This does not rebuild the rolling analysis panel or run any regressions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
PAPER_SHA256 = "086a7d333c7a9b85f6476e90de30f4f3cfd76d1720996ca0aebfb532803d5e41"
START, END = 1867, 1890


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_workbook(directory: Path, stem: str) -> Path:
    matches = [
        p
        for p in directory.iterdir()
        if p.stem.casefold() == stem.casefold() and p.suffix.casefold() in {".xls", ".xlsx"}
    ]
    if not matches:
        raise ValueError(f"Missing {stem} workbook")
    # Match the production loader's .xlsx preference, but verify conversions.
    matches.sort(key=lambda p: (p.suffix != ".xlsx", p.suffix != ".XLS", p.name))
    if len(matches) > 1:
        reference = pd.read_excel(matches[0])
        for alternative in matches[1:]:
            try:
                pd.testing.assert_frame_equal(
                    reference, pd.read_excel(alternative), check_dtype=False, check_exact=True
                )
            except AssertionError as exc:
                raise ValueError(f"Conflicting workbook variants for {stem}") from exc
    return matches[0]


def canonical(frame: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Independent column matching, not the production normalisation helper."""
    mapping = {str(col).strip().casefold(): col for col in frame.columns}
    if len(mapping) != len(frame.columns):
        raise ValueError("Ambiguous column names after case normalisation")
    missing = [col for col in required if col.casefold() not in mapping]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame.rename(columns={mapping[col.casefold()]: col for col in required})


def prepare_sample(religion: pd.DataFrame, vital: pd.DataFrame):
    """Match county membership, then restrict actual Year values, without imputation."""
    rel = canonical(religion, ["Code", "Type", "Kreis", "Pop", "Relcathm", "Relcathf"])
    vit = canonical(vital, ["Code", "Type", "Year"])
    rel = rel.loc[(rel["Code"] < 900) & (rel["Type"] == 0)].copy()
    vit = vit.loc[(vit["Code"] < 900) & (vit["Type"] == 0)].copy()
    if rel["Code"].duplicated().any():
        raise ValueError("REL1871 contains duplicate county identifiers")
    values = rel[["Pop", "Relcathm", "Relcathf"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (rel["Pop"] <= 0).any():
        raise ValueError("Missing/non-finite religion counts or non-positive population")
    rel["cath_share"] = 100 * (rel["Relcathm"] + rel["Relcathf"]) / rel["Pop"]
    if not rel["cath_share"].between(0, 100).all():
        raise ValueError("Catholic share is outside 0-100 percent")
    merged = vit.merge(rel[["Code", "cath_share"]], on="Code", how="inner", validate="many_to_one")
    inside = merged["Year"].between(START, END)
    panel = merged.loc[inside].copy()
    if panel.empty or panel.duplicated(["Code", "Year"]).any():
        raise ValueError("Empty sample or duplicate county-year keys")
    if set(panel["Year"]) != set(range(START, END + 1)):
        raise ValueError("The requested calendar-year window is incomplete")
    counties = rel.loc[rel["Code"].isin(panel["Code"]), ["Code", "Kreis", "cath_share"]]
    counties = counties.sort_values("Code").reset_index(drop=True)
    excluded = rel.loc[~rel["Code"].isin(panel["Code"]), ["Code", "Kreis"]]
    out_columns = [col for col in ["Code", "Year", "_source_file"] if col in merged]
    diagnostics = {
        "religion_counties_before_matching": len(rel),
        "matched_county_years": len(panel),
        "counties_without_all_24_years": int((panel.groupby("Code")["Year"].nunique() < 24).sum()),
        "excluded_religion_counties": excluded.to_dict("records"),
        "excluded_out_of_window_records": merged.loc[~inside, out_columns].to_dict("records"),
    }
    return counties, diagnostics


def summarise(counties: pd.DataFrame) -> dict:
    if counties["Code"].duplicated().any():
        raise ValueError("Distribution requires one observation per county")
    shares = counties["cath_share"]
    if shares.empty or not np.isfinite(shares).all() or not shares.between(0, 100).all():
        raise ValueError("Invalid county shares")
    counts, edges = np.histogram(shares, bins=30)
    return {
        "counties": len(shares),
        "low_catholic_le_50": int((shares <= 50).sum()),
        "high_catholic_gt_50": int((shares > 50).sum()),
        "unweighted_mean_percent": float(shares.mean()),
        "unweighted_median_percent": float(shares.median()),
        "counties_between_20_and_80_inclusive": int(shares.between(20, 80).sum()),
        "percent_counties_between_20_and_80_inclusive": float(100 * shares.between(20, 80).mean()),
        "counties_exactly_at_20_or_80": int(shares.isin([20, 80]).sum()),
        "histogram_counts": counts.tolist(),
        "histogram_edges_percent": edges.tolist(),
    }


def paper_checks(metrics: dict, diagnostics: dict) -> dict:
    """Compare counts exactly and stated means/medians at manuscript precision."""
    return {
        "figure_counties_392": metrics["counties"] == 392,
        "figure_low_counties_262": metrics["low_catholic_le_50"] == 262,
        "figure_high_counties_130": metrics["high_catholic_gt_50"] == 130,
        "prose_mean_rounds_to_34_percent": round(metrics["unweighted_mean_percent"]) == 34,
        "prose_median_rounds_to_13_3_percent": round(metrics["unweighted_median_percent"], 1)
        == 13.3,
        "section_4_1_county_years_9261": diagnostics["matched_county_years"] == 9261,
        "section_4_1_incomplete_counties_33": diagnostics["counties_without_all_24_years"] == 33,
    }


def run(data_dir: Path, output_dir: Path, paper: Path | None = None) -> dict:
    inputs = [find_workbook(data_dir, "REL1871")]
    inputs += [find_workbook(data_dir, f"VIT{year}") for year in range(START, END + 1)]
    # Hash selected files and every alternative conversion checked above.
    stems = {path.stem.casefold() for path in inputs}
    audited_inputs = sorted(
        path
        for path in data_dir.iterdir()
        if path.stem.casefold() in stems and path.suffix.casefold() in {".xls", ".xlsx"}
    )
    hashes_before = {path.name: sha256(path) for path in audited_inputs}
    if paper is not None and sha256(paper) != PAPER_SHA256:
        raise ValueError("The supplied PDF is not the recorded submitted-paper version")
    religion = pd.read_excel(inputs[0])
    vital_parts = []
    for path in inputs[1:]:
        part = pd.read_excel(path)
        part["_source_file"] = path.name
        vital_parts.append(part)
    # Normalise each workbook before concatenation: source casing varies by year.
    vital = pd.concat(
        [
            canonical(part, ["Code", "Type", "Year"])[["Code", "Type", "Year", "_source_file"]]
            for part in vital_parts
        ],
        ignore_index=True,
    )
    counties, diagnostics = prepare_sample(religion, vital)
    metrics = summarise(counties)
    checks = paper_checks(metrics, diagnostics)
    if not all(checks.values()):
        raise AssertionError(f"Submitted-paper checks failed: {checks}")

    # Independently computed shares must agree with the existing source loader.
    from src.data.load_data import load_rel1871

    production = load_rel1871(inputs[0]).set_index("Code")["cath_share"]
    np.testing.assert_allclose(
        counties["cath_share"], counties["Code"].map(production), rtol=0, atol=1e-12
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.visualization.plots import plot_cath_distribution

    fig, ax = plot_cath_distribution(counties)
    ax.set_title(
        "Distribution of Catholic shares across Prussian counties (1871)",
        fontsize=12,
        fontweight="bold",
    )
    np.testing.assert_array_equal(
        [patch.get_height() for patch in ax.patches], metrics["histogram_counts"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure_4_1_reproduced.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    hashes_after = {path.name: sha256(path) for path in audited_inputs}
    if hashes_before != hashes_after:
        raise AssertionError("A raw input changed during validation")
    report = {
        "target": "Submitted paper, Figure 4.1, PDF page 9; related prose pages 8-10",
        "reference_pdf_sha256": PAPER_SHA256,
        "actual_year_window": [START, END],
        "status": "figure_reproduced_with_prose_discrepancy",
        "metrics": metrics,
        "sample_diagnostics": diagnostics,
        "matched_manuscript_checks": checks,
        "known_prose_discrepancy": {
            "manuscript_approximate_percent": 22,
            "recomputed_percent": metrics["percent_counties_between_20_and_80_inclusive"],
            "note": (
                "92/392 = 23.47%, about 1.47 percentage points above the prose approximation. "
                "No observations are exactly at either endpoint. PDF unchanged."
            ),
        },
        "scope": (
            "County membership, distribution and stated sample counts only. "
            "Not a validation of regression estimates, causal identification, "
            "or every later repository output."
        ),
    }
    manifest = {
        "selected_workbooks": [path.name for path in inputs],
        "variant_policy": (
            "Prefer .xlsx, then .XLS, then .xls; require exact cell equality "
            "(ignoring dtype) for alternatives."
        ),
        "raw_inputs_sha256": hashes_after,
        "raw_inputs_unchanged_during_run": True,
        "code_sha256": {
            str(path.relative_to(PROJECT)): sha256(path)
            for path in [
                Path(__file__).resolve(),
                PROJECT / "src/data/load_data.py",
                PROJECT / "src/visualization/plots.py",
            ]
        },
    }
    environment = {
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "matplotlib", "xlrd", "openpyxl"]
        },
    }
    for name, payload in [
        ("report.json", report),
        ("input_manifest.json", manifest),
        ("environment.json", environment),
    ]:
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/raw/galloway_data")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "outputs/validation/figure_4_1"
    )
    parser.add_argument(
        "--paper", type=Path, help="Optional local submitted PDF; its SHA-256 must match"
    )
    args = parser.parse_args()
    report = run(args.data_dir, args.output_dir, args.paper)
    print(
        "Figure 4.1 reproduced: 392 counties, 262 low / 130 high; "
        "all seven matched manuscript checks passed."
    )
    print(
        "Prose discrepancy: 20-80% interval contains 92/392 counties = "
        f"{report['metrics']['percent_counties_between_20_and_80_inclusive']:.2f}%, "
        "not approximately 22%."
    )
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
