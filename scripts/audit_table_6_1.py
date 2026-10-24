"""Narrow, AI-assisted audit of submitted Table 6.1, not full-paper replication.

Rebuilds from raw sources; never reads a cached analysis panel or overwrites tables.
Run with Python 3.12 after installing requirements-table-6-1.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "exam_project2"
HERE = ROOT / "reproducibility" / "table-6-1"
SOURCE_FILES = [
    "src/data/build_dataset.py",
    "src/data/load_data.py",
    "src/data/merge_ipehd.py",
    "src/analysis/coale_indices.py",
    "src/analysis/regressions.py",
    "src/analysis/utils.py",
]
OUTCOMES = ["cbr", "gmfr_static_1871", "nonmarital_paper_b5", "general_marriage_rate"]
DESIGNS = ["twfe", "year_x_rb"]
# Transcribed from the unchanged submitted PDF, Table 6.1, printed page 15.
# Each entry: coefficient, county-clustered SE, N, within R2, pre-trends p.
PRINTED = {
    "twfe": [
        (-0.011, 0.003, 9255, -0.013, 0.000),
        (-0.023, 0.026, 9254, -0.009, 0.000),
        (-0.036, 0.006, 9255, -0.036, 0.051),
        (-0.011, 0.003, 9261, 0.036, 0.000),
    ],
    "year_x_rb": [
        (0.001, 0.005, 9255, 0.000, 0.001),
        (0.043, 0.041, 9254, 0.014, 0.002),
        (-0.021, 0.008, 9255, -0.012, 0.712),
        (-0.003, 0.002, 9261, 0.014, 0.842),
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(canonical=True) -> dict[str, str]:
    paths = [PROJECT / p for p in SOURCE_FILES]
    # Conservative inventory: also hash raw files not needed by the four outcomes.
    for folder in ["galloway_data", "ipehd_data"]:
        paths.extend(
            p
            for p in (PROJECT / "data/raw" / folder).rglob("*")
            if p.suffix.lower() in {".xls", ".xlsx", ".csv"}
        )
    output = {}
    for path in sorted(paths):
        content = path.read_bytes()
        if canonical and path.suffix.lower() in {".py", ".csv"}:
            content = content.replace(b"\r\n", b"\n")
        output[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
    return output


def verify_inputs(actual: dict, expected: dict) -> None:
    if actual != expected:
        differing = sorted(
            k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)
        )
        raise ValueError("Missing, added or changed inputs: " + ", ".join(differing))


def select_paper_sample(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.loc[panel.Year.between(1867, 1890)].copy()
    if panel.duplicated(["Code", "Year"]).any():
        raise ValueError("Non-unique county-year keys")
    # Paper Appendix B, equation B.5: contemporaneous unmarried women, NOT
    # the later code's static-1871-prevalence alternative. The source builder
    # flags six implausible CBR rows; use the same birth-data mask here.
    unmarried = panel.women_15_49 - panel.married_women_15_49
    panel["nonmarital_paper_b5"] = (
        panel.Birbastot.div(unmarried).mul(1000).where(unmarried > 0).mask(panel.cbr_flag)
    )
    return panel


def make_model(panel: pd.DataFrame, outcome: str, design: str, event_study=False):
    sub = panel.dropna(subset=[outcome, "cath_share", "Rb"]).copy()
    if not np.isfinite(sub[[outcome, "cath_share"]].to_numpy()).all():
        raise ValueError("Non-finite model values")
    sub = sub.set_index(["Code", "Year"])
    years = sub.index.get_level_values("Year")
    if event_study:
        X = pd.DataFrame(
            {
                f"year_{year}": sub.cath_share * (years == year)
                for year in sorted(set(years))
                if year != 1872
            },
            index=sub.index,
        )
    else:
        X = pd.DataFrame({"cath_share_x_post": sub.cath_share * (years >= 1873)}, index=sub.index)
    if design == "twfe":
        model = PanelOLS(sub[outcome], X, entity_effects=True, time_effects=True)
    elif design == "year_x_rb":
        other = pd.DataFrame(
            {"year_rb": years.astype(str) + "_" + sub.Rb.astype(str)}, index=sub.index
        )
        model = PanelOLS(sub[outcome], X, entity_effects=True, other_effects=other)
    else:
        raise ValueError(design)
    return model.fit(cov_type="clustered", cluster_entity=True), sub


def partial_out_coef(sub: pd.DataFrame, outcome: str, design: str) -> float:
    """Separate alternating-projection coefficient check; not a second SE estimator."""
    years = sub.index.get_level_values("Year")
    groups = [
        pd.factorize(sub.index.get_level_values("Code"))[0],
        pd.factorize(years if design == "twfe" else years.astype(str) + "_" + sub.Rb.astype(str))[
            0
        ],
    ]
    values = np.column_stack([sub[outcome].to_numpy(), sub.cath_share.to_numpy() * (years >= 1873)])
    for _ in range(10000):
        previous = values.copy()
        for group in groups:
            count = np.bincount(group)
            for col in range(2):
                means = np.bincount(group, weights=values[:, col]) / count
                values[:, col] -= means[group]
        if np.max(np.abs(values - previous)) < 1e-11:
            return float(values[:, 1] @ values[:, 0] / (values[:, 1] @ values[:, 1]))
    raise RuntimeError("Alternating projections did not converge")


def displayed_match(value: float, expected: float) -> bool:
    return f"{value:.3f}" == f"{expected:.3f}"


def audit(panel: pd.DataFrame) -> dict:
    rows = []
    for design in DESIGNS:
        for outcome, printed in zip(OUTCOMES, PRINTED[design]):
            result, sample = make_model(panel, outcome, design)
            event, _ = make_model(panel, outcome, design, event_study=True)
            terms = [f"year_{year}" for year in range(1867, 1872)]
            b = event.params.loc[terms].to_numpy()
            v = event.cov.loc[terms, terms].to_numpy()
            wald = float(b @ np.linalg.solve(v, b))
            independent = partial_out_coef(sample, outcome, design)
            recomputed = dict(
                coef=float(result.params.iloc[0]),
                se=float(result.std_errors.iloc[0]),
                n=int(result.nobs),
                r2=float(result.rsquared_within),
                pretrends_p=float(chi2.sf(wald, len(terms))),
            )
            expected = dict(zip(["coef", "se", "n", "r2", "pretrends_p"], printed))
            matches = {
                k: (recomputed[k] == v if k == "n" else displayed_match(recomputed[k], v))
                for k, v in expected.items()
            }
            delta = abs(independent - recomputed["coef"])
            if delta > 1e-9:
                raise ValueError("Independent coefficient check failed")
            rows.append(
                dict(
                    design=design,
                    outcome=outcome,
                    printed=expected,
                    recomputed=recomputed,
                    matches=matches,
                    counties=int(sample.index.get_level_values("Code").nunique()),
                    partial_out_coef=independent,
                    partial_out_abs_difference=delta,
                    pretrends_wald=wald,
                    pretrends_df=len(terms),
                )
            )
            print(f"{design}/{outcome}: {sum(matches.values())}/5 printed fields match", flush=True)
    return {
        "date": "2026-10-24",
        "table": "Submitted Table 6.1 (printed page 15)",
        "scope": (
            "Eight regressions and their pre-trends tests; "
            "not full-paper replication or causal validation"
        ),
        "paperSha256": "086a7d333c7a9b85f6476e90de30f4f3cfd76d1720996ca0aebfb532803d5e41",
        "specification": {
            "years": [1867, 1890],
            "post": "Year >= 1873",
            "treatment": "1871 Catholic share, percentage points",
            "controls": [],
            "weights": "none",
            "covariance": "county clustered, linearmodels default debiasing",
            "eventStudyReference": 1872,
            "pretrendYears": list(range(1867, 1872)),
            "pretrendDistribution": (
                "Wald chi-square with 5 degrees of freedom; separate FE designs"
            ),
        },
        "sample": {
            "observations": len(panel),
            "counties": int(panel.Code.nunique()),
            "birthFlaggedRows": panel.loc[panel.cbr_flag, ["Code", "Year"]]
            .astype(int)
            .to_dict("records"),
        },
        "rows": rows,
        "allPrintedFieldsMatch": all(all(row["matches"].values()) for row in rows),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "linearmodels", "scipy", "openpyxl", "xlrd"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/table-6-1-results.json")
    parser.add_argument(
        "--record-inputs", action="store_true", help="Maintainer: create reviewable input manifest"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Fail unless all 40 printed fields match"
    )
    args = parser.parse_args()
    actual = inputs()
    original_bytes = inputs(canonical=False)
    manifest_path = HERE / "inputs.json"
    if args.record_inputs:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    verify_inputs(actual, json.loads(manifest_path.read_text(encoding="utf-8")))
    sys.path.insert(0, str(PROJECT))
    from src.data import build_dataset

    # The legacy builder writes an ancillary crosswalk even with save=False.
    # Redirect that write to a disposable folder, never the user's analysis files.
    with tempfile.TemporaryDirectory(prefix="table-6-1-") as temp:
        previous = build_dataset.DATA_PROCESSED
        try:
            build_dataset.DATA_PROCESSED = Path(temp)
            original = build_dataset.build_analysis_panel(
                year_start=1867, year_end=1890, save=False
            )
        finally:
            build_dataset.DATA_PROCESSED = previous
    panel = select_paper_sample(original)
    report = audit(panel)
    report["excludedOutOfWindow"] = (
        original.loc[~original.Year.between(1867, 1890), ["Code", "Year"]]
        .astype(int)
        .to_dict("records")
    )
    verify_inputs(inputs(), actual)
    verify_inputs(inputs(canonical=False), original_bytes)
    report["inputCount"] = len(actual)
    report["hashPolicy"] = (
        "Manifest normalizes CRLF to LF for .py/.csv; binary inputs exact. "
        "Before/after preservation also checks all original bytes."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.verify and not report["allPrintedFieldsMatch"]:
        raise SystemExit("Some printed fields differ; see the JSON report")


if __name__ == "__main__":
    main()
