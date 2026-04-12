"""
load_data.py
============
Functions for loading and harmonising Galloway Prussia Database files
and the iPEHD Becker-Woessmann replication dataset.

Usage (from notebook):
    from src.load_data import load_rel1871, load_vit_panel, load_ipehd_master
    from src.load_data import load_pop_census, interpolate_population
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths – adjust DATA_DIR if your layout differs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _find_file(data_dir: Path, basename: str) -> Optional[Path]:
    """Try common extension variants (.xlsx, .XLS, .xls) for a file."""
    for ext in [".xlsx", ".XLS", ".xls"]:
        p = data_dir / f"{basename}{ext}"
        if p.exists():
            return p
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to mixed-case convention used in VIT1875.
    
    Some Galloway files use ALL CAPS (VIT1862, VIT1881), others use
    mixed case (VIT1875). This maps everything to a consistent form.
    """
    canonical = {
        "code": "Code", "rb": "Rb", "kreis": "Kreis",
        "type": "Type", "year": "Year",
        "popm": "Popm", "popf": "Popf", "poptot": "Poptot", "pop": "Pop",
        "popmilitary": "Popmilitary",
        "birtot": "Birtot",
        "birm": "Birm", "birf": "Birf",
        "birinstm": "Birinstm", "birinstf": "Birinstf",
        "birlegtot": "Birlegtot",
        "birleglivem": "Birleglivem", "birleglivef": "Birleglivef",
        "birlegdeadm": "Birlegdeadm", "birlegdeadf": "Birlegdeadf",
        "birbastot": "Birbastot",
        "birbaslivem": "Birbaslivem", "birbaslivef": "Birbaslivef",
        "birbasdeadm": "Birbasdeadm", "birbasdeadf": "Birbasdeadf",
        "birdeadtot": "Birdeadtot",
        "dthtot": "Dthtot",
        "dthm": "Dthm", "dthf": "Dthf",
        "dthinstm": "Dthinstm", "dthinstf": "Dthinstf",
        "dth<1leg": "Dth<1leg", "dth<1bas": "Dth<1bas",
        "dthyoung": "Dthyoung", "dthsuicide": "Dthsuicide",
        "martot": "Martot",
        "marevan": "Marevan", "marcath": "Marcath",
        "marjew": "Marjew", "marother": "Marother",
        "inmigm": "Inmigm", "inmigf": "Inmigf",
        "inmigtot": "Inmigtot",
        "outmigm": "Outmigm", "outmigf": "Outmigf",
        "outmigtot": "Outmigtot", "outmigunoff": "Outmigunoff",
        "relevan": "Relevan", "relcath": "Relcath",
        "relcathm": "Relcathm", "relcathf": "Relcathf",
        "reljew": "Reljew", "relother": "Relother",
    }
    
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in canonical:
            rename_map[col] = canonical[lower]
    
    return df.rename(columns=rename_map)


# ===================================================================
# 1.  REL1871 – Religion census (one cross-section)
# ===================================================================

def load_rel1871(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Galloway REL1871 file and compute denomination shares.
    Returns only Type 0 (Stadt+Land combined) Kreise, Code < 900.
    """
    if path is None:
        path = _find_file(DATA_RAW, "REL1871")
        if path is None:
            raise FileNotFoundError("REL1871 not found in data/raw/")

    df = _normalize_columns(pd.read_excel(path))
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    df["cath_share"] = (df["Relcathm"] + df["Relcathf"]) / df["Pop"] * 100
    df["prot_share"] = 100 - df["cath_share"]

    cols_out = ["Code", "Rb", "Kreis", "Pop", "cath_share", "prot_share"]
    return df[cols_out].reset_index(drop=True)


# ===================================================================
# 2.  POP files – Population census (for interpolation)
# ===================================================================

def load_pop_census(data_dir: Optional[Path] = None, years=None) -> pd.DataFrame:
    """
    Load POP census files and extract population by county.
    Returns DataFrame with Code, Year, Pop_census (Type 0, Code < 900).
    """
    if data_dir is None:
        data_dir = DATA_RAW
    if years is None:
        years = [1861, 1864, 1867, 1871, 1875, 1880, 1885, 1890]
    
    frames = []
    for yr in years:
        path = _find_file(data_dir, f"POP{yr}")
        if path is None:
            continue
        
        df = _normalize_columns(pd.read_excel(path))
        df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()
        
        if "Pop" in df.columns:
            pop_col = "Pop"
        elif "Poptot" in df.columns:
            pop_col = "Poptot"
        else:
            continue
        
        frames.append(df[["Code", "Year", pop_col]].rename(columns={pop_col: "Pop_census"}))
        print(f"  POP{yr}: {len(df)} counties loaded")
    
    if not frames:
        return pd.DataFrame(columns=["Code", "Year", "Pop_census"])
    
    return pd.concat(frames, ignore_index=True)


def interpolate_population(
    panel: pd.DataFrame,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fill missing Poptot values using linear interpolation from POP census files.
    For years where VIT files don't include population (pre-1872).
    """
    pop_census = load_pop_census(data_dir)
    
    if len(pop_census) == 0:
        print("Warning: No POP census files found, cannot interpolate.")
        return panel
    
    panel = panel.copy()
    
    # Merge census population
    panel = panel.merge(pop_census, on=["Code", "Year"], how="left")
    
    # Where Poptot is missing but census pop exists, use census
    mask = panel["Poptot"].isna() & panel["Pop_census"].notna()
    panel.loc[mask, "Poptot"] = panel.loc[mask, "Pop_census"]
    
    # Interpolate within each county
    panel = panel.sort_values(["Code", "Year"])
    panel["Poptot"] = panel.groupby("Code")["Poptot"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )
    
    panel = panel.drop(columns=["Pop_census"], errors="ignore")
    
    n_still_missing = panel["Poptot"].isna().sum()
    if n_still_missing > 0:
        print(f"Warning: {n_still_missing} obs still missing Poptot after interpolation")
    else:
        print("Population interpolation complete — no missing values.")
    
    return panel


# ===================================================================
# 3.  VIT files – Vital registration panel (annual, 1862-1914)
# ===================================================================

def _load_single_vit(path: Path) -> pd.DataFrame:
    """
    Load a single VIT file and harmonise column names.
    Handles all four Galloway file formats transparently.
    """
    df = _normalize_columns(pd.read_excel(path))
    
    out = pd.DataFrame()
    out["Code"] = df["Code"]
    out["Rb"] = df["Rb"]
    out["Kreis"] = df["Kreis"]
    out["Type"] = df["Type"]
    out["Year"] = df["Year"]
    
    # --- Population ---
    if "Poptot" in df.columns:
        out["Poptot"] = df["Poptot"]
    elif "Pop" in df.columns:
        out["Poptot"] = df["Pop"]
    else:
        out["Poptot"] = np.nan
    
    # --- Total births ---
    if "Birtot" in df.columns:
        out["Birtot"] = df["Birtot"]
    elif "Birm" in df.columns and "Birf" in df.columns:
        out["Birtot"] = df["Birm"] + df["Birf"]
    else:
        out["Birtot"] = np.nan
    
    # --- Legitimate births ---
    if "Birlegtot" in df.columns:
        out["Birlegtot"] = df["Birlegtot"]
    elif "Birleglivem" in df.columns:
        out["Birlegtot"] = (
            df.get("Birleglivem", 0) + df.get("Birleglivef", 0) +
            df.get("Birlegdeadm", 0) + df.get("Birlegdeadf", 0)
        )
    else:
        out["Birlegtot"] = np.nan
    
    # --- Illegitimate births ---
    if "Birbastot" in df.columns:
        out["Birbastot"] = df["Birbastot"]
    elif "Birbaslivem" in df.columns:
        out["Birbastot"] = (
            df.get("Birbaslivem", 0) + df.get("Birbaslivef", 0) +
            df.get("Birbasdeadm", 0) + df.get("Birbasdeadf", 0)
        )
    else:
        out["Birbastot"] = np.nan
    
    # --- Total deaths ---
    if "Dthtot" in df.columns:
        out["Dthtot"] = df["Dthtot"]
    elif "Dthm" in df.columns and "Dthf" in df.columns:
        out["Dthtot"] = df["Dthm"] + df["Dthf"]
    else:
        out["Dthtot"] = np.nan
    
    # --- Infant deaths ---
    if "Dth<1leg" in df.columns:
        out["Dth_infant_leg"] = df["Dth<1leg"]
    elif "Dthyoung" in df.columns:
        out["Dth_infant_leg"] = df["Dthyoung"]
    else:
        out["Dth_infant_leg"] = np.nan
    
    # --- Marriages ---
    out["Martot"] = df["Martot"] if "Martot" in df.columns else np.nan
    out["Marevan"] = df["Marevan"] if "Marevan" in df.columns else np.nan
    out["Marcath"] = df["Marcath"] if "Marcath" in df.columns else np.nan
    
    return out


def load_vit_panel(
    data_dir: Optional[Path] = None,
    year_start: int = 1862,
    year_end: int = 1914,
    type_filter: int = 0,
) -> pd.DataFrame:
    """
    Load all VIT files from year_start to year_end and stack into a panel.
    Automatically handles different file extensions and column name cases.
    """
    if data_dir is None:
        data_dir = DATA_RAW

    frames = []
    for year in range(year_start, year_end + 1):
        fpath = _find_file(data_dir, f"VIT{year}")
        if fpath is None:
            print(f"  [skip] VIT{year} not found")
            continue
        
        try:
            df = _load_single_vit(fpath)
            frames.append(df)
        except Exception as e:
            print(f"  [error] VIT{year}: {e}")
            continue

    if not frames:
        raise FileNotFoundError(f"No VIT files found in {data_dir}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["Code"] < 900].copy()
    
    if type_filter is not None:
        panel = panel[panel["Type"] == type_filter].copy()
    
    panel = panel.sort_values(["Code", "Year"]).reset_index(drop=True)
    
    print(f"Loaded VIT panel: {len(panel)} observations, "
          f"{panel['Code'].nunique()} counties, "
          f"years {panel['Year'].min()}-{panel['Year'].max()}")
    
    return panel


# ===================================================================
# 4.  iPEHD master dataset (reference only)
# ===================================================================

def load_ipehd_master(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Becker-Woessmann (2009) replication dataset.
    Cross-section of 452 counties in 1871. For reference / validation only.
    """
    if path is None:
        path = DATA_RAW / "ipehd_qje2009_master.dta"
    return pd.read_stata(path)