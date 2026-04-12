"""
load_data.py
============
Functions for loading and harmonising Galloway Prussia Database files
and the iPEHD Becker-Woessmann replication dataset.

Usage (from notebook):
    from src.load_data import load_rel1871, load_vit_panel, load_ipehd_master
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


# ===================================================================
# 1.  REL1871 – Religion census (one cross-section)
# ===================================================================

def load_rel1871(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Galloway REL1871 file and compute denomination shares.
    
    Returns only Type 0 (Stadt+Land combined) Kreise, excluding
    Regierungsbezirk/province totals (Code >= 900).
    
    Columns returned
    ----------------
    Code, Rb, Kreis, Pop, cath_share, prot_share
        where cath_share = (Relcathm + Relcathf) / Pop * 100
        and   prot_share = 100 - cath_share  (approximation; ignores Jews & others)
    """
    if path is None:
        path = DATA_RAW / "REL1871.XLS"

    df = pd.read_excel(path)

    # Keep only actual Kreise (not totals) and Type 0 (combined Stadt+Land)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    # Compute Catholic share (percentage)
    df["cath_share"] = (df["Relcathm"] + df["Relcathf"]) / df["Pop"] * 100

    # Protestant share as residual (approximate – small Jewish/other population)
    df["prot_share"] = 100 - df["cath_share"]

    # Keep clean set of columns
    cols_out = ["Code", "Rb", "Kreis", "Pop", "cath_share", "prot_share"]
    return df[cols_out].reset_index(drop=True)


# ===================================================================
# 2.  VIT files – Vital registration panel (annual, 1862-1914)
# ===================================================================

# Column mappings for different VIT file formats.
# The Galloway files changed structure over time:
#   1862-1867 : 15 fields, totals only (Birtot, Dthtot, ...)
#   1868-1871 : 13 fields, similar but fewer migration vars
#   1872-1874 : 16 fields, adds Poptot
#   1875-1886 : 36 fields, gender-split births, marriage by religion
#   1887-1914 : 37 fields, adds Marsinglem/Marsinglef

# We harmonise all formats to a common set of output columns.

# Columns we always want in the output:
_OUTPUT_COLS = [
    "Code", "Rb", "Kreis", "Type", "Year",
    "Poptot",           # population (may be NaN for 1862-1871)
    "Birtot",           # total births
    "Birlegtot",        # legitimate births
    "Birbastot",        # illegitimate births
    "Dthtot",           # total deaths
    "Dth_infant_leg",   # infant deaths (legitimate, under 1 year)
    "Martot",           # total marriages
    "Marevan",          # evangelical marriages
    "Marcath",          # catholic marriages
]


def _load_single_vit(path: Path) -> pd.DataFrame:
    """
    Load a single VIT file and harmonise column names.
    
    Handles the four different file formats transparently.
    """
    df = pd.read_excel(path)
    year = df["Year"].iloc[0]
    
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
        # Sum live + dead, male + female
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
    
    # --- Infant deaths (legitimate, under 1) ---
    if "Dth<1leg" in df.columns:
        out["Dth_infant_leg"] = df["Dth<1leg"]
    elif "Dthyoung" in df.columns:
        # Early files only have total young deaths, not split by legitimacy
        out["Dth_infant_leg"] = df["Dthyoung"]
    else:
        out["Dth_infant_leg"] = np.nan
    
    # --- Marriages ---
    out["Martot"] = df.get("Martot", np.nan)
    out["Marevan"] = df.get("Marevan", np.nan)
    out["Marcath"] = df.get("Marcath", np.nan)
    
    return out


def load_vit_panel(
    data_dir: Optional[Path] = None,
    year_start: int = 1862,
    year_end: int = 1914,
    type_filter: int = 0,
) -> pd.DataFrame:
    """
    Load all VIT files from year_start to year_end and stack into a panel.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing VIT{year}.XLS files.
    year_start, year_end : int
        Range of years to load (inclusive).
    type_filter : int
        Kreis type to keep. Default 0 = Stadt+Land combined.
        Set to None to keep all types.
    
    Returns
    -------
    pd.DataFrame
        Panel with columns: Code, Rb, Kreis, Type, Year, Poptot,
        Birtot, Birlegtot, Birbastot, Dthtot, Dth_infant_leg,
        Martot, Marevan, Marcath.
        Filtered to Code < 900 (excludes totals).
    """
    if data_dir is None:
        data_dir = DATA_RAW

    frames = []
    for year in range(year_start, year_end + 1):
        fpath = data_dir / f"VIT{year}.XLS"
        if not fpath.exists():
            # Try lowercase
            fpath = data_dir / f"vit{year}.xls"
        if not fpath.exists():
            print(f"  [skip] VIT{year}.XLS not found")
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
    
    # Drop totals
    panel = panel[panel["Code"] < 900].copy()
    
    # Filter by Type
    if type_filter is not None:
        panel = panel[panel["Type"] == type_filter].copy()
    
    panel = panel.sort_values(["Code", "Year"]).reset_index(drop=True)
    
    print(f"Loaded VIT panel: {len(panel)} observations, "
          f"{panel['Code'].nunique()} counties, "
          f"years {panel['Year'].min()}-{panel['Year'].max()}")
    
    return panel


# ===================================================================
# 3.  iPEHD master dataset (reference only)
# ===================================================================

def load_ipehd_master(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Becker-Woessmann (2009) replication dataset.
    
    This is a cross-section of 452 counties in 1871.
    Use for reference / validation, not as your main data source.
    """
    if path is None:
        path = DATA_RAW / "ipehd_qje2009_master.dta"
    
    return pd.read_stata(path)
