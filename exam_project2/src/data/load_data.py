"""
load_data.py
============
Functions for loading and harmonising Galloway Prussia Database files
and the iPEHD Becker-Woessmann replication dataset.

Usage (from notebook):
    from src.data.load_data import load_rel1871, load_vit_panel, load_ipehd_master
    from src.data.load_data import load_pop_census, interpolate_population
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths – adjust DATA_DIR if your layout differs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "galloway_data"
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

def load_pop_census(data_dir: Optional[Path] = None, years: Optional[List[int]] = None) -> pd.DataFrame:
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
        logger.info("  POP%d: %d counties loaded", yr, len(df))
    
    if not frames:
        return pd.DataFrame(columns=["Code", "Year", "Pop_census"])
    
    return pd.concat(frames, ignore_index=True)


def compute_midyear_population(
    panel: pd.DataFrame,
    data_dir: Optional[Path] = None,
    census_month: int = 12,
) -> pd.DataFrame:
    """
    Add a `Poptot_midyear` column to the panel by linearly interpolating
    between consecutive Prussian December census anchors and evaluating
    at mid-year (July 1) of each panel year.

    Why: Galloway's `Poptot` in inter-census years is the *previous*
    December census value, carried forward unchanged. That is end-of-year
    (Dec 1) rather than mid-year (July 1) population, which biases CBR
    upward by ~1-3% in growing populations and produces a sawtooth
    artefact at each census year. The standard demographic convention is
    mid-year population (an approximation to person-years lived), so
    proper CBR / GFR comparisons should use a smoothed mid-year denominator.

    Treats each POP{c}.xls file as a snapshot taken on Day 1 of
    `census_month` of year c (default December 1, matching Prussian
    practice). Interpolates linearly in fractional time, evaluates at
    `Year + 0.5` (July 1) for each panel observation. Counties outside
    the convex hull of available censuses (e.g. mid-1862 lies between
    Dec 1861 and Dec 1864 anchors) get the boundary-clamped extrapolation
    that ``np.interp`` provides.

    Returns a copy of `panel` with `Poptot_midyear` appended.
    """
    pop_census = load_pop_census(data_dir)
    if len(pop_census) == 0:
        logger.warning("No POP census files found, cannot compute mid-year Poptot.")
        out = panel.copy()
        out["Poptot_midyear"] = np.nan
        return out

    # Census date: Day 1 of `census_month` (default Dec) of the labelled year.
    pop_census = pop_census.copy()
    pop_census["t_census"] = pop_census["Year"] + (census_month - 1) / 12.0
    pop_census = (
        pop_census.dropna(subset=["Pop_census"])
        .sort_values(["Code", "t_census"])
        .reset_index(drop=True)
    )

    panel = panel.copy()
    panel["Poptot_midyear"] = np.nan

    # Per-county linear interpolation at mid-year (July 1 = year + 6/12).
    for code, grp in pop_census.groupby("Code"):
        anchors_t = grp["t_census"].to_numpy(dtype=float)
        anchors_p = grp["Pop_census"].to_numpy(dtype=float)
        if len(anchors_t) < 2:
            continue  # need at least two census anchors to interpolate
        mask = panel["Code"] == code
        if not mask.any():
            continue
        t_mid = panel.loc[mask, "Year"].to_numpy(dtype=float) + 0.5
        # np.interp clips to endpoint values for out-of-range points,
        # which is the right fallback for years before the first or after
        # the last census we have.
        panel.loc[mask, "Poptot_midyear"] = np.interp(t_mid, anchors_t, anchors_p)

    n_missing = panel["Poptot_midyear"].isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d obs still missing Poptot_midyear after interpolation "
            "(counties with <2 census anchors)", n_missing,
        )
    else:
        logger.info("Mid-year Poptot computed for all %d obs.", len(panel))
    return panel


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
        logger.warning("No POP census files found, cannot interpolate.")
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
        logger.warning("%d obs still missing Poptot after interpolation", n_still_missing)
    else:
        logger.info("Population interpolation complete — no missing values.")
    
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
    
    # --- Infant deaths, legitimate ---
    # Post-1875: Galloway publishes Dth<1leg (deaths of legitimate
    # children under 1 year). Pre-1875 the only available proxy is
    # Dthyoung (broader / less complete young-deaths category), which
    # produces the ~3-4x level discontinuity at 1875 documented in
    # fig_imr_break.png. IMR-based regressions are therefore restricted
    # to 1875+ in channels.infant_mortality_analysis.
    if "Dth<1leg" in df.columns:
        out["Dth_infant_leg"] = df["Dth<1leg"]
    elif "Dthyoung" in df.columns:
        out["Dth_infant_leg"] = df["Dthyoung"]
    else:
        out["Dth_infant_leg"] = np.nan

    # --- Infant deaths, illegitimate ---
    # Galloway publishes Dth<1bas (deaths of illegitimate children
    # under 1 year) from 1875 onwards. Pre-1875 there is no separate
    # illegitimate-infant-death column, so we leave this NaN -- the
    # `Dthyoung` fallback for legitimate infants does not extend to
    # the illegitimate series and the combined-infant total below is
    # therefore well-defined only for 1875+.
    if "Dth<1bas" in df.columns:
        out["Dth_infant_bas"] = df["Dth<1bas"]
    else:
        out["Dth_infant_bas"] = np.nan

    # --- Marriages ---
    out["Martot"] = df["Martot"] if "Martot" in df.columns else np.nan
    out["Marevan"] = df["Marevan"] if "Marevan" in df.columns else np.nan
    out["Marcath"] = df["Marcath"] if "Marcath" in df.columns else np.nan

    # --- In-migration ---
    # Pre-1875 files store Inmigtot directly; 1875-1886 files store by sex.
    # 1868-1871 and 1887+ have no migration data => NaN.
    if "Inmigtot" in df.columns:
        out["Inmigtot"] = df["Inmigtot"]
    elif "Inmigm" in df.columns and "Inmigf" in df.columns:
        m = df["Inmigm"]
        f = df["Inmigf"]
        both_nan = m.isna() & f.isna()
        out["Inmigtot"] = m.fillna(0) + f.fillna(0)
        out.loc[both_nan, "Inmigtot"] = np.nan
    else:
        out["Inmigtot"] = np.nan

    # --- Out-migration (official) ---
    if "Outmigtot" in df.columns:
        out["Outmigtot"] = df["Outmigtot"]
    elif "Outmigm" in df.columns and "Outmigf" in df.columns:
        m = df["Outmigm"]
        f = df["Outmigf"]
        both_nan = m.isna() & f.isna()
        out["Outmigtot"] = m.fillna(0) + f.fillna(0)
        out.loc[both_nan, "Outmigtot"] = np.nan
    else:
        out["Outmigtot"] = np.nan

    # --- Unofficial out-migration (only recorded 1875-1886) ---
    out["Outmigunoff"] = df["Outmigunoff"] if "Outmigunoff" in df.columns else np.nan

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
            logger.debug("  [skip] VIT%d not found", year)
            continue
        
        try:
            df = _load_single_vit(fpath)
            frames.append(df)
        except Exception as e:
            logger.error("  [error] VIT%d: %s", year, e)
            continue

    if not frames:
        raise FileNotFoundError(f"No VIT files found in {data_dir}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["Code"] < 900].copy()
    
    if type_filter is not None:
        panel = panel[panel["Type"] == type_filter].copy()
    
    panel = panel.sort_values(["Code", "Year"]).reset_index(drop=True)
    
    logger.info("Loaded VIT panel: %d observations, %d counties, years %d-%d",
                len(panel), panel['Code'].nunique(), panel['Year'].min(), panel['Year'].max())
    
    return panel


# ===================================================================
# 4.  POP1871 age x sex pyramid (cross-section, time-invariant covariates)
# ===================================================================

# Galloway POP1871 raw column -> analysis-friendly suffixed name (`_1871`).
POP1871_COLUMN_RENAME = {
    "Area":          "pop_area_1871",
    "Pop":           "pop_total_1871",
    "Popm":          "pop_m_1871",
    "Popf":          "pop_f_1871",
    "Popmilitary":   "pop_military_1871",
    "Age0-4m":       "age_0_4_m_1871",   "Age0-4f":       "age_0_4_f_1871",
    "Age5-14m":      "age_5_14_m_1871",  "Age5-14f":      "age_5_14_f_1871",
    "Age15-19m":     "age_15_19_m_1871", "Age15-19f":     "age_15_19_f_1871",
    "Age20-29m":     "age_20_29_m_1871", "Age20-29f":     "age_20_29_f_1871",
    "Age30-39m":     "age_30_39_m_1871", "Age30-39f":     "age_30_39_f_1871",
    "Age40-49m":     "age_40_49_m_1871", "Age40-49f":     "age_40_49_f_1871",
    "Age50-59m":     "age_50_59_m_1871", "Age50-59f":     "age_50_59_f_1871",
    "Age60andoverm": "age_60p_m_1871",   "Age60andoverf": "age_60p_f_1871",
}


def load_pop1871_age_structure(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load POP1871 and extract the age x sex pyramid plus area, as a
    time-invariant 1871 cross-section. Used downstream to build a proper
    General Fertility Rate (GFR = births / women aged 15-49) and to expose
    1871 age structure as a covariate / Bai-Hsiao baseline.

    Returns a DataFrame keyed by Code with columns
    `pop_area_1871`, `pop_total_1871`, `pop_m_1871`, `pop_f_1871`,
    `pop_military_1871`, plus eight age-by-sex bands suffixed `_1871`
    (`age_0_4_m_1871`, ..., `age_60p_f_1871`). Filters to Type=0, Code<900.
    """
    if path is None:
        path = _find_file(DATA_RAW, "POP1871")
        if path is None:
            raise FileNotFoundError("POP1871 not found in data/raw/")

    df = _normalize_columns(pd.read_excel(path))
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()
    df = df.rename(columns=POP1871_COLUMN_RENAME)

    keep = ["Code"] + [v for v in POP1871_COLUMN_RENAME.values() if v in df.columns]
    return df[keep].reset_index(drop=True)


# ===================================================================
# 5.  ELE1871 -- 1871 Reichstag election vote shares
# ===================================================================
#
# Galloway's ELE1871 reports Reichstag-election vote totals at the
# *Wahlkreis* (electoral district) level, not the *Kreis* (county) level.
# Wahlkreise are unions of multiple Kreise; the Wahlkreis name encodes
# the constituent Kreis names, e.g. "6 BRAUNSBERG-HEILSBERG" pools
# Kreise 13 (Braunsberg) and 14 (Heilsberg).
#
# To merge into the analysis panel we (i) parse the constituent Kreis
# names from each Wahlkreis label, (ii) match each constituent name to a
# panel Kreis (within the same Rb, then across all Rbs), and (iii)
# assign each Kreis the vote shares of its Wahlkreis. Counties that fall
# in unmatched Wahlkreise (city-rural splits like "DANZIG STADT" vs
# "DANZIG LAND" that do not exist as separate Type-0 Kreise; or spelling
# variants like "WESTPRIEGNITZ" vs panel's "WESTPRIGNITZ") retain NaN
# for the election variables.
#
# Vote-share variables derived (each is N_party_votes / N_valid_votes x 100):
#   zentrum_share_1871        -- Catholic Center Party (founded 1870)
#   polen_share_1871          -- Polish-nationalist Catholic party
#   catholic_party_share_1871 -- Zentrum + Polen combined
#   conservative_share_1871   -- Konservativ + Deutsche Reichspartei
#   liberal_share_1871        -- National-liberal + Liberal Reichspartei
#                                + Fortschrittspartei + Volkspartei
#   nat_liberal_share_1871    -- just National-liberal (the dominant
#                                Protestant-aligned liberal party in 1871)
#
# Validation: corr(cath_share, zentrum_share_1871) approx +0.66;
# corr(cath_share, catholic_party_share_1871) approx +0.77 -- consistent
# with Zentrum being the political expression of Catholic identity.


def load_ele1871(
    path: Optional[Path] = None,
    panel_kreise: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load ELE1871 and produce a Kreis-level DataFrame of 1871 Reichstag
    vote shares by party family.

    Parameters
    ----------
    path : Path, optional
        Path to ``ELE1871.XLS``. Defaults to ``DATA_RAW / "ELE1871.XLS"``.
    panel_kreise : pd.DataFrame, optional
        Reference frame with columns ``[Code, Kreis, Rb]`` listing every
        Type-0 Kreis. Used to match Wahlkreis constituent names to panel
        Kreise. If omitted, it is built from a freshly-loaded REL1871.

    Returns
    -------
    pd.DataFrame keyed by ``Code`` (Kreis code) with the columns above
    suffixed ``_1871``. Counties not matched to any Wahlkreis are
    omitted from the returned frame; the caller's outer-merge with the
    panel will yield NaN for those Kreise.
    """
    import re
    from src.data.merge_ipehd import _clean_name

    if path is None:
        path = _find_file(DATA_RAW, "ELE1871")
        if path is None:
            raise FileNotFoundError("ELE1871 not found in data/raw/")

    ele = pd.read_excel(path)

    # Strip leading "{N} " from the Wahlkreis label and split into
    # constituent Kreis names on the en-dash separator.
    ele["wahlkreis_name"] = ele["Wahlkreis"].str.replace(r"^\d+\s+", "", regex=True)
    ele["constituent_names"] = ele["wahlkreis_name"].str.split("-")

    # Reference panel of Kreise.
    if panel_kreise is None:
        rel = load_rel1871(path=_find_file(DATA_RAW, "REL1871"))
        panel_kreise = rel[["Code", "Kreis", "Rb"]].copy()
    panel_kreise = panel_kreise.copy()
    panel_kreise["kreis_clean"] = panel_kreise["Kreis"].apply(_clean_name)

    def _clean_constituent(name: str) -> str:
        """Normalise a constituent Kreis name: strip STADT/LAND/I/II/etc.
        suffixes that Wahlkreise use but panel Kreise (Type=0) do not."""
        s = _clean_name(name)
        s = re.sub(
            r"\s+(STADT|LAND|NORD|SUED|OST|WEST|I|II|III|IV)$",
            "", s,
        )
        # Handle minor spelling variants Galloway uses inconsistently.
        s = s.replace("PRIEGNITZ", "PRIGNITZ")
        return s.replace("  ", " ").strip()

    # Crosswalk: (wahlkreis_code, wahlkreis_rb) -> [kreis_code, ...]
    matches: list[tuple[int, str, int]] = []
    for _, row in ele.iterrows():
        rb_w = row["Rb"]
        for name in row["constituent_names"]:
            if not isinstance(name, str):
                continue
            k_clean = _clean_constituent(name.strip())
            if not k_clean:
                continue
            # 1. exact within Rb
            m = panel_kreise[
                (panel_kreise["Rb"] == rb_w)
                & (panel_kreise["kreis_clean"] == k_clean)
            ]
            if len(m) == 1:
                matches.append((row["Code"], rb_w, int(m.iloc[0]["Code"])))
                continue
            # 2. exact across Rbs (Galloway Rb assignment differs for a few)
            m = panel_kreise[panel_kreise["kreis_clean"] == k_clean]
            if len(m) == 1:
                matches.append((row["Code"], rb_w, int(m.iloc[0]["Code"])))
                continue
            # 3. contains within Rb
            m = panel_kreise[
                (panel_kreise["Rb"] == rb_w)
                & panel_kreise["kreis_clean"].str.contains(k_clean, regex=False)
            ]
            if len(m) == 1:
                matches.append((row["Code"], rb_w, int(m.iloc[0]["Code"])))
                continue
            # 4. contains across Rbs
            m = panel_kreise[
                panel_kreise["kreis_clean"].str.contains(k_clean, regex=False)
            ]
            if len(m) == 1:
                matches.append((row["Code"], rb_w, int(m.iloc[0]["Code"])))
                continue
            # unmatched: silently drop -- common cause is city-rural split
            # where the Wahlkreis names something that is not a Type-0 Kreis.

    crosswalk = pd.DataFrame(
        matches, columns=["wahlkreis_code", "wahlkreis_rb", "Code"]
    ).drop_duplicates(subset=["Code"])

    # Compute vote shares (per cent of valid votes) for each Wahlkreis.
    party_cols = {
        "Konservativ": "konservativ",
        "Deutsche reichspartei": "deutsche_reichspartei",
        "Liberal reichspartei": "liberal_reichspartei",
        "National-liberal": "national_liberal",
        "Fortschritts partei": "fortschritts",
        "Volkspartei": "volkspartei",
        "Sozialdemokrat": "sozialdemokrat",
        "Zentrum": "zentrum",
        "Polen": "polen",
    }
    ele_shares = ele[["Code", "Rb", "Gultige stimmen"] + list(party_cols.keys())].copy()
    ele_shares = ele_shares.rename(
        columns={"Code": "wahlkreis_code", "Rb": "wahlkreis_rb"}
    )
    valid = ele_shares["Gultige stimmen"].replace(0, np.nan)
    for raw, clean in party_cols.items():
        ele_shares[f"{clean}_share_1871"] = ele_shares[raw] / valid * 100
    # Composite party-family shares.
    ele_shares["catholic_party_share_1871"] = (
        ele_shares["zentrum_share_1871"].fillna(0)
        + ele_shares["polen_share_1871"].fillna(0)
    )
    ele_shares["conservative_share_1871"] = (
        ele_shares["konservativ_share_1871"].fillna(0)
        + ele_shares["deutsche_reichspartei_share_1871"].fillna(0)
    )
    ele_shares["liberal_share_1871"] = (
        ele_shares["liberal_reichspartei_share_1871"].fillna(0)
        + ele_shares["national_liberal_share_1871"].fillna(0)
        + ele_shares["fortschritts_share_1871"].fillna(0)
        + ele_shares["volkspartei_share_1871"].fillna(0)
    )
    ele_shares = ele_shares.rename(
        columns={"national_liberal_share_1871": "nat_liberal_share_1871"}
    )

    # Final merge to Kreis level.
    keep_cols = [
        "zentrum_share_1871", "polen_share_1871",
        "catholic_party_share_1871", "conservative_share_1871",
        "liberal_share_1871", "nat_liberal_share_1871",
        "sozialdemokrat_share_1871",
    ]
    out = crosswalk.merge(
        ele_shares[["wahlkreis_code", "wahlkreis_rb"] + keep_cols],
        on=["wahlkreis_code", "wahlkreis_rb"],
        how="inner",
    )[["Code"] + keep_cols]

    logger.info(
        "ELE1871: %d Kreise matched to Wahlkreis vote shares "
        "(coverage %.1f%% of panel Kreise)",
        len(out), 100.0 * len(out) / len(panel_kreise),
    )
    return out


# ===================================================================
# 6.  iPEHD master dataset (reference only)
# ===================================================================

def load_ipehd_master(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Becker-Woessmann (2009) replication dataset.
    Cross-section of 452 counties in 1871. For reference / validation only.
    """
    if path is None:
        path = DATA_RAW / "ipehd_qje2009_master.dta"
    return pd.read_stata(path)