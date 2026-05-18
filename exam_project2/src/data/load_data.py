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


ELECTION_YEARS = (1871, 1874, 1878, 1881, 1884, 1887, 1890)


def _build_wahlkreis_crosswalk(
    ele: pd.DataFrame,
    panel_kreise: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Wahlkreis -> Kreis crosswalk for a given ELE file.

    Shared by ``load_ele1871`` and ``load_election_panel`` so the same
    name-parsing logic applies to every election year.
    """
    import re
    from src.data.merge_ipehd import _clean_name

    def _clean_constituent(name: str) -> str:
        s = _clean_name(name)
        s = re.sub(
            r"\s+(STADT|LAND|NORD|SUED|OST|WEST|I|II|III|IV)$", "", s,
        )
        s = s.replace("PRIEGNITZ", "PRIGNITZ")
        return s.replace("  ", " ").strip()

    ele = ele.copy()
    ele["wahlkreis_name"] = ele["Wahlkreis"].str.replace(r"^\d+\s+", "", regex=True)
    ele["constituent_names"] = ele["wahlkreis_name"].str.split("-")

    pk = panel_kreise.copy()
    pk["kreis_clean"] = pk["Kreis"].apply(_clean_name)

    matches: list[tuple[int, str, int]] = []
    for _, row in ele.iterrows():
        rb_w = row["Rb"]
        for name in row["constituent_names"]:
            if not isinstance(name, str):
                continue
            k_clean = _clean_constituent(name.strip())
            if not k_clean:
                continue
            # 1. exact within Rb -> 2. exact across Rbs ->
            # 3. contains within Rb -> 4. contains across Rbs.
            for filt in [
                (pk["Rb"] == rb_w) & (pk["kreis_clean"] == k_clean),
                pk["kreis_clean"] == k_clean,
                (pk["Rb"] == rb_w) & pk["kreis_clean"].str.contains(k_clean, regex=False),
                pk["kreis_clean"].str.contains(k_clean, regex=False),
            ]:
                m = pk[filt]
                if len(m) == 1:
                    matches.append(
                        (int(row["Code"]), rb_w, int(m.iloc[0]["Code"]))
                    )
                    break

    return pd.DataFrame(
        matches, columns=["wahlkreis_code", "wahlkreis_rb", "Code"]
    ).drop_duplicates(subset=["Code"])


# NOTE: an earlier `load_edu1886` lived here. The richer version (raw
# counts + attendance rate + pupils-per-teacher, used by
# `channels.schooling_channel`) is defined further down in this module.


URB_YEARS = (1875, 1880, 1885, 1890)


def load_urb_panel(
    years: tuple[int, ...] = URB_YEARS,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load Galloway's URB{year}.XLS files (urbanisation cross-sections
    at 1875, 1880, 1885, 1890) and return a long-format panel of
    Kreis-level urban share by year.

    Output columns
    --------------
      Code, Year, percenturban, popurban, poptot

    Filters to Type=0 (combined Stadt+Land Kreise, matching the main
    analysis panel) and Code<900. URB1885 stores total population in
    column ``Poptot`` and urban population in ``Poptot-1`` (a Galloway
    formatting quirk); we harmonise to ``poptot`` / ``popurban`` so
    downstream code can treat all four years uniformly.

    Note. Galloway's URB ``Percenturban`` is *not* identical to iPEHD's
    1871 ``f_urban``: the URB definition has a stricter urban-place
    threshold and uses Galloway's own population denominator, whereas
    iPEHD harmonises to Becker-Woessmann's Reichstag-1871 base. Levels
    differ by ~6 pp on average. Treat them as separate measures: keep
    ``f_urban`` (1871) for the static iPEHD heterogeneity slot, and
    use the URB-derived ``urban_share_current`` (time-varying, 1875+
    only, linearly interpolated between URB anchors) for the
    Bai/Hsiao time-varying-trend spec.
    """
    if data_dir is None:
        data_dir = DATA_RAW

    frames = []
    for year in years:
        path = _find_file(data_dir, f"URB{year}")
        if path is None:
            logger.warning("URB%d not found; skipping", year)
            continue
        df = pd.read_excel(path)
        # Type-0 Kreise only.
        df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

        # URB1885 uses 'Poptot-1' for urban-pop (Galloway typo); harmonise.
        urban_col = "Popurban" if "Popurban" in df.columns else "Poptot-1"
        out = pd.DataFrame({
            "Code": df["Code"].astype(int),
            "Year": int(year),
            "percenturban": df["Percenturban"].astype(float),
            "popurban": df[urban_col].astype(float) if urban_col in df.columns else np.nan,
            "poptot": df["Poptot"].astype(float),
        })
        frames.append(out)
        logger.info("URB%d: %d Type-0 Kreise loaded (mean percenturban=%.2f)",
                    year, len(out), float(out["percenturban"].mean()))

    if not frames:
        return pd.DataFrame(columns=["Code", "Year", "percenturban", "popurban", "poptot"])
    return pd.concat(frames, ignore_index=True).sort_values(
        ["Code", "Year"]).reset_index(drop=True)


def load_election_panel(
    years: tuple[int, ...] = ELECTION_YEARS,
    panel_kreise: Optional[pd.DataFrame] = None,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load all 7 Galloway Reichstag-election files (1871, 1874, 1878,
    1881, 1884, 1887, 1890) and produce a long-format Kreis-by-election
    panel of vote shares.

    Returns
    -------
    pd.DataFrame with columns:
      - Code (Kreis identifier)
      - election_year
      - zentrum_share, polen_share, catholic_party_share (per cent of
        valid votes)

    Coverage. Same Wahlkreis -> Kreis crosswalk as ``load_ele1871``;
    typical coverage is ~85-90% of Type-0 Kreise per election year.
    Unmatched Kreise are omitted from the long panel for that year.

    Notes on column naming. Galloway's German party labels drift over
    time (e.g. "Konservativ" 1871/1874 vs "Deutsch konservativ" 1878+).
    Zentrum and Polen are the only two parties with stable names across
    all seven files, and they are the politically Catholic parties --
    exactly what we need to study Catholic political mobilisation as a
    Kulturkampf outcome. Other parties are not extracted in this
    long-format function; see ``load_ele1871`` for the full per-party
    breakdown of the 1871 cross-section.

    Use case (political mobilisation as a Kulturkampf outcome): stack
    the seven election years into a 7x338 = ~2,366-row DiD panel with
    cath_share x Post as the treatment, Zentrum share as the outcome.
    This directly measures whether Catholic political support
    accelerated during enforcement (1874, 1878), persisted during
    rollback (1881, 1884, 1887), or had decayed by post-rollback
    (1890).
    """
    if data_dir is None:
        data_dir = DATA_RAW

    # Reference Kreis frame (one row per Kreis).
    if panel_kreise is None:
        rel_path = _find_file(data_dir, "REL1871")
        if rel_path is None:
            raise FileNotFoundError("REL1871 not found in data/raw/")
        rel = load_rel1871(path=rel_path)
        panel_kreise = rel[["Code", "Kreis", "Rb"]].copy()

    all_rows = []
    for year in years:
        path = _find_file(data_dir, f"ELE{year}")
        if path is None:
            logger.warning("ELE%d not found; skipping", year)
            continue

        ele = pd.read_excel(path)
        cw = _build_wahlkreis_crosswalk(ele, panel_kreise)
        if len(cw) == 0:
            logger.warning("ELE%d: empty crosswalk; skipping", year)
            continue

        # Zentrum and Polen are stable column names across all 7 files.
        valid = ele["Gultige stimmen"].replace(0, np.nan)
        ele = ele.assign(
            zentrum_share=ele["Zentrum"] / valid * 100,
            polen_share=ele["Polen"] / valid * 100,
        )
        ele["catholic_party_share"] = (
            ele["zentrum_share"].fillna(0) + ele["polen_share"].fillna(0)
        )
        ele = ele.rename(columns={"Code": "wahlkreis_code", "Rb": "wahlkreis_rb"})
        merged = cw.merge(
            ele[
                ["wahlkreis_code", "wahlkreis_rb",
                 "zentrum_share", "polen_share", "catholic_party_share"]
            ],
            on=["wahlkreis_code", "wahlkreis_rb"], how="inner",
        )
        merged["election_year"] = year
        all_rows.append(
            merged[["Code", "election_year", "zentrum_share",
                    "polen_share", "catholic_party_share"]]
        )
        logger.info(
            "ELE%d: %d of %d Kreise matched (%.1f%%)",
            year, len(merged), len(panel_kreise),
            100.0 * len(merged) / len(panel_kreise),
        )

    if not all_rows:
        return pd.DataFrame(
            columns=["Code", "election_year", "zentrum_share",
                     "polen_share", "catholic_party_share"]
        )
    return pd.concat(all_rows, ignore_index=True).sort_values(
        ["Code", "election_year"]
    ).reset_index(drop=True)


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


# ===================================================================
# 7.  Additional Galloway cross-sections — STA / BIR / TAX / AGR / GEL / EDU
# ===================================================================
#
# These files were uploaded after the initial pipeline was built and
# round out the Kulturkampf-era picture: STA1871 gives marital status,
# BIR1871 birthplace, TAX1876 income tax revenue, AGR1882 farm-size
# distribution, GEL1882 service-sector employment, EDU1886 post-
# Kulturkampf school attendance. Each file is one cross-section; the
# loaders return a county-level frame keyed by Code that the build_dataset
# pipeline merges as time-invariant rows.


def load_bir1871(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Galloway BIR1871 and compute migration / origin shares.

    BIR1871 records, for each Kreis, how many residents were born
    inside the locality, inside the Kreis, inside the Provinz, in
    Prussia, in Germany, or outside Germany. Provides a Galloway-
    native mobility proxy that complements iPEHD's ``f_ortsgeb``.

    Returns columns:
        Code, born_in_locality_share_1871, born_in_kreis_share_1871,
        born_in_prussia_share_1871, born_outside_prussia_share_1871
    """
    if path is None:
        path = _find_file(DATA_RAW, "BIR1871")
        if path is None:
            raise FileNotFoundError("BIR1871 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    # Galloway BIR1871 categories are *nested*, not disjoint: each
    # successive category is a superset of the previous one (locality
    # subset of Kreis subset of Provinz subset of Prussia). We therefore
    # use each column directly as its own population share rather than
    # cumulating.
    pop = (df["Popm"].astype(float) + df["Popf"].astype(float)).replace(0, np.nan)
    loc = df["Borninlocalitym"].astype(float) + df["Borninlocalityf"].astype(float)
    kr = df["Borninkreism"].astype(float) + df["Borninkreisf"].astype(float)
    pruss = df["Borninprussiam"].astype(float) + df["Borninprussiaf"].astype(float)
    outside = df["Bornoutofgermanyandunk"].astype(float)

    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "born_in_locality_share_1871": loc / pop,
        "born_in_kreis_share_1871": kr / pop,
        "born_in_prussia_share_1871": pruss / pop,
        "born_outside_prussia_share_1871": outside / pop,
    })
    return out.reset_index(drop=True)


def load_tax1876(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load TAX1876 income-tax revenue per Kreis and produce a per-capita
    log measure. Note TAX1876 is mid-treatment (1876 is year 3 of the
    May Laws), so this enters as a heterogeneity moderator, not a
    pre-period control.
    """
    if path is None:
        path = _find_file(DATA_RAW, "TAX1876")
        if path is None:
            raise FileNotFoundError("TAX1876 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    pop = df["Pop1875"].astype(float).replace(0, np.nan)
    tax = df["Income tax"].astype(float)
    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "income_tax_pc_1876": tax / pop,
        "ln_income_tax_pc_1876": np.log((tax / pop).where(tax > 0)),
    })
    return out.reset_index(drop=True)


def load_agr1882(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load AGR1882 farm-holdings count by size bin and compute a Gini
    coefficient over the bins as a land-inequality moderator. AGR1882
    is 1882, i.e. post-treatment, but farm-size distribution evolves
    slowly enough that it is approximately structural.

    Size bins (hectares): <1, 1-2, 2-10, 10-50, 50-100, >100.
    """
    if path is None:
        path = _find_file(DATA_RAW, "AGR1882")
        if path is None:
            raise FileNotFoundError("AGR1882 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    bins = ["Landwirt uber <1", "Landwirt uber 1-2", "Landwirt uber 2-10",
            "Landwirt uber 10-50", "Landwirt uber 50-100",
            "Landwirt uber >100"]
    # Midpoint of each size bin (hectares); >100 capped at 200 (rough mean).
    midpoints = np.array([0.5, 1.5, 6.0, 30.0, 75.0, 200.0])

    arr = df[bins].astype(float).fillna(0).to_numpy()
    totals = arr.sum(axis=1)
    safe_totals = np.where(totals > 0, totals, np.nan)[:, None]
    shares = arr / safe_totals

    # Land share = bin_count * bin_midpoint / total_land.
    land = arr * midpoints
    land_sum = land.sum(axis=1, keepdims=True)
    safe_land_sum = np.where(land_sum > 0, land_sum, np.nan)
    land_share = land / safe_land_sum

    # Gini over farm-size distribution via cumulative Lorenz.
    # Sort by midpoint ascending (already in order); compute area
    # between Lorenz curve and the 45-degree line.
    cum_count = np.cumsum(shares, axis=1)
    cum_land = np.cumsum(land_share, axis=1)
    # Gini = 1 - 2*area_under_Lorenz; trapezoid rule between bins.
    # Padded with zero at the origin.
    cum_count_pad = np.concatenate([np.zeros((len(arr), 1)), cum_count], axis=1)
    cum_land_pad = np.concatenate([np.zeros((len(arr), 1)), cum_land], axis=1)
    width = np.diff(cum_count_pad, axis=1)
    height = (cum_land_pad[:, 1:] + cum_land_pad[:, :-1]) / 2
    area = (width * height).sum(axis=1)
    gini = 1 - 2 * area
    # Zero out rows with no holdings reported.
    gini = np.where(totals > 0, gini, np.nan)

    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "farms_total_1882": totals,
        "farms_share_under_2ha_1882": shares[:, :2].sum(axis=1),
        "farms_share_over_50ha_1882": shares[:, 4:].sum(axis=1),
        "land_gini_1882": gini,
    })
    return out.reset_index(drop=True)


def load_gel1882(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load GEL1882 service-sector employment. Columns include
    ``Xxiii.3.rel-erz-unterr`` (religion-education-instruction
    occupations) which is the most Kulturkampf-relevant — direct
    measurement of clerical and educational employment after the
    May Laws. Columns suffixed "a" are self-employed, "b" are
    employees; we sum them.
    """
    if path is None:
        path = _find_file(DATA_RAW, "GEL1882")
        if path is None:
            raise FileNotFoundError("GEL1882 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    pop = df["Pop1880"].astype(float).replace(0, np.nan)

    def _ab(col_a: str, col_b: str) -> pd.Series:
        a = df[col_a].astype(float) if col_a in df.columns else 0.0
        b = df[col_b].astype(float) if col_b in df.columns else 0.0
        return a + b

    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "rel_edu_emp_1882": _ab("Xxiii.3.rel-erz-unterr a",
                                "Xxiii.3.rel-erz-unterr b"),
        "transport_emp_1882": _ab("Xx.1. post-tele-eisen a",
                                  "Xx.1. post-tele-eisen b")
                              + _ab("Xx.2.fuhr-frachtwesen a.",
                                    "Xx.2.fuhr-frachtwesen b.")
                              + _ab("Xx.3.wasserverkehr a.",
                                    "Xx.3.wasserverkehr b."),
        "health_emp_1882": _ab("Xxiii.4.gesund-krank a.",
                               "Xxiii.4.gesund-krank b."),
        "finance_emp_1882": _ab("Xviii.2. geld-kredit a",
                                "Xviii.2. geld-kredit b"),
        "pop_1880_gel": df["Pop1880"].astype(float),
    })
    # Per-1k normalisations
    out["rel_edu_emp_per_1k_1882"] = out["rel_edu_emp_1882"] / pop * 1000
    out["transport_emp_per_1k_1882"] = out["transport_emp_1882"] / pop * 1000
    out["health_emp_per_1k_1882"] = out["health_emp_1882"] / pop * 1000
    out["finance_emp_per_1k_1882"] = out["finance_emp_1882"] / pop * 1000
    return out.reset_index(drop=True)


def load_edu1886(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load EDU1886 schooling cross-section. Provides post-Kulturkampf
    schooling endpoints to pair with EDU1849 / iPEHD school1517 (1871).

    Returns columns:
        Code, school_age_pop_1886 (= compulsory-school-age 6-14),
        attend_public_1886, attend_private_1886, attend_rate_1886,
        teachers_1886, teacher_income_1886.
    """
    if path is None:
        path = _find_file(DATA_RAW, "EDU1886")
        if path is None:
            raise FileNotFoundError("EDU1886 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    school_age = df["I.22. schulpfl 6 to 14"].astype(float).replace(0, np.nan)
    attend_public = df["I.29. besuchen volksschu"].astype(float)
    attend_private = df["I.23. besuchen privat"].astype(float).fillna(0)

    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "school_age_pop_1886": df["I.22. schulpfl 6 to 14"].astype(float),
        "attend_public_1886": attend_public,
        "attend_private_1886": attend_private,
        "attend_rate_1886": (attend_public + attend_private) / school_age,
        "teachers_1886": df["V.9. vollb lehrer"].astype(float),
        "teacher_income_1886": df["X.2. einkom vollb lehrer"].astype(float),
    })
    out["pupils_per_teacher_1886"] = (
        (attend_public + attend_private) / out["teachers_1886"].replace(0, np.nan)
    )
    return out.reset_index(drop=True)


def load_sta1871(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load STA1871 marital-status cross-section.

    Returns columns:
        Code, pct_never_married_m_1871, pct_never_married_f_1871,
        pct_widowed_f_1871, married_share_over15_f_1871,
        hh_avg_size_1871.

    ``married_share_over15_f_1871`` (= Marriedover15f / Popover15f) is
    the county-specific marriage-prevalence among women aged 15+. It
    feeds the proper Coale $I_g$ recalibration in
    ``coale_indices.compute_coale_indices(use_sta1871=True)`` -- the
    Princeton EFP framework normally assumes a Prussia-wide constant
    married-share schedule; with STA1871 we let each county's actual
    marriage prevalence shift the marital-fertility denominator.
    """
    if path is None:
        path = _find_file(DATA_RAW, "STA1871")
        if path is None:
            raise FileNotFoundError("STA1871 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    pop_m = df["Popover15m"].astype(float).replace(0, np.nan)
    pop_f = df["Popover15f"].astype(float).replace(0, np.nan)

    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "pct_never_married_m_1871": df["Singleover15m"].astype(float) / pop_m,
        "pct_never_married_f_1871": df["Singleover15f"].astype(float) / pop_f,
        "pct_widowed_f_1871": df["Widowover15f"].astype(float) / pop_f,
        "married_share_over15_f_1871":
            df["Marriedover15f"].astype(float) / pop_f,
        "hh_avg_size_1871": df["Hhfamily"].astype(float) / (
            (pop_m + pop_f).replace(0, np.nan)
        ),
        # Pop 15+ at 1871: exact (STA1871's Popover15 sum across sexes).
        # Anchor for the general marriage rate denominator (= marriages /
        # mid-year pop 15+, the "marriageable age" rate from Newell 1988
        # / standard demographic textbooks).
        "pop_15plus_1871": (
            df["Popover15m"].astype(float).fillna(0)
            + df["Popover15f"].astype(float).fillna(0)
        ),
    })
    return out.reset_index(drop=True)


def load_age1890(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load AGE1890 age-by-marital-status cross-section. Provides
    Galloway's *own* tabulation of the count of women aged 15-49
    (``Age15-49f``) and the count of married women aged 15-49
    (``Age15-49marriedf``) at the 1890 census - the actual denominators
    required by the Princeton EFP marital-fertility index and the
    Galloway, Hammel & Lee (1994) GMFR. Combined with the 1871 cross-
    sectional anchors (POP1871 + STA1871) it lets `compute_coale_indices`
    interpolate proper time-varying age x marital counts across 1862-1890
    instead of holding the 1871 marriage-prevalence schedule fixed.

    Returns columns:
        Code, women_15_49_1890, married_women_15_49_1890,
        married_share_15_49_f_1890,
        r_w_15_49_in_popf_1890 (= Age15-49f / total females; needed to
            extract 15-49 from AGE1882's coarse 0-19 / 20-69 bins),
        r_m_15_49_in_marriedf_1890 (= Age15-49marriedf / total married
            females; same purpose).
    """
    if path is None:
        path = _find_file(DATA_RAW, "AGE1890")
        if path is None:
            raise FileNotFoundError("AGE1890 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    w = df["Age15-49f"].astype(float)
    m = df["Age15-49marriedf"].astype(float)

    # Compute Kreis-level "ratio of 15-49 to total" for women and for
    # married women, used as a calibration constant when extracting
    # 15-49 counts from AGE1882's coarse 0-19 / 20-69 / 70+ bins.
    # Age 0-13 columns lack a sex split, so we assume a 50:50 sex ratio
    # for the child-age bins (a standard demographic approximation that
    # holds within <1% in 19th-c. Prussia).
    popf_1890 = (
        df["Age0"].astype(float) / 2.0
        + df["Age1-5"].astype(float) / 2.0
        + df["Age6-13"].astype(float) / 2.0
        + df["Age14-19f"].astype(float)
        + df["Age20-49f"].astype(float)
        + df["Age50-69f"].astype(float)
        + df["Age70andoverf"].astype(float)
    )
    marriedf_1890 = df["Marriedf"].astype(float)

    # Pop 15+ at 1890: sum the bins covering ages 15+. The 14-19 bin
    # includes 14-year-olds; we take 5/6 of it as a within-bin
    # approximation for the 15-19 portion (5 years out of 6). Bins
    # 20-49, 50-69, 70+ are fully within the 15+ range. Done for both
    # sexes.
    pop_14_19_total = (
        df["Age14-19m"].astype(float).fillna(0)
        + df["Age14-19f"].astype(float).fillna(0)
    )
    pop_15plus_1890 = (
        (5.0 / 6.0) * pop_14_19_total
        + df["Age20-49m"].astype(float).fillna(0)
        + df["Age20-49f"].astype(float).fillna(0)
        + df["Age50-69m"].astype(float).fillna(0)
        + df["Age50-69f"].astype(float).fillna(0)
        + df["Age70andoverm"].astype(float).fillna(0)
        + df["Age70andoverf"].astype(float).fillna(0)
    )

    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "women_15_49_1890": w,
        "married_women_15_49_1890": m,
        "married_share_15_49_f_1890": m / w.replace(0, np.nan),
        "r_w_15_49_in_popf_1890": w / popf_1890.replace(0, np.nan),
        "r_m_15_49_in_marriedf_1890":
            m / marriedf_1890.replace(0, np.nan),
        # Pop 15+ at 1890: anchor for the general marriage rate.
        "pop_15plus_1890": pop_15plus_1890,
    })
    return out.reset_index(drop=True)


def load_age1882(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load AGE1882 coarse age-by-marital-status cross-section. AGE1882
    only resolves bins (0-19, 20-69, 70+) -- there is no clean Age15-49
    column. We extract approximate 15-49 counts via the AGE1890-
    derived per-Kreis ratios
    ``r_w_15_49_in_popf_1890`` and ``r_m_15_49_in_marriedf_1890``
    (which are merged into the panel via :func:`load_age1890`). The
    extraction step itself is done inside :func:`build_analysis_panel`
    so the ratios are available; here we just return the raw 1882
    counts plus the female and married-female totals.

    Returns columns:
        Code, pop_1882f, marriedf_1882, women_15_49_1882_raw,
        married_women_15_49_1882_raw (the latter two are placeholders
        zero-filled when AGE1890 ratios are not yet available; the
        build pipeline overwrites them with proper estimates).
    """
    if path is None:
        path = _find_file(DATA_RAW, "AGE1882")
        if path is None:
            raise FileNotFoundError("AGE1882 not found in data/raw/")
    df = pd.read_excel(path)
    df = df[(df["Code"] < 900) & (df["Type"] == 0)].copy()

    out = pd.DataFrame({
        "Code": df["Code"].astype(int),
        "pop_1882f": df["Pop1882f"].astype(float),
        "marriedf_1882": (
            df["Married0-19f"].astype(float).fillna(0)
            + df["Married20-69f"].astype(float).fillna(0)
            + df["Married70andoverf"].astype(float).fillna(0)
        ),
    })
    return out.reset_index(drop=True)
