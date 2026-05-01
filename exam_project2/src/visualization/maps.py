"""
maps.py
=======
Geospatial visualisation for the Kulturkampf-fertility paper.

Uses the MPIDR / Census Mosaic German Empire 1871 shapefile, where the
ID column corresponds to Galloway's Prussian Kreis codes.

Usage (from notebook):
    from src.visualization.maps import (
        load_prussia_shapefile,
        map_catholic_share,
        map_fertility_change,
        map_polish_german_provinces,
        map_kulturkampf_residuals,
    )
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from typing import Optional

try:
    import geopandas as gpd
except ImportError:
    raise ImportError(
        "geopandas is required for mapping. Install with:\n"
        "    pip install geopandas"
    )


def load_prussia_shapefile(
    shapefile_path: Path,
    keep_types: list = None,
) -> "gpd.GeoDataFrame":
    """
    Load the German Empire 1871 shapefile and filter to Prussia.
    
    Parameters
    ----------
    shapefile_path : Path
        Path to German_Empire_1871_v_1_0.shp (the .shx, .dbf, .prj 
        files must be in the same directory).
    keep_types : list of int, optional
        Shapefile TYPE values to keep. Default: None (keep all).
    
    Returns
    -------
    GeoDataFrame with columns including:
        Code (renamed from ID), NAME, RB, STATUS, TYPE, geometry
    Filtered to LAND == 1000 (Prussia) and Code < 1000 (excludes
    the small number of outlier IDs that don't match Galloway codes).
    """
    gdf = gpd.read_file(shapefile_path)
    
    # Filter to Prussia
    gdf = gdf[gdf["LAND"] == 1000].copy()
    
    # Rename ID to Code for merging with Galloway data
    gdf = gdf.rename(columns={"ID": "Code"})
    gdf["Code"] = gdf["Code"].astype(int)
    
    # Drop outlier codes (> 1000) that don't match Galloway's numbering
    gdf = gdf[gdf["Code"] < 1000].copy()
    
    if keep_types is not None:
        gdf = gdf[gdf["TYPE"].isin(keep_types)].copy()
    
    return gdf.reset_index(drop=True)


def _base_map(ax, gdf, edgecolor="white", linewidth=0.2):
    """Draw county boundaries on axis."""
    gdf.boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth)
    ax.set_axis_off()
    ax.set_aspect("equal")


def map_catholic_share(
    gdf: "gpd.GeoDataFrame",
    panel: pd.DataFrame,
    savepath: Optional[str] = None,
):
    """
    Choropleth of Catholic population share in 1871 (your treatment variable).
    
    This is Figure 1 of your paper — it shows the confessional geography
    of Prussia and makes the treatment variation visible.
    """
    # Get cath_share from the panel (one value per county)
    cath = panel.groupby("Code")["cath_share"].first().reset_index()
    merged = gdf.merge(cath, on="Code", how="left")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    merged.plot(
        column="cath_share",
        ax=ax,
        cmap="RdBu_r",
        vmin=0, vmax=100,
        legend=True,
        legend_kwds={
            "label": "Catholic population share (%)",
            "orientation": "vertical",
            "shrink": 0.6,
        },
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    _base_map(ax, merged)
    
    ax.set_title(
        "Catholic population share across Prussian counties, 1871",
        fontsize=14, fontweight="bold",
    )
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    n_matched = merged["cath_share"].notna().sum()
    print(f"Map drawn: {n_matched} counties with data, "
          f"{len(merged) - n_matched} without.")
    
    return fig, ax


def map_fertility_change(
    gdf: "gpd.GeoDataFrame",
    panel: pd.DataFrame,
    pre_years: tuple = (1868, 1872),
    post_years: tuple = (1878, 1882),
    outcome: str = "cbr",
    savepath: Optional[str] = None,
):
    """
    Choropleth of fertility change from pre- to post-Kulturkampf.
    
    For each county, compute:
        Δ = mean(outcome in post_years) - mean(outcome in pre_years)
    
    Reds = fertility decline, Blues = fertility increase.
    
    Default windows:
        pre:  1868-1872 (just before Kulturkampf, post-war recovery)
        post: 1878-1882 (peak enforcement → early rollback)
    """
    pre_mask = panel["Year"].between(*pre_years)
    post_mask = panel["Year"].between(*post_years)
    
    pre_cbr = panel[pre_mask].groupby("Code")[outcome].mean()
    post_cbr = panel[post_mask].groupby("Code")[outcome].mean()
    change = (post_cbr - pre_cbr).rename("fertility_change").reset_index()
    
    merged = gdf.merge(change, on="Code", how="left")
    
    # Diverging colourmap centred on zero
    vmax = np.nanpercentile(np.abs(merged["fertility_change"]), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    merged.plot(
        column="fertility_change",
        ax=ax,
        cmap="RdBu",
        norm=norm,
        legend=True,
        legend_kwds={
            "label": f"Change in {outcome} (per 1,000)",
            "orientation": "vertical",
            "shrink": 0.6,
        },
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    _base_map(ax, merged)
    
    ax.set_title(
        f"Change in crude birth rate: {pre_years[0]}-{pre_years[1]} "
        f"vs {post_years[0]}-{post_years[1]}",
        fontsize=14, fontweight="bold",
    )
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    return fig, ax


def map_polish_german_provinces(
    gdf: "gpd.GeoDataFrame",
    panel: pd.DataFrame,
    savepath: Optional[str] = None,
):
    """
    Map highlighting Polish Catholic vs German Catholic vs Protestant provinces.
    
    This visualises the subregional classification used in the 
    polish_vs_german_catholics() heterogeneity analysis.
    """
    polish_rbs = ["POS", "BRO"]
    german_cath_rbs = ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]
    
    # Get Rb from panel (should match shapefile RB)
    rb_map = panel.groupby("Code")["Rb"].first().reset_index()
    merged = gdf.merge(rb_map, on="Code", how="left")
    
    # Classify
    def classify(rb):
        if pd.isna(rb):
            return "No data"
        if rb in polish_rbs:
            return "Polish Catholic"
        if rb in german_cath_rbs:
            return "German Catholic"
        return "Protestant (rest)"
    
    merged["region_class"] = merged["Rb"].apply(classify)
    
    colours = {
        "Polish Catholic": "#C0392B",
        "German Catholic": "#2471A3",
        "Protestant (rest)": "#CCCCCC",
        "No data": "#EEEEEE",
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for region, colour in colours.items():
        subset = merged[merged["region_class"] == region]
        if len(subset) == 0:
            continue
        subset.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.2)
    
    ax.set_axis_off()
    ax.set_aspect("equal")
    
    # Legend
    patches = [
        mpatches.Patch(color=c, label=label)
        for label, c in colours.items()
        if label != "No data" or (merged["region_class"] == "No data").any()
    ]
    ax.legend(handles=patches, loc="upper left", frameon=True, fontsize=10)
    
    ax.set_title(
        "Regional classification: Polish Catholic, German Catholic, "
        "and Protestant provinces",
        fontsize=13, fontweight="bold",
    )
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    # Print summary
    print("Region counts:")
    print(merged["region_class"].value_counts().to_string())
    
    return fig, ax


def map_kulturkampf_residuals(
    gdf: "gpd.GeoDataFrame",
    panel: pd.DataFrame,
    pre_years: tuple = (1868, 1872),
    post_years: tuple = (1873, 1878),
    outcome: str = "cbr",
    savepath: Optional[str] = None,
):
    """
    Map of county-level residuals: each county's fertility change minus
    the aggregate trend.
    
    Logic: for each county, compute the change (post - pre). Subtract 
    the overall mean change. The residual is how much that county 
    deviated from the aggregate trend during the Kulturkampf.
    
    This visualises where the Kulturkampf "effect" (if any) clustered 
    spatially, after removing the common fertility trend.
    """
    pre_mask = panel["Year"].between(*pre_years)
    post_mask = panel["Year"].between(*post_years)
    
    pre_cbr = panel[pre_mask].groupby("Code")[outcome].mean()
    post_cbr = panel[post_mask].groupby("Code")[outcome].mean()
    change = (post_cbr - pre_cbr).rename("fertility_change")
    
    aggregate_change = change.mean()
    residual = (change - aggregate_change).rename("residual").reset_index()
    
    print(f"Aggregate change ({pre_years[0]}-{pre_years[1]} → "
          f"{post_years[0]}-{post_years[1]}): {aggregate_change:+.2f} per 1,000")
    
    merged = gdf.merge(residual, on="Code", how="left")
    
    vmax = np.nanpercentile(np.abs(merged["residual"]), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    merged.plot(
        column="residual",
        ax=ax,
        cmap="RdBu",
        norm=norm,
        legend=True,
        legend_kwds={
            "label": "Deviation from aggregate fertility change (per 1,000)",
            "orientation": "vertical",
            "shrink": 0.6,
        },
        missing_kwds={"color": "lightgrey"},
    )
    _base_map(ax, merged)
    
    ax.set_title(
        f"County-level residuals: fertility change minus aggregate trend\n"
        f"({pre_years[0]}-{pre_years[1]} vs {post_years[0]}-{post_years[1]})",
        fontsize=13, fontweight="bold",
    )
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    
    return fig, ax