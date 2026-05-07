"""
centroids.py
============
Build a crosswalk from Galloway county Code to county centroid coordinates
(in km, projected). Used by ``src.analysis.conley_se`` for spatial HAC
standard errors.

Source shapefile: ``data/raw/gis_data/German_Empire_1871_v.1.0.shp``
(EPSG:32633 / UTM zone 33N — units in metres).

Run as a script to (re)generate the crosswalk parquet::

    python -m src.data.centroids
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.data.load_data import DATA_PROCESSED, DATA_RAW, load_rel1871
from src.data.merge_ipehd import _clean_name

logger = logging.getLogger(__name__)

GIS_PATH = DATA_RAW.parent / "gis_data" / "German_Empire_1871_v.1.0.shp"
CENTROIDS_PATH = DATA_PROCESSED / "centroids.parquet"

# Major Catholic bishops' seats in the Prussian Empire as of 1871.
# Source: standard Catholic encyclopedias / Prussian church-history references.
# Coordinates are (latitude, longitude) in WGS84.
BISHOP_SEATS: dict[str, tuple[float, float]] = {
    "Koeln":      (50.9413, 6.9583),
    "Trier":      (49.7596, 6.6442),
    "Muenster":   (51.9624, 7.6258),
    "Paderborn":  (51.7189, 8.7575),
    "Hildesheim": (52.1542, 9.9456),
    "Osnabrueck": (52.2799, 8.0472),
    "Fulda":      (50.5519, 9.6753),
    "Limburg":    (50.3852, 8.0653),
    "Breslau":    (51.1079, 17.0385),
    "Frauenburg": (54.3593, 19.6817),
    "Gnesen":     (52.5347, 17.5827),
    "Posen":      (52.4064, 16.9252),
    "Kulm":       (53.3478, 18.4439),
}


def build_centroid_crosswalk() -> pd.DataFrame:
    """Return a DataFrame indexed by Galloway Code with x_km, y_km columns."""
    if not GIS_PATH.exists():
        raise FileNotFoundError(f"Shapefile not found at {GIS_PATH}")

    gdf = gpd.read_file(GIS_PATH)
    # Filter to Galloway-equivalent county units. TYPE=='0' in this shapefile
    # corresponds to the same administrative tier as Galloway Type 0
    # (Kreise / Stadtkreise; ~410 entries vs Galloway's 393).
    gdf = gdf[gdf["TYPE"] == "0"].copy()
    gdf["centroid"] = gdf.geometry.centroid
    gdf["x_km"] = gdf["centroid"].x / 1000.0
    gdf["y_km"] = gdf["centroid"].y / 1000.0
    gdf["name_clean"] = gdf["NAME"].apply(_clean_name)

    # load_rel1871 already filters to Type-0 records (393 counties).
    rel = load_rel1871()
    rel = rel[rel["Code"] < 900].copy()
    rel["name_clean"] = rel["Kreis"].apply(_clean_name)
    rel_t0 = rel  # alias to keep the rest of the function readable

    merged = rel_t0[["Code", "name_clean"]].merge(
        gdf[["name_clean", "x_km", "y_km"]],
        on="name_clean",
        how="inner",
    ).drop_duplicates(subset="Code")

    coverage = len(merged) / len(rel_t0)
    logger.info(
        "Centroid crosswalk: %d / %d Galloway counties matched (%.1f%%)",
        len(merged), len(rel_t0), 100 * coverage,
    )
    if coverage < 0.7:
        logger.warning(
            "Low centroid match rate. Conley SEs will be computed only on "
            "the matched subset; check name normalization in _clean_name."
        )

    return merged[["Code", "x_km", "y_km"]].sort_values("Code").reset_index(drop=True)


def load_centroids(rebuild: bool = False) -> pd.DataFrame:
    """Load the cached centroid crosswalk, building it if missing."""
    if rebuild or not CENTROIDS_PATH.exists():
        cross = build_centroid_crosswalk()
        CENTROIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        cross.to_parquet(CENTROIDS_PATH, index=False)
        logger.info("Wrote %s", CENTROIDS_PATH)
        return cross
    return pd.read_parquet(CENTROIDS_PATH)


def add_bishop_distance(centroids: pd.DataFrame) -> pd.DataFrame:
    """
    Append ``km_bishop`` to a centroid crosswalk: distance (in km) to the
    nearest 1871-Prussian Catholic bishop's seat. Used as an alternative
    instrument for Catholic-share treatment intensity (Becker--Woessmann
    style, with positive rather than negative sign on cath_share).
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    bishop_xy_km = []
    for name, (lat, lon) in BISHOP_SEATS.items():
        x, y = transformer.transform(lon, lat)
        bishop_xy_km.append((x / 1000.0, y / 1000.0))
    bishops = np.array(bishop_xy_km)  # shape (n_bishops, 2)

    coords = centroids[["x_km", "y_km"]].values  # (n_counties, 2)
    diff = coords[:, None, :] - bishops[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    nearest = dist.min(axis=1)

    out = centroids.copy()
    out["km_bishop"] = nearest
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    cross = build_centroid_crosswalk()
    cross_with_bishop = add_bishop_distance(cross)
    CENTROIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cross_with_bishop.to_parquet(CENTROIDS_PATH, index=False)
    logger.info("Wrote %s with %d counties and km_bishop column",
                CENTROIDS_PATH, len(cross_with_bishop))
