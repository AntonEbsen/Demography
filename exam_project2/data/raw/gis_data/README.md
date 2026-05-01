# GIS & Spatial Data

This directory contains the geospatial files required for mapping and spatial econometrics.

## Contents

- `prussia_1871.geojson` / `*.shp`: Historical shapefiles representing the administrative boundaries of Prussian counties (Kreise) as of the 1871 unification.
- `rb_boundaries.json`: Geometric outlines of the *Regierungsbezirke* (Administrative Districts) used for fixed-effects grouping.

## Coordinate System

All files are standardized to **EPSG:4326 (WGS84)** for compatibility with modern web-mapping libraries (Leaflet/Mapbox) and GeoPandas.

## Source

Historical boundaries are harmonized from the [MPIDR Population History of Germany](https://www.demogr.mpg.de/) and adjusted to match the Galloway county codes.
