# Data Dictionary: Victorian Demographic Transition (1851-1881)

This codebook describes the variables used in the "Cost of Quality" research project. The dataset is a longitudinal panel at the Registration District (RD) level for England and Wales.

## Core Variables

| Variable | Description | Units | Source |
| :--- | :--- | :--- | :--- |
| `REGDIST` | Registration District Name | String | Populations Past |
| `POP` | Total Population in Census Year | Count | Populations Past |
| `POP_DENS` | Population Density | People per Acre | Calculated |
| `TFR` | Total Fertility Rate | Children per Woman | Populations Past |
| `IMR` | Infant Mortality Rate | Deaths per 1000 births | Populations Past |
| `F_TEX` | Female Employment in Textiles | Percentage (%) | Populations Past |
| `F_CL_1013` | Female Child Labor (Ages 10-13) | Participation rate (%) | Populations Past |
| `is_textile` | Treatment Indicator (Textile District) | Binary (0/1) | Analysis Derived |
| `geometry` | Boundary Polygons | WKT/GeoJSON | Populations Past (GIS) |

## Data Provenance
Most variables are derived from the **Populations Past** platform, which digitizes the Integrated Census Microdata (I-CeM) for the 1851, 1861, 1871, and 1881 censuses.

## Methodology Note
The `is_textile` indicator is defined at the 1851 baseline. Districts with `F_TEX` values above the national median in 1851 are assigned to the treatment group to ensure identification via the initial industrial structure.
