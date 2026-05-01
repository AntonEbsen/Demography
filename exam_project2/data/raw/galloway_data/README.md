# Galloway Prussia Database

This directory contains the core vital statistics and census files for the Prussian provinces (1861–1914), sourced from the **Galloway Database**.

## Contents

- `VIT*.xlsx`: Annual vital registration files containing total births (legitimate/illegitimate), deaths, and marriages.
- `POP*.xlsx`: Quinquennial population census files (1861, 1864, 1867, 1871, 1875, 1880, 1885, 1890) used for denominator interpolation.
- `REL1871.xlsx`: The 1871 Religion Census, providing the share of Catholics and Protestants at the county level.

## Data Structure

The files utilize a "Type" column to distinguish between total county (0), city (1), and rural (2) aggregates. Our analysis primarily utilizes Type 0 to ensure consistent regional coverage.
