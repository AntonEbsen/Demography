# Monograph Pages

This directory defines the routing and content structure of the digital research monograph.

## Page Architecture

- `index.astro`: The executive summary, high-impact stats, and "Audio Brief."
- `methods.astro`: Detailed identification strategy, LaTeX formulas, and data-flow visualizations.
- `evidence.astro`: The empirical core—interactive trajectory explorers, robustness browsers, and Prussian maps.
- `context.astro`: Historical background, the "Hajnal Line" analysis, and the "Bismarck vs. Pope" Tension Meter.
- `data.astro`: The Data Hub—variable dictionaries, pipeline lineage, and archive quality maps.
- `appendix.astro`: Sensitivity analyses, technical notes, and print-ready formatting.

## Tech Specs

All pages are built using **Astro**, enabling zero-JavaScript by default while hydrating interactive components (like the County Explorer) only where needed for maximum performance.
