---
title: "The Genesis of the Factory Acts Dataset"
date: 2026-04-09
summary: "An overview of how we cleaned and merged the Victorian census snapshots into a unified spatial panel."
tags: ["dataset", "spatial", "history"]
---

## Harmonizing the Census

Building a long-run spatial panel for 19th-century Britain involves significant hurdles. The primary challenge is the **shifting administrative boundaries** of Registration Districts (RDs).

### The Methodology

We utilized a "Constant Boundary" approach, where we mapped statistical returns from 1851, 1861, 1871, and 1881 onto a unified geospatial framework.

1. **Spatial Join**: Every census village was matched to its parent Registration District.
2. **Textile Intensity**: We calculated the percentage of the female workforce engaged in textile manufacturing.
3. **Fertility Rates**: We derived age-specific fertility rates using the own-child method.

### Initial Findings

Preliminary regressions suggest a strong negative correlation between textile intensity and fertility *after* the 1833 shock, supporting the **Q-Q Trade-off** model.

```python
# Simple preview of the regression logic
results = model.fit(cov_type='HC1')
print(results.summary())
```

Stay tuned for our next entry on **Oster's Delta** and unobserved selection!
