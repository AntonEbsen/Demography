"""
multiple_testing.py
===================
Multiple-testing corrections for jointly reporting several outcome variables.

Implements the Anderson (2008, JASA) sharpened q-value procedure: a
Benjamini--Hochberg FDR correction with monotonicity enforced by stepping
up from the largest p-value. With M outcomes, the q-value of the k-th
smallest p-value is

    q_(k) = min_{j >= k} ( M / j ) * p_(j)

clipped to [0, 1]. Sharpening eliminates the non-monotonic artefacts
that the raw BH procedure can produce in finite samples.

Usage::

    from src.analysis.multiple_testing import sharpened_q_values
    qs = sharpened_q_values({"cbr": 0.001, "legitimate_br": 0.045, ...})
"""

from __future__ import annotations

from typing import Dict, Mapping


def sharpened_q_values(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Anderson (2008) sharpened FDR q-values for a dict of named p-values."""
    if not p_values:
        return {}
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    # Step 1: raw BH q-values (M / k * p_(k))
    raw = [(name, p, (m / (k + 1)) * p) for k, (name, p) in enumerate(items)]
    # Step 2: enforce monotonicity by stepping up from the largest
    sharpened = [0.0] * m
    sharpened[-1] = min(raw[-1][2], 1.0)
    for k in range(m - 2, -1, -1):
        sharpened[k] = min(raw[k][2], sharpened[k + 1], 1.0)
    return {raw[k][0]: sharpened[k] for k in range(m)}
