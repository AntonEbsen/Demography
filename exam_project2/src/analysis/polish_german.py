"""
polish_german.py
================
Polish vs German Catholic × Rollback interaction analysis.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.utils import safe_panel_ols

logger = logging.getLogger(__name__)


def polish_german_rollback(
    df: pd.DataFrame, outcome: str = "cbr", savepath: str = None,
):
    """Does the Polish-vs-German divergence reverse during the rollback period?"""
    logger.info("POLISH x ROLLBACK vs GERMAN x ROLLBACK")

    df = df.copy()
    df["enforcement"] = df["Year"].between(1873, 1878).astype(int)
    df["rollback"] = df["Year"].between(1880, 1887).astype(int)
    df["post_rollback"] = (df["Year"] >= 1888).astype(int)
    df["cath_x_enforcement"] = df["cath_share"] * df["enforcement"]
    df["cath_x_rollback"] = df["cath_share"] * df["rollback"]
    df["cath_x_postrollback"] = df["cath_share"] * df["post_rollback"]

    polish_rbs = ["POS", "BRO"]
    german_cath_rbs = ["KOL", "KOB", "TRI", "AAC", "OPP", "MUN"]

    sub_results = {}
    for label, mask in [
        ("Polish high-Catholic counties", df["Rb"].isin(polish_rbs)),
        ("German high-Catholic counties", df["Rb"].isin(german_cath_rbs)),
        ("Protestant low-Catholic counties (rest)", ~df["Rb"].isin(polish_rbs + german_cath_rbs)),
    ]:
        sub = df[mask].copy()
        if sub["Code"].nunique() < 10:
            continue
        exog = ["cath_x_enforcement", "cath_x_rollback", "cath_x_postrollback", "ln_pop"]
        res = safe_panel_ols(sub, outcome, exog)
        sub_results[label] = {
            "enforcement": {"coef": res.params["cath_x_enforcement"], "se": res.std_errors["cath_x_enforcement"], "p": res.pvalues["cath_x_enforcement"]},
            "rollback": {"coef": res.params["cath_x_rollback"], "se": res.std_errors["cath_x_rollback"], "p": res.pvalues["cath_x_rollback"]},
            "post_rollback": {"coef": res.params["cath_x_postrollback"], "se": res.std_errors["cath_x_postrollback"], "p": res.pvalues["cath_x_postrollback"]},
            "n_counties": sub["Code"].nunique(),
        }
        for period_label, key in [("Enforcement", "enforcement"), ("Rollback", "rollback"), ("Post-rollback", "post_rollback")]:
            r = sub_results[label][key]
            logger.info("  %s — %s: b=%+.4f (SE=%.4f, p=%.3f)", label, period_label, r["coef"], r["se"], r["p"])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Polish provinces": "#C0392B", "German Catholic provinces": "#2471A3", "Protestant provinces (rest)": "#555555"}
    period_labels = ["Enforcement\n(1873-1878)", "Rollback\n(1880-1887)", "Post-rollback\n(1888+)"]
    period_keys = ["enforcement", "rollback", "post_rollback"]
    width = 0.25
    x = np.arange(len(period_labels))
    for i, (label, r) in enumerate(sub_results.items()):
        if label not in colors:
            continue
        coefs = [r[k]["coef"] for k in period_keys]
        ses = [r[k]["se"] for k in period_keys]
        ax.bar(x + (i - 1) * width, coefs, width=width, yerr=[1.96*s for s in ses],
               color=colors[label], alpha=0.8, label=label, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(period_labels)
    ax.set_ylabel("Coefficient on cath_share x period", fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return {"results": sub_results, "fig": fig}
