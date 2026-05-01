"""
utils.py
========
Shared utility functions for panel econometric analysis.

This module contains common helpers used across multiple analysis modules
to avoid code duplication.
"""

import logging
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults
from typing import List

logger = logging.getLogger(__name__)


def safe_panel_ols(
    df: pd.DataFrame, 
    outcome: str, 
    exog_vars: List[str], 
    entity: str = "Code", 
    time: str = "Year"
) -> PanelEffectsResults:
    """
    Deduplicate, drop NaN, set multi-index, and fit PanelOLS with
    entity-clustered standard errors.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (not yet indexed).
    outcome : str
        Name of the dependent variable column.
    exog_vars : list of str
        Names of the exogenous regressor columns.
    entity : str
        Column used as the entity (panel) identifier.
    time : str
        Column used as the time identifier.

    Returns
    -------
    linearmodels.panel.results.PanelEffectsResults
        Fitted PanelOLS result object.
    """
    cols_needed = [entity, time, outcome] + exog_vars
    sub = df[cols_needed].drop_duplicates(subset=[entity, time]).dropna().copy()
    sub = sub.set_index([entity, time])

    y = sub[outcome]
    X = sub[exog_vars]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True)
