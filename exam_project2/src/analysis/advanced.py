"""
advanced.py
===========
Backward-compatible re-export module.

All functions have been split into focused sub-modules:
  - rollback.py          : rollback_event_study
  - channels.py          : illegitimacy_analysis, infant_mortality_analysis
  - war_robustness.py    : franco_prussian_war_analysis, robustness_exclude_war
  - trend_and_placebo.py : trend_adjusted_did, placebo_test
  - polish_german.py     : polish_german_rollback

This file re-exports them all so existing notebook imports continue to work.
"""

from src.analysis.rollback import rollback_event_study
from src.analysis.channels import illegitimacy_analysis, infant_mortality_analysis
from src.analysis.war_robustness import franco_prussian_war_analysis, robustness_exclude_war
from src.analysis.trend_and_placebo import trend_adjusted_did, placebo_test
from src.analysis.polish_german import polish_german_rollback

__all__ = [
    "rollback_event_study",
    "illegitimacy_analysis",
    "infant_mortality_analysis",
    "franco_prussian_war_analysis",
    "robustness_exclude_war",
    "trend_adjusted_did",
    "placebo_test",
    "polish_german_rollback",
]