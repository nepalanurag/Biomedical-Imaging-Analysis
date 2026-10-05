"""Series-eligibility rule for the analysis table.

Only axial diagnostic CT series belong in the quantification table.
Excluded as code (case-insensitive match on Series Description):

- scout / localizer series ("SCOUT CHEST", "Scout", "PE SCOUT", ...)
- bolus-tracking / smart-prep series ("Smart Prep Series", ...)
- coronal / sagittal / MIP reformats ("COR 3X3", "SAG 3X3", "PE MIP COR 15X5", ...)

Rationale: localizer and reformat series are not diagnostic axial acquisitions;
quantifying "infection percentage" on them is meaningless, and several produced
impossible >100% values under the old whole-image numerator. Series whose
description is uninformative ("NO DETAILS") are kept — we cannot judge what we
cannot see — and flagged in the curation log.
"""
from __future__ import annotations

import re

_EXCLUDE_PATTERNS = [
    re.compile(r"scout", re.IGNORECASE),          # localizer scans
    re.compile(r"smart.?prep", re.IGNORECASE),    # bolus-tracking series
    re.compile(r"\bmip\b", re.IGNORECASE),        # maximum-intensity projections
    re.compile(r"\bcor\b|\bcoronal\b", re.IGNORECASE),   # coronal reformats
    re.compile(r"\bsag\b|\bsagittal\b", re.IGNORECASE), # sagittal reformats
]


def is_eligible_series(series_description: str | None) -> bool:
    """True if the series is an axial diagnostic acquisition.

    Uninformative descriptions ("NO DETAILS", empty) return True: they are
    kept and flagged, not silently dropped.
    """
    if not series_description or not series_description.strip():
        return True
    desc = series_description.strip()
    if desc.upper() == "NO DETAILS":
        return True
    return not any(p.search(desc) for p in _EXCLUDE_PATTERNS)


def eligibility_reason(series_description: str | None) -> str:
    """Human-readable reason, for the curation log."""
    if not series_description or not series_description.strip():
        return "kept: empty description, flagged for review"
    desc = series_description.strip()
    if desc.upper() == "NO DETAILS":
        return "kept: uninformative description, flagged for review"
    for p in _EXCLUDE_PATTERNS:
        if p.search(desc):
            return f"excluded: matches '{p.pattern}'"
    return "kept: axial diagnostic series"
