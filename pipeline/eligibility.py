"""Series-eligibility rule, as code.

Decides whether a CT series may enter the infection-analysis table.
This is the rule that structurally prevents the historical failure mode of
non-diagnostic series (SCOUT/localizer, reformatted/derived reconstructions)
leaking into quantification results.

A series is ELIGIBLE only if ALL of the following hold:

1. Modality is CT.
2. The series description does not name a localizer/scout/topogram survey.
3. The ImageType is ORIGINAL (not DERIVED/SECONDARY reformats, MIPs, etc.).
4. The acquisition geometry is axial (ImageOrientationPatient close to the
   canonical axial direction cosines [1,0,0,0,1,0]).

Anything failing a check is rejected with a machine-readable reason, and the
validator (validate.py) quarantines rejected series instead of silently
dropping them.

The rule set is deliberately conservative: a series that cannot be proven
axial and diagnostic is rejected. If a rejected series is later judged fine
by hand, whitelist it explicitly in ``ELIGIBILITY_OVERRIDES`` with a comment
explaining why, never by weakening the default rules.
"""

from __future__ import annotations

import re

import numpy as np

# Explicit, auditable overrides. Format: series_uid -> reason.
ELIGIBILITY_OVERRIDES: dict[str, str] = {}

# Tokens that mark a localizer/scout/topogram survey series.
SCOUT_TOKENS = (
    "scout",
    "localizer",
    "topogram",
    "topo",
    "pilot",
    "surview",
    "survey",
    "scanogram",
    "plan scan",
)

# ImageType values that mark a series as reformatted/derived, never primary.
DERIVED_MARKERS = ("derived", "secondary", "reformat", "mip", "minip", "average")

# Canonical axial direction cosines [row_x,row_y,row_z,col_x,col_y,col_z].
AXIAL_IOP = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
# Tolerance on each cosine; real axial acquisitions sit well inside 0.05.
AXIAL_TOLERANCE = 0.05


def _reason(desc, image_type, modality, iop):
    """Return the first failing check's reason, or None if eligible."""
    if modality.strip().upper() != "CT":
        return f"not-CT modality '{modality}'"
    desc_l = (desc or "").lower()
    for token in SCOUT_TOKENS:
        if re.search(r"\b" + re.escape(token) + r"\b", desc_l):
            return f"localizer/scout series (matched '{token}' in description)"
    itype_l = (image_type or "").lower().replace("\\", " ")
    for marker in DERIVED_MARKERS:
        if marker in itype_l:
            return f"derived/reformatted series (ImageType contains '{marker}')"
    if iop:
        try:
            vec = np.array([float(x) for x in iop.split(",")], dtype=float)
            if vec.shape != (6,) or np.any(np.abs(vec - AXIAL_IOP) > AXIAL_TOLERANCE):
                return "non-axial acquisition geometry"
        except (ValueError, AttributeError):
            return "unparseable ImageOrientationPatient"
    return None


def check_series(row) -> tuple[bool, str]:
    """Eligibility for one manifest row (dict-like). Returns (ok, reason)."""
    uid = str(row.get("series_uid", ""))
    if uid in ELIGIBILITY_OVERRIDES:
        return True, f"whitelisted: {ELIGIBILITY_OVERRIDES[uid]}"
    reason = _reason(
        desc=row.get("series_description", ""),
        image_type=row.get("image_type", ""),
        modality=row.get("modality", ""),
        iop=row.get("image_orientation_patient", ""),
    )
    if reason is None:
        return True, "axial diagnostic CT series"
    return False, reason


def apply(manifest_df):
    """Add eligibility_ok / eligibility_reason columns to a manifest DataFrame."""
    df = manifest_df.copy()
    oks, reasons = [], []
    for _, row in df.iterrows():
        ok, reason = check_series(row)
        oks.append(ok)
        reasons.append(reason)
    df["eligibility_ok"] = oks
    df["eligibility_reason"] = reasons
    return df
