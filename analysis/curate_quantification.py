#!/usr/bin/env python3
"""Analysis-table curation: apply the series-eligibility rule.

Reads infection_quantification_by_subject.csv, drops non-diagnostic series
(scout/localizer, smart-prep, coronal/sagittal/MIP reformats) via
analysis.series_eligibility, and writes
infection_quantification_by_subject_curated.csv plus a curation log.

The original CSV is left untouched; the curated table is added alongside.

Usage:
    python analysis/curate_quantification.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from series_eligibility import is_eligible_series, eligibility_reason

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "infection_quantification_by_subject.csv")
DST = os.path.join(ROOT, "infection_quantification_by_subject_curated.csv")
LOG = os.path.join(ROOT, "analysis", "curation_log.csv")


def main() -> None:
    with open(SRC, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    kept, dropped = [], []
    for r in rows:
        desc = r.get("Series Description", "")
        if is_eligible_series(desc):
            kept.append(r)
        else:
            dropped.append(r)

    with open(DST, "w", newline="") as fh:
        out_fields = fieldnames + ["curation_flag"]
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        for r in kept:
            # Rows computed under the pre-consolidation whole-image numerator can
            # exceed 100%; they are kept as-is and flagged for recompute on the
            # next refresh with the consolidated intersection formula.
            flag = ""
            if float(r.get("infection_percentage") or 0) > 100:
                flag = ("pre-consolidation numerator; kept as-is, "
                        "recompute on refresh with the intersection formula")
            w.writerow({**r, "curation_flag": flag})

    with open(LOG, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Series UID", "Series Description", "decision"])
        for r in dropped:
            w.writerow([r.get("Series UID", ""), r.get("Series Description", ""),
                        eligibility_reason(r.get("Series Description", ""))])
        for r in kept:
            reason = eligibility_reason(r.get("Series Description", ""))
            if "flagged" in reason:
                w.writerow([r.get("Series UID", ""), r.get("Series Description", ""), reason])

    over = [r for r in kept if float(r.get("infection_percentage") or 0) > 100]
    print(f"input rows:    {len(rows)}")
    print(f"kept:          {len(kept)}")
    print(f"dropped:       {len(dropped)} (non-diagnostic series)")
    print(f"kept rows >100%: {len(over)}")
    print(f"wrote {os.path.basename(DST)} and analysis/curation_log.csv")


if __name__ == "__main__":
    main()
