"""Validation gates for the DICOM manifest.

Runs the versioned Pandera contract (contracts.ManifestSchema) plus
cross-checks over the manifest, then:

* marks every series with the eligibility rule (eligibility.py),
* writes quarantined series (ineligible, or failing validation) to a
  quarantine report instead of silently dropping them,
* exits 0 on success, 1 on validation failure, 2 on bad input.

This is the gate that prevents the two historical failure modes of this
project's analysis CSVs: non-diagnostic series (SCOUT/localizer) entering
the analysis table, and malformed manifest rows (duplicates, impossible HU
ranges, identifiable patient fields) flowing downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandera as pa
import pandas as pd

from . import contracts, eligibility
from .config import PipelineSettings, apply_cli_overrides
from .log import get_logger, setup_logging

logger = get_logger(__name__)

VALIDATION_REPORT_VERSION = "1.0"


def extra_checks(df: pd.DataFrame):
    """Cross-row checks the schema cannot express.

    Returns (failures, notes). A scout/localizer series marked eligible is a
    hard failure; anything else is a note for the report.
    """
    failures, notes = [], []

    for _, row in df.iterrows():
        desc = str(row.get("series_description", "") or "").lower()
        if any(tok in desc for tok in eligibility.SCOUT_TOKENS):
            if row.get("eligibility_ok"):
                failures.append(
                    {
                        "check": "no_scout_in_analysis",
                        "series_uid": row["series_uid"],
                        "detail": (
                            f"scout-like series '{row['series_description']}' marked "
                            f"eligible: {row.get('eligibility_reason')}"
                        ),
                    }
                )

    n_studies = df.groupby("patient_id")["study_uid"].nunique()
    for pid, n in n_studies[n_studies > 1].items():
        notes.append(f"patient {pid} has {n} study UIDs")

    for _, row in df.iterrows():
        p = row.get("nifti_path", "")
        if p and not os.path.exists(p):
            failures.append(
                {
                    "check": "nifti_exists",
                    "series_uid": row["series_uid"],
                    "detail": f"nifti_path missing: {p}",
                }
            )
    return failures, notes


def validate_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    """Apply eligibility + schema + cross-checks. Returns (df, failures, notes)."""
    df = eligibility.apply(df)
    failures: list = []
    try:
        contracts.ManifestSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        for case in exc.failure_cases.itertuples():
            failures.append(
                {
                    "check": f"schema:{case.check}",
                    "column": case.column,
                    "detail": str(case.failure_case),
                }
            )
    cross_failures, notes = extra_checks(df)
    failures.extend(cross_failures)
    return df, failures, notes


def validate(manifest_path: str, out_dir: str) -> tuple[bool, str, dict]:
    """Validate manifest; write quarantine report; return (ok, report_path, report)."""
    try:
        df = contracts.read_manifest(manifest_path)
    except contracts.ContractError:
        logger.exception("contract_mismatch", manifest_path=manifest_path)
        raise
    df, failures, notes = validate_manifest(df)

    quarantined = df[~df["eligibility_ok"]].copy()
    quarantine_rows = [
        {
            "series_uid": r["series_uid"],
            "series_description": r["series_description"],
            "modality": r["modality"],
            "slice_count": int(r["slice_count"]),
            "reason": r["eligibility_reason"],
        }
        for _, r in quarantined.iterrows()
    ]

    report = {
        "report_version": VALIDATION_REPORT_VERSION,
        "manifest": os.path.abspath(manifest_path),
        "manifest_contract": contracts.MANIFEST_CONTRACT,
        "n_series": len(df),
        "n_eligible": int(df["eligibility_ok"].sum()),
        "n_quarantined": len(quarantine_rows),
        "quarantined": quarantine_rows,
        "validation_failures": failures,
        "notes": notes,
    }
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "validation_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    return not failures, report_path, report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Validate the DICOM manifest against the versioned contract; "
        "quarantine ineligible series. Exits 1 on validation failure."
    )
    ap.add_argument("--manifest", default=None, help="manifest parquet path")
    ap.add_argument("--out-dir", default=None, help="directory for the validation report")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = apply_cli_overrides(PipelineSettings(), args, ["out_dir"])
    if args.manifest is not None:
        settings_manifest = args.manifest
    else:
        settings_manifest = os.path.join(settings.out_dir, settings.manifest_name)
    setup_logging(settings.log_level, settings.log_format)

    if not os.path.isfile(settings_manifest):
        logger.error("manifest_not_found", path=settings_manifest)
        return 2
    try:
        ok, report_path, report = validate(settings_manifest, settings.out_dir)
    except contracts.ContractError:
        return 1

    logger.info(
        "validate_done",
        ok=ok,
        n_series=report["n_series"],
        n_eligible=report["n_eligible"],
        n_quarantined=report["n_quarantined"],
        n_failures=len(report["validation_failures"]),
        report_path=report_path,
    )
    for q in report["quarantined"]:
        logger.warning("series_quarantined", **q)
    for f in report["validation_failures"]:
        logger.error("validation_failure", **f)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
