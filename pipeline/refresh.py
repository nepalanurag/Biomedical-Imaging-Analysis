"""Scheduled quantification refresh.

Re-runs quantification over the versioned manifest and writes versioned
results (Parquet + CSV) stamped with the run timestamp, the code hash of
quantify.py, and the git SHA. Old result versions are kept, so every number
in the results table is traceable to the exact manifest row and code version
that produced it.

Intended use: cron / scheduled job, e.g.
    0 3 * * * cd /path/to/repo && python -m pipeline.refresh

The refresh is idempotent: re-running with an unchanged manifest and code
simply writes a new versioned run with the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


from . import contracts, quantify
from .config import PipelineSettings, apply_cli_overrides
from .log import get_logger, setup_logging

logger = get_logger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "nogit"


def refresh(manifest_path: str, results_dir: str, out_dir: str, lung_band, inf_band) -> dict:
    """Run quantification; write a versioned result set. Returns metadata."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"v{ts}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    rdf = quantify.run_quantification(manifest_path, lung_band, inf_band)
    rdf["run_ts"] = ts
    rdf["version"] = version
    rdf["git_sha"] = _git_sha()
    if len(rdf):
        contracts.ResultsSchema.validate(rdf, lazy=True)

    base = os.path.join(results_dir, f"quantification_{version}")
    contracts.write_results(rdf, base + ".parquet")
    rdf.to_csv(base + ".csv", index=False)

    for ext in ("parquet", "csv"):
        latest = os.path.join(results_dir, f"quantification_latest.{ext}")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(base) + f".{ext}", latest)
        except OSError:
            logger.warning("latest_symlink_failed", path=latest)

    return {
        "version": version,
        "run_ts": ts,
        "manifest": os.path.abspath(manifest_path),
        "manifest_contract": contracts.MANIFEST_CONTRACT,
        "results_contract": contracts.RESULTS_CONTRACT,
        "n_quantified": len(rdf),
        "code_hash": quantify.code_hash(),
        "git_sha": _git_sha(),
        "files": [base + ".parquet", base + ".csv"],
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Scheduled refresh: re-quantify the manifest, write versioned results."
    )
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--lung-lo", type=int, default=None)
    ap.add_argument("--lung-hi", type=int, default=None)
    ap.add_argument("--inf-lo", type=int, default=None)
    ap.add_argument("--inf-hi", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = apply_cli_overrides(
        PipelineSettings(),
        args,
        ["out_dir", "results_dir", "lung_lo", "lung_hi", "inf_lo", "inf_hi"],
    )
    setup_logging(settings.log_level, settings.log_format)
    try:
        settings.validate_bands()
    except ValueError as exc:
        logger.error("bad_bands", error=str(exc))
        return 2
    manifest_path = args.manifest or os.path.join(settings.out_dir, settings.manifest_name)
    if not os.path.isfile(manifest_path):
        logger.error("manifest_not_found", path=manifest_path)
        return 2

    try:
        meta = refresh(
            manifest_path,
            settings.results_dir,
            settings.out_dir,
            lung_band=(settings.lung_lo, settings.lung_hi),
            inf_band=(settings.inf_lo, settings.inf_hi),
        )
    except Exception:
        logger.exception("refresh_failed", manifest=manifest_path)
        return 1

    meta_path = os.path.join(settings.out_dir, f"refresh_{meta['version']}.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("refresh_done", **{k: v for k, v in meta.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
