"""Infection quantification: HU-threshold segmentation + volume metrics.

Implements the project's lung/infection segmentation pipeline as importable
numpy/scipy functions (no ITK dependency), with the infection numerator
defined as infected voxels INTERSECTED with the lung mask.

This is the single source of truth for the metric. The fixed formula matches
the intersection version in segmentation.py
(COVIDLungSegmentation.quantify_infection): the numerator counts only voxels
inside the lung, so the percentage can never exceed 100. An earlier
whole-image numerator is never reintroduced here.

Pipeline parameters (project standard, overridable via config/CLI):
  lung mask:      HU in [-950, -300], fill holes, largest connected region, smooth
  infection mask: HU in [-700, -200], median filter

Usage:
    python -m pipeline.quantify --manifest pipeline_data/manifest.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import contracts, eligibility
from .config import PipelineSettings, apply_cli_overrides
from .log import bind, get_logger, setup_logging

logger = get_logger(__name__)

# Default HU bands (project standard).
LUNG_LO, LUNG_HI = -950, -300
INF_LO, INF_HI = -700, -200


def quantify_infection(lung_mask, infection_mask) -> dict:
    """Fixed infection metric.

    Infected voxels are counted only INSIDE the lung mask, then divided by
    lung voxels. Matches the intersection version in segmentation.py; the
    percentage is structurally bounded by [0, 100].
    """
    lung_array = np.asarray(lung_mask) > 0
    mask_array = np.asarray(infection_mask) > 0
    total_voxels = int(np.sum(lung_array))
    infected_voxels = int(np.sum(mask_array & lung_array))
    infection_percentage = 100.0 * infected_voxels / total_voxels if total_voxels else 0.0
    return {
        "total_voxels": total_voxels,
        "infected_voxels": infected_voxels,
        "infection_percentage": infection_percentage,
    }


def segment_lungs(vol: np.ndarray, lung_lo: int = LUNG_LO, lung_hi: int = LUNG_HI) -> np.ndarray:
    """Threshold -> fill holes -> largest connected component -> smooth."""
    m = (vol >= lung_lo) & (vol <= lung_hi)
    m = ndi.binary_fill_holes(m)
    lab, nlab = ndi.label(m)
    if nlab == 0:
        return np.zeros_like(m)
    sizes = ndi.sum(m, lab, range(1, nlab + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    struct = np.ones((5, 5), dtype=bool)
    out = np.empty_like(m)
    for i in range(m.shape[0]):
        out[i] = ndi.binary_closing(m[i], structure=struct)
    return out


def segment_infection(vol: np.ndarray, inf_lo: int = INF_LO, inf_hi: int = INF_HI) -> np.ndarray:
    """Threshold -> slice-wise median filter (stands in for ITK 3D radius 2)."""
    m = (vol >= inf_lo) & (vol <= inf_hi)
    out = np.empty_like(m)
    for i in range(m.shape[0]):
        out[i] = ndi.median_filter(m[i].astype(np.uint8), size=5) > 0
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    d = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - d) / den, (c + d) / den


def load_volume(path: str) -> np.ndarray:
    """Load an HU volume from a NIfTI file or a directory of DICOM slices."""
    if os.path.isdir(path):
        from .ingest import find_dicom_files, read_series_volume

        files = find_dicom_files(path)
        vol, _ds0 = read_series_volume(files)
        return vol
    import nibabel as nib

    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    # ingest.py stored (x, y, z); restore (z=n, y, x)
    return np.transpose(data, (2, 1, 0))


def quantify_series(
    row,
    lung_band: tuple[int, int] = (LUNG_LO, LUNG_HI),
    inf_band: tuple[int, int] = (INF_LO, INF_HI),
) -> dict:
    """Segment + quantify one manifest row. Returns a result dict."""
    src = row["nifti_path"] if row.get("nifti_path") else row["source_dir"]
    if not src:
        raise ValueError(f"series {row['series_uid']}: no nifti_path or source_dir")
    vol = load_volume(src)
    lung = segment_lungs(vol, *lung_band)
    inf = segment_infection(vol, *inf_band)
    q = quantify_infection(lung, inf)
    lo, hi = wilson_ci(q["infected_voxels"], q["total_voxels"])
    return {
        "series_uid": row["series_uid"],
        "patient_id": row["patient_id"],
        "lung_band": f"[{lung_band[0]},{lung_band[1]}]",
        "inf_band": f"[{inf_band[0]},{inf_band[1]}]",
        "slices": vol.shape[0],
        "lung_voxels": q["total_voxels"],
        "infected_voxels": q["infected_voxels"],
        "infection_pct": q["infection_percentage"],
        "ci_low": 100.0 * lo,
        "ci_high": 100.0 * hi,
        "source": src,
    }


def code_hash() -> str:
    """Short hash of this module's source, stamped on every result row."""
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def run_quantification(manifest_path: str, lung_band, inf_band) -> pd.DataFrame:
    """Quantify every eligible series in the manifest."""
    df = contracts.read_manifest(manifest_path)
    df = eligibility.apply(df)
    eligible = df[df["eligibility_ok"]]
    skipped = df[~df["eligibility_ok"]]
    logger.info("eligibility_filter", eligible=len(eligible), quarantined=len(skipped))

    results = []
    for _, row in eligible.iterrows():
        slog = bind(logger, series_uid=str(row["series_uid"])[:32])
        try:
            res = quantify_series(row, lung_band=lung_band, inf_band=inf_band)
            results.append(res)
            slog.info(
                "series_quantified",
                infection_pct=round(res["infection_pct"], 2),
                ci_low=round(res["ci_low"], 2),
                ci_high=round(res["ci_high"], 2),
            )
        except Exception as exc:
            slog.error("quantification_failed", error=str(exc))

    rdf = pd.DataFrame(results)
    if len(rdf):
        rdf["code_hash"] = code_hash()
    # refresh.py overwrites these; direct runs leave them null.
    for col in ("run_ts", "version", "git_sha"):
        if col not in rdf.columns:
            rdf[col] = None
    return rdf


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Quantify lung infection for every eligible series in the manifest."
    )
    ap.add_argument("--manifest", default=None, help="manifest parquet path")
    ap.add_argument("--out-dir", default=None, help="output directory")
    ap.add_argument("--results-name", default="quantification.parquet")
    ap.add_argument("--lung-lo", type=int, default=None)
    ap.add_argument("--lung-hi", type=int, default=None)
    ap.add_argument("--inf-lo", type=int, default=None)
    ap.add_argument("--inf-hi", type=int, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = apply_cli_overrides(
        PipelineSettings(), args, ["out_dir", "lung_lo", "lung_hi", "inf_lo", "inf_hi"]
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
        rdf = run_quantification(
            manifest_path,
            lung_band=(settings.lung_lo, settings.lung_hi),
            inf_band=(settings.inf_lo, settings.inf_hi),
        )
        os.makedirs(settings.out_dir, exist_ok=True)
        out_path = os.path.join(settings.out_dir, args.results_name)
        contracts.write_results(rdf, out_path)
        # Schema-level assertion: the contract itself bounds infection_pct.
        if len(rdf):
            contracts.ResultsSchema.validate(rdf, lazy=True)
    except Exception:
        logger.exception("quantify_failed", manifest=manifest_path)
        return 1

    logger.info(
        "quantify_done",
        n_series=len(rdf),
        mean_infection_pct=round(float(rdf["infection_pct"].mean()), 2) if len(rdf) else 0.0,
        max_infection_pct=round(float(rdf["infection_pct"].max()), 2) if len(rdf) else 0.0,
        results_path=os.path.abspath(out_path),
        code_hash=code_hash(),
        contract=contracts.RESULTS_CONTRACT,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
