"""DICOM ingestion: walk a DICOM tree, read with pydicom, convert to NIfTI,
and write a versioned Parquet manifest of every series found.

The manifest is the versioned source of truth everything downstream reads:
validation (validate.py), series eligibility (eligibility.py), quantification
(quantify.py) and refreshes (refresh.py). It is stamped with contract
``manifest/1.0`` (see contracts.py).

Designed for the MIDRC-RICORD-1A layout: patient/study/series directory
nesting holding 255 axial DICOM slices per series, but grouping is purely by
SeriesInstanceUID so mixed trees work too.

Usage:
    python -m pipeline.ingest --dicom-root /data/dicom --out-dir pipeline_data
    CTPIPE_DICOM_ROOT=/data/dicom python -m pipeline.ingest
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pydicom
from pydicom.multival import MultiValue

from . import contracts
from .config import PipelineSettings, apply_cli_overrides
from .log import bind, get_logger, setup_logging

logger = get_logger(__name__)


def find_dicom_files(root: str) -> list[str]:
    """All DICOM files under root, recursive."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".dcm", ".dicom", ".ima")):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def group_by_series(files: list[str]) -> dict[str, list[str]]:
    """Group file paths by SeriesInstanceUID (header-only read)."""
    series: dict[str, list[str]] = {}
    for path in files:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception as exc:  # corrupt file: skip, counted downstream
            logger.warning("unreadable_dicom_skipped", path=path, error=str(exc))
            continue
        uid = str(ds.get("SeriesInstanceUID", "UNKNOWN"))
        series.setdefault(uid, []).append(path)
    return series


def _tag_str(ds, keyword: str, default: str = "") -> str:
    val = ds.get(keyword, default)
    if isinstance(val, MultiValue) or isinstance(val, (list, tuple)):
        return ",".join(str(v) for v in val)
    return str(val)


def _tag_float(ds, keyword: str) -> float | None:
    val = ds.get(keyword, None)
    if val is None:
        return None
    try:
        if isinstance(val, (list, tuple)) and len(val):
            return float(val[0])
        return float(val)
    except (TypeError, ValueError):
        return None


def _affine(ds0) -> np.ndarray:
    """Patient-space affine for a single-frame axial stack."""
    ipp = np.array([float(x) for x in ds0.ImagePositionPatient], dtype=float)
    iop = np.array([float(x) for x in ds0.ImageOrientationPatient], dtype=float)
    spacing = np.array([float(x) for x in ds0.PixelSpacing], dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    slice_dir = np.cross(row_dir, col_dir)
    thickness = _tag_float(ds0, "SliceThickness")
    if thickness is None or thickness <= 0:
        thickness = float(np.linalg.norm(slice_dir)) or 1.0
    affine = np.eye(4)
    affine[:3, 0] = row_dir * spacing[1]
    affine[:3, 1] = col_dir * spacing[0]
    affine[:3, 2] = slice_dir * thickness
    affine[:3, 3] = ipp
    return affine


def read_series_volume(paths: list[str]) -> tuple[np.ndarray, object]:
    """Read a slice stack as an HU float32 volume (n, rows, cols)."""
    ordered = []
    for path in paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        pos = ds.get("ImagePositionPatient", None)
        if pos is not None and len(pos) == 3:
            key = float(pos[2])
        else:
            key = float(ds.get("InstanceNumber", 0))
        ordered.append((key, path))
    ordered.sort(key=lambda t: t[0])

    ds0 = pydicom.dcmread(ordered[0][1])
    slope = float(ds0.get("RescaleSlope", 1))
    intercept = float(ds0.get("RescaleIntercept", 0))
    rows, cols = int(ds0.Rows), int(ds0.Columns)

    vol = np.empty((len(ordered), rows, cols), dtype=np.float32)
    for i, (_key, path) in enumerate(ordered):
        ds = pydicom.dcmread(path)
        if (int(ds.Rows), int(ds.Columns)) != (rows, cols):
            raise ValueError(
                f"geometry mismatch in {path}: " f"{ds.Rows}x{ds.Columns} vs {rows}x{cols}"
            )
        vol[i] = ds.pixel_array.astype(np.float32) * slope + intercept
    return vol, ds0


def convert_to_nifti(vol: np.ndarray, ds0, out_path: str) -> str:
    """Write an HU volume to NIfTI (nibabel), preserving patient geometry."""
    import nibabel as nib

    affine = _affine(ds0)
    # nibabel expects (x, y, z); DICOM stack is (z=n, y=rows, x=cols)
    data = np.transpose(vol, (2, 1, 0))
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_xyzt_units(xyz="mm")
    img.header["descrip"] = b"HU volume from DICOM; see manifest for provenance"
    nib.save(img, out_path)
    return out_path


def build_manifest(root: str, out_dir: str, write_nifti: bool = True) -> pd.DataFrame:
    """Walk root, convert each series, return the manifest DataFrame."""
    files = find_dicom_files(root)
    if not files:
        raise ValueError(f"no DICOM files found under {root}")
    grouped = group_by_series(files)
    if not grouped:
        raise ValueError(f"no readable DICOM series under {root}")

    nifti_dir = os.path.join(out_dir, "nifti")
    if write_nifti:
        os.makedirs(nifti_dir, exist_ok=True)

    records = []
    for series_uid, paths in sorted(grouped.items()):
        slog = bind(logger, series_uid=series_uid[:32])
        vol, ds0 = read_series_volume(paths)
        short = hashlib.md5(series_uid.encode()).hexdigest()[:12]
        nifti_path = ""
        if write_nifti:
            nifti_path = os.path.join(nifti_dir, f"series_{short}.nii.gz")
            convert_to_nifti(vol, ds0, nifti_path)
        slog.info(
            "series_ingested",
            slices=len(paths),
            description=str(ds0.get("SeriesDescription", "")),
            hu_min=float(np.min(vol)),
            hu_max=float(np.max(vol)),
            nifti=bool(nifti_path),
        )
        records.append(
            {
                "patient_id": _tag_str(ds0, "PatientID"),
                "study_uid": _tag_str(ds0, "StudyInstanceUID"),
                "series_uid": series_uid,
                "series_number": _tag_str(ds0, "SeriesNumber"),
                "series_description": _tag_str(ds0, "SeriesDescription"),
                "modality": _tag_str(ds0, "Modality"),
                "sop_class_uid": _tag_str(ds0, "SOPClassUID"),
                "image_type": _tag_str(ds0, "ImageType"),
                "image_orientation_patient": _tag_str(ds0, "ImageOrientationPatient"),
                "slice_count": len(paths),
                "rows": int(ds0.Rows),
                "cols": int(ds0.Columns),
                "slice_thickness": _tag_float(ds0, "SliceThickness"),
                "pixel_spacing": _tag_str(ds0, "PixelSpacing"),
                "hu_min": float(np.min(vol)),
                "hu_max": float(np.max(vol)),
                "nifti_path": nifti_path,
                "source_dir": os.path.dirname(os.path.commonpath(paths)),
            }
        )
    return pd.DataFrame(records)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Ingest a DICOM tree into NIfTI volumes + a versioned Parquet manifest."
    )
    ap.add_argument(
        "--dicom-root", default=None, help="directory tree with DICOM files (or CTPIPE_DICOM_ROOT)"
    )
    ap.add_argument("--out-dir", default=None, help="output directory")
    ap.add_argument("--manifest-name", default=None, help="manifest file name")
    ap.add_argument(
        "--nifti",
        dest="write_nifti",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="convert series to NIfTI (default: --nifti)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = apply_cli_overrides(
        PipelineSettings(), args, ["dicom_root", "out_dir", "manifest_name", "write_nifti"]
    )
    setup_logging(settings.log_level, settings.log_format)

    if not settings.dicom_root:
        logger.error("missing_input", hint="pass --dicom-root or set CTPIPE_DICOM_ROOT")
        return 2
    if not os.path.isdir(settings.dicom_root):
        logger.error("not_a_directory", path=settings.dicom_root)
        return 2

    try:
        os.makedirs(settings.out_dir, exist_ok=True)
        manifest = build_manifest(
            settings.dicom_root, settings.out_dir, write_nifti=settings.write_nifti
        )
        manifest_path = os.path.join(settings.out_dir, settings.manifest_name)
        contracts.write_manifest(manifest, manifest_path)
    except Exception:
        logger.exception("ingest_failed", dicom_root=settings.dicom_root)
        return 1

    logger.info(
        "ingest_done",
        n_series=len(manifest),
        n_slices_total=int(manifest["slice_count"].sum()),
        modalities=sorted(manifest["modality"].unique().tolist()),
        manifest_path=os.path.abspath(manifest_path),
        contract=contracts.MANIFEST_CONTRACT,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
