"""Versioned data contracts for the pipeline.

Two artifacts cross stage boundaries, and both carry an explicit contract
version stamped into the Parquet file metadata:

* **manifest** (contract ``manifest/1.0``): one row per DICOM series, written
  by ``ingest.py``, read by ``validate.py`` / ``quantify.py`` / ``refresh.py``.
* **results** (contract ``results/1.0``): one row per quantified series,
  written by ``quantify.py`` / ``refresh.py``.

Readers check the stamped version and fail loudly on mismatch instead of
silently misreading a stale file. Bump the version constant and document the
change here whenever a column is added, removed, or redefined.

Manifest columns (manifest/1.0):
    patient_id, study_uid, series_uid, series_number, series_description,
    modality, sop_class_uid, image_type, image_orientation_patient,
    slice_count, rows, cols, slice_thickness, pixel_spacing,
    hu_min, hu_max, nifti_path, source_dir,
    [added by validate.py] eligibility_ok, eligibility_reason

Results columns (results/1.0):
    series_uid, patient_id, lung_band, inf_band, slices,
    lung_voxels, infected_voxels, infection_pct, ci_low, ci_high, source,
    code_hash, run_ts, version, git_sha
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pandera as pa
import pyarrow as pat
import pyarrow.parquet as pq
from pandera.typing import Series

MANIFEST_CONTRACT = "manifest/1.0"
RESULTS_CONTRACT = "results/1.0"
SWEEP_CONTRACT = "sweep/1.0"

# De-identified PatientID pattern: letters/digits/dots/dashes only, no spaces,
# nothing that looks like a real name. (MIDRC-RICORD-1A IDs look like
# "MIDRC-RICORD-1A-419639-000082".)
DEID_PATTERN = r"^[A-Za-z0-9._-]+$"

# Plausible CT HU window. The floor is -4096 (not -1024) because voxels
# outside the reconstruction circle are commonly padded with values like
# -3024 (21% of the reference RICORD series); the check still catches
# garbage far outside any real encoding.
HU_MIN, HU_MAX = -4096.0, 3071.0

# Slice counts outside this range are implausible for a chest CT series.
MIN_SLICES, MAX_SLICES = 1, 5000


class ContractError(Exception):
    """Raised when an artifact's contract version is missing or unsupported."""


class ManifestSchema(pa.DataFrameModel):
    """Pandera schema for the manifest (contract manifest/1.0)."""

    patient_id: Series[str] = pa.Field(
        str_matches=DEID_PATTERN,
        nullable=False,
        description="pseudonymized patient ID; must match the de-ID pattern",
    )
    study_uid: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    series_uid: Series[str] = pa.Field(
        nullable=False, unique=True, description="one row per series UID"
    )
    series_number: Series[str] = pa.Field(nullable=True)
    series_description: Series[str] = pa.Field(nullable=True)
    modality: Series[str] = pa.Field(nullable=False)
    sop_class_uid: Series[str] = pa.Field(nullable=True)
    image_type: Series[str] = pa.Field(nullable=True)
    image_orientation_patient: Series[str] = pa.Field(nullable=True)
    slice_count: Series[int] = pa.Field(ge=MIN_SLICES, le=MAX_SLICES)
    rows: Series[int] = pa.Field(ge=16, le=4096)
    cols: Series[int] = pa.Field(ge=16, le=4096)
    slice_thickness: Series[float] = pa.Field(nullable=True, ge=0.1, le=20.0)
    pixel_spacing: Series[str] = pa.Field(nullable=True)
    hu_min: Series[float] = pa.Field(ge=HU_MIN, le=HU_MAX)
    hu_max: Series[float] = pa.Field(ge=HU_MIN, le=HU_MAX)
    nifti_path: Series[str] = pa.Field(nullable=True)
    source_dir: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_check
    def hu_range_ordered(cls, df: pd.DataFrame) -> pd.Series:
        return df["hu_min"] <= df["hu_max"]


class ResultsSchema(pa.DataFrameModel):
    """Pandera schema for quantification results (contract results/1.0)."""

    series_uid: Series[str] = pa.Field(nullable=False)
    patient_id: Series[str] = pa.Field(str_matches=DEID_PATTERN, nullable=False)
    lung_band: Series[str] = pa.Field(nullable=False)
    inf_band: Series[str] = pa.Field(nullable=False)
    slices: Series[int] = pa.Field(ge=MIN_SLICES, le=MAX_SLICES)
    lung_voxels: Series[int] = pa.Field(ge=0)
    infected_voxels: Series[int] = pa.Field(ge=0)
    infection_pct: Series[float] = pa.Field(
        ge=0.0,
        le=100.0,
        description="bounded by construction: numerator is intersected with the lung mask",
    )
    ci_low: Series[float] = pa.Field(ge=0.0, le=100.0)
    ci_high: Series[float] = pa.Field(ge=0.0, le=100.0)
    source: Series[str] = pa.Field(nullable=False)
    code_hash: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    run_ts: Series[str] = pa.Field(nullable=True)
    version: Series[str] = pa.Field(nullable=True)
    git_sha: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_check
    def infected_within_lung(cls, df: pd.DataFrame) -> pd.Series:
        return df["infected_voxels"] <= df["lung_voxels"]


def _write_parquet(df: pd.DataFrame, path: str, contract: str) -> None:
    table = pat.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"contract"] = contract.encode()
    meta[b"written_ts"] = datetime.now(timezone.utc).isoformat().encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def _read_parquet(path: str, contract: str) -> pd.DataFrame:
    table = pq.read_table(path)
    stamped = (table.schema.metadata or {}).get(b"contract", b"").decode()
    if stamped != contract:
        raise ContractError(
            f"{path}: contract mismatch (file stamped {stamped!r}, "
            f"expected {contract!r}). Re-run the producing stage."
        )
    return table.to_pandas()


def write_manifest(df: pd.DataFrame, path: str) -> None:
    _write_parquet(df, path, MANIFEST_CONTRACT)


def read_manifest(path: str) -> pd.DataFrame:
    return _read_parquet(path, MANIFEST_CONTRACT)


def write_results(df: pd.DataFrame, path: str) -> None:
    _write_parquet(df, path, RESULTS_CONTRACT)


def read_results(path: str) -> pd.DataFrame:
    return _read_parquet(path, RESULTS_CONTRACT)


class SweepSchema(pa.DataFrameModel):
    """Pandera schema for HU-band sweep results (contract sweep/1.0)."""

    lung_lo: Series[int] = pa.Field()
    lung_hi: Series[int] = pa.Field()
    inf_lo: Series[int] = pa.Field()
    inf_hi: Series[int] = pa.Field()
    n_series: Series[int] = pa.Field(ge=1)
    mean_infection_pct: Series[float] = pa.Field(ge=0.0, le=100.0)
    max_infection_pct: Series[float] = pa.Field(ge=0.0, le=100.0)
    seconds: Series[float] = pa.Field(ge=0.0)
    code_hash: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})


def write_sweep(df: pd.DataFrame, path: str) -> None:
    _write_parquet(df, path, SWEEP_CONTRACT)


def read_sweep(path: str) -> pd.DataFrame:
    return _read_parquet(path, SWEEP_CONTRACT)
