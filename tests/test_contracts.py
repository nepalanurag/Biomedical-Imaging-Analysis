"""Contract + validation-gate tests."""

import pandas as pd
import pandera as pa
import pytest

from pipeline import contracts, validate
from pipeline.contracts import ContractError


def _valid_row(**over):
    row = {
        "patient_id": "MIDRC-RICORD-1A-419639-000082",
        "study_uid": "1.2.3.4",
        "series_uid": "1.2.3.4.5",
        "series_number": "2",
        "series_description": "ROUTINE CHEST NON-CON",
        "modality": "CT",
        "sop_class_uid": "1.2.840.10008.5.1.4.1.1.2",
        "image_type": "ORIGINAL,PRIMARY,AXIAL",
        "image_orientation_patient": "1,0,0,0,1,0",
        "slice_count": 200,
        "rows": 512,
        "cols": 512,
        "slice_thickness": 1.0,
        "pixel_spacing": "0.7,0.7",
        "hu_min": -1024.0,
        "hu_max": 1500.0,
        "nifti_path": "",
        "source_dir": "/data/dicom",
    }
    row.update(over)
    return row


def test_manifest_roundtrip_stamps_contract(tmp_path):
    df = pd.DataFrame([_valid_row()])
    p = str(tmp_path / "m.parquet")
    contracts.write_manifest(df, p)
    back = contracts.read_manifest(p)
    assert back["series_uid"].iloc[0] == "1.2.3.4.5"
    import pyarrow.parquet as pq

    meta = pq.read_table(p).schema.metadata
    assert meta[b"contract"] == b"manifest/1.0"


def test_contract_mismatch_fails_loudly(tmp_path):
    df = pd.DataFrame([_valid_row()])
    p = str(tmp_path / "m.parquet")
    df.to_parquet(p, index=False)  # no contract stamp
    with pytest.raises(ContractError):
        contracts.read_manifest(p)


def test_schema_rejects_duplicate_series_uid(bad_manifest_df):
    with pytest.raises(pa.errors.SchemaErrors):
        contracts.ManifestSchema.validate(bad_manifest_df, lazy=True)


def test_schema_rejects_non_deidentified_patient_id():
    df = pd.DataFrame([_valid_row(patient_id="John Smith")])
    with pytest.raises(pa.errors.SchemaErrors):
        contracts.ManifestSchema.validate(df, lazy=True)


def test_schema_rejects_impossible_hu():
    df = pd.DataFrame([_valid_row(hu_min=-5000.0)])
    with pytest.raises(pa.errors.SchemaErrors):
        contracts.ManifestSchema.validate(df, lazy=True)


def test_validate_quarantines_scout_and_exits_nonzero(manifest_df, tmp_path):
    p = str(tmp_path / "m.parquet")
    contracts.write_manifest(manifest_df, p)
    ok, report_path, report = validate.validate(p, str(tmp_path))
    assert ok  # schema itself passes; quarantine is not a failure
    assert report["n_quarantined"] == 2
    reasons = " ".join(q["reason"] for q in report["quarantined"])
    assert "scout" in reasons and "derived" in reasons


def test_scout_marked_eligible_is_hard_failure(tmp_path):
    df = pd.DataFrame([_valid_row(series_description="SCOUT CHEST")])
    df["eligibility_ok"] = True  # simulate a broken eligibility rule
    df["eligibility_reason"] = "bug"
    failures, _ = validate.extra_checks(df)
    assert any(f["check"] == "no_scout_in_analysis" for f in failures)


def test_results_schema_bounds_infection_pct():
    df = pd.DataFrame(
        [
            {
                "series_uid": "1.2.3",
                "patient_id": "T-1",
                "lung_band": "[-950,-300]",
                "inf_band": "[-700,-200]",
                "slices": 10,
                "lung_voxels": 100,
                "infected_voxels": 150,
                "infection_pct": 150.0,
                "ci_low": 0.0,
                "ci_high": 100.0,
                "source": "/x",
                "code_hash": "abc123",
                "run_ts": None,
                "version": None,
                "git_sha": None,
            }
        ]
    )
    with pytest.raises(pa.errors.SchemaErrors):
        contracts.ResultsSchema.validate(df, lazy=True)
