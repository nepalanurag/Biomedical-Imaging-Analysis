"""Shared fixtures: synthetic manifest rows and synthetic DICOM series."""

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def _manifest_row(**over):
    row = {
        "patient_id": "MIDRC-RICORD-1A-419639-000082",
        "study_uid": "1.2.3.4",
        "series_uid": "1.2.3.4.5",
        "series_number": "2",
        "series_description": "ROUTINE CHEST NON-CON",
        "modality": "CT",
        "sop_class_uid": CTImageStorage,
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


@pytest.fixture
def manifest_df():
    """Three rows: eligible axial CT, a SCOUT, and a derived reformat."""
    return pd.DataFrame(
        [
            _manifest_row(series_uid="1.2.3.4.5.1"),
            _manifest_row(
                series_uid="1.2.3.4.5.2",
                series_description="SCOUT CHEST",
                image_type="ORIGINAL,PRIMARY,LOCALIZER",
                slice_count=2,
            ),
            _manifest_row(
                series_uid="1.2.3.4.5.3",
                series_description="CHEST CORONAL REFORMAT",
                image_type="DERIVED,SECONDARY,REFORMATTED",
                image_orientation_patient="0,1,0,0,0,-1",
            ),
        ]
    )


@pytest.fixture
def bad_manifest_df():
    """Rows violating the contract: duplicate UID, bad patient ID, bad HU."""
    return pd.DataFrame(
        [
            _manifest_row(series_uid="1.2.3.4.5.1"),
            _manifest_row(
                series_uid="1.2.3.4.5.1",  # duplicate UID
                patient_id="John Smith",  # not de-identified
                hu_min=-5000.0,
            ),  # outside HU window
        ]
    )


def write_synthetic_slice(
    path, instance, series_uid, study_uid, description="ROUTINE CHEST", rows=64, seed=0
):
    """Write one minimal but valid CT slice with pydicom."""
    rng = np.random.default_rng(seed + instance)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.PatientID = "TEST-PHANTOM-001"
    ds.Modality = "CT"
    ds.SeriesDescription = description
    ds.SeriesNumber = "2"
    ds.InstanceNumber = instance
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [0.0, 0.0, float(instance)]
    ds.PixelSpacing = [0.7, 0.7]
    ds.SliceThickness = 1.0
    ds.Rows, ds.Columns = rows, rows
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    # phantom: air background (-1000 HU), soft disc (-700 HU, "lung"),
    # dense core (-100 HU, "consolidation")
    yy, xx = np.mgrid[0:rows, 0:rows]
    r = np.sqrt((xx - rows / 2) ** 2 + (yy - rows / 2) ** 2)
    hu = np.full((rows, rows), -1000.0)
    hu[r < rows * 0.35] = -700.0
    hu[r < rows * 0.12] = -100.0
    hu += rng.normal(0, 8, hu.shape)
    ds.PixelData = (hu + 1024).astype(np.int16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def dicom_tree(tmp_path):
    """A small tree: one axial CT series (5 slices) + one scout series."""
    series_uid = generate_uid()
    study_uid = generate_uid()
    ct_dir = tmp_path / "ct"
    ct_dir.mkdir()
    for i in range(1, 6):
        write_synthetic_slice(str(ct_dir / f"1-{i:03d}.dcm"), i, series_uid, study_uid)
    scout_dir = tmp_path / "scout"
    scout_dir.mkdir()
    scout_uid = generate_uid()
    for i in range(1, 3):
        write_synthetic_slice(
            str(scout_dir / f"s-{i:03d}.dcm"),
            i,
            scout_uid,
            study_uid,
            description="SCOUT CHEST",
            seed=99,
        )
    return str(tmp_path)
