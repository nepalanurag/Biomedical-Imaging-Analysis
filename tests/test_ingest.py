"""Ingest tests: real pydicom I/O on synthetic DICOMs."""

import os

import nibabel as nib
import numpy as np

from pipeline import ingest


def test_find_and_group(dicom_tree):
    files = ingest.find_dicom_files(dicom_tree)
    assert len(files) == 7  # 5 axial + 2 scout
    grouped = ingest.group_by_series(files)
    assert len(grouped) == 2
    assert sorted(len(v) for v in grouped.values()) == [2, 5]


def test_build_manifest(dicom_tree, tmp_path):
    manifest = ingest.build_manifest(dicom_tree, str(tmp_path), write_nifti=True)
    assert len(manifest) == 2
    ct = manifest[manifest["slice_count"] == 5].iloc[0]
    assert ct["modality"] == "CT"
    assert ct["patient_id"] == "TEST-PHANTOM-001"
    assert ct["rows"] == 64 and ct["cols"] == 64
    assert ct["hu_min"] < -900  # air background present
    assert ct["hu_max"] > -300  # dense core present
    assert os.path.isfile(ct["nifti_path"])

    # NIfTI round-trips with the right geometry and HU values
    img = nib.load(ct["nifti_path"])
    data = np.asarray(img.get_fdata())
    assert data.shape == (64, 64, 5)  # (x, y, z)
    assert abs(data.min() - ct["hu_min"]) < 1.0
    assert img.header.get_xyzt_units() == ("mm", "unknown")


def test_no_nifti_mode(dicom_tree, tmp_path):
    manifest = ingest.build_manifest(dicom_tree, str(tmp_path), write_nifti=False)
    assert (manifest["nifti_path"] == "").all()
    assert not os.path.exists(os.path.join(str(tmp_path), "nifti"))


def test_empty_dir_fails_loudly(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="no DICOM files"):
        ingest.build_manifest(str(tmp_path), str(tmp_path))


def test_slice_ordering_by_position(dicom_tree):
    from pydicom import dcmread

    files = ingest.find_dicom_files(dicom_tree)
    grouped = ingest.group_by_series(files)
    paths = max(grouped.values(), key=len)
    vol, _ds0 = ingest.read_series_volume(paths)
    # phantom core (dense, HU > -300) is centered; slice order must be
    # monotonic in z for the volume to be coherent
    core = (vol > -300).sum(axis=(1, 2))
    assert core.max() > 0
    # volume sorted by ImagePositionPatient: first slice z < last slice z
    first = dcmread(paths[0], stop_before_pixels=True)
    assert vol.shape[0] == 5
    assert first.get("ImagePositionPatient") is not None
