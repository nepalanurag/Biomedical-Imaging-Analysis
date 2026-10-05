"""Quantification tests: the >100%-infection bug class must be impossible."""

import numpy as np
import pytest

from pipeline import quantify

rng = np.random.default_rng(7)


def _masks(lung_n=10000, inf_in=2000, inf_out=5000):
    """Lung mask with infection partly inside and partly outside it."""
    lung = np.zeros(200 * 200, dtype=bool)
    lung[:lung_n] = True
    inf = np.zeros(200 * 200, dtype=bool)
    inf[:inf_in] = True  # inside lung
    inf[lung_n : lung_n + inf_out] = True  # outside lung (the old bug)
    return lung, inf


def test_intersection_bounds_percentage():
    # Even with 5000 infected voxels OUTSIDE the lung, the percentage
    # counts only the 2000 inside: 20%, never 70%.
    lung, inf = _masks()
    q = quantify.quantify_infection(lung, inf)
    assert q["infected_voxels"] == 2000
    assert q["total_voxels"] == 10000
    assert q["infection_percentage"] == pytest.approx(20.0)
    assert 0.0 <= q["infection_percentage"] <= 100.0


def test_whole_lung_infected_is_100_not_more():
    lung = np.ones(1000, dtype=bool)
    inf = np.ones(1000, dtype=bool)
    q = quantify.quantify_infection(lung, inf)
    assert q["infection_percentage"] == pytest.approx(100.0)


def test_empty_lung_is_zero_not_nan():
    q = quantify.quantify_infection(np.zeros(10, dtype=bool), np.ones(10, dtype=bool))
    assert q["infection_percentage"] == 0.0
    assert q["total_voxels"] == 0


def test_matches_segmentation_py_formula():
    """Same inputs as segmentation.py's fixed quantify_infection give the
    same outputs (intersection version)."""
    lung = rng.random((30, 40)) > 0.4
    inf = rng.random((30, 40)) > 0.8
    q = quantify.quantify_infection(lung, inf)
    expected_inf = int(np.sum(inf & lung))
    expected_total = int(np.sum(lung))
    assert q["infected_voxels"] == expected_inf
    assert q["total_voxels"] == expected_total
    assert q["infection_percentage"] == pytest.approx(100.0 * expected_inf / expected_total)


def test_wilson_ci_bounds():
    lo, hi = quantify.wilson_ci(0, 10000)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = quantify.wilson_ci(10000, 10000)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = quantify.wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 1.0)


def test_segmentation_runs_on_phantom(dicom_tree, tmp_path):
    """End-to-end on synthetic DICOMs: segment + quantify stays in [0,100]."""
    from pipeline import ingest

    manifest = ingest.build_manifest(dicom_tree, str(tmp_path), write_nifti=True)
    # two series: axial CT (eligible) + scout
    assert len(manifest) == 2
    row = manifest[manifest["series_description"] == "ROUTINE CHEST"].iloc[0]
    vol = quantify.load_volume(row["nifti_path"])
    assert vol.shape == (5, 64, 64)
    # phantom is in HU already (air -1000)
    assert vol.min() < -900
    lung = quantify.segment_lungs(vol)
    inf = quantify.segment_infection(vol)
    assert lung.sum() > 0
    q = quantify.quantify_infection(lung, inf)
    assert 0.0 <= q["infection_percentage"] <= 100.0
