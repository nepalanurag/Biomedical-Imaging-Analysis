"""Eligibility rule tests: the SCOUT-in-CSV bug class must be impossible."""

from pipeline import eligibility


def _row(**over):
    base = {
        "series_uid": "1.2.3",
        "series_description": "ROUTINE CHEST NON-CON",
        "modality": "CT",
        "image_type": "ORIGINAL,PRIMARY,AXIAL",
        "image_orientation_patient": "1,0,0,0,1,0",
    }
    base.update(over)
    return base


def test_axial_diagnostic_ct_is_eligible():
    ok, reason = eligibility.check_series(_row())
    assert ok, reason


def test_scout_series_excluded():
    ok, reason = eligibility.check_series(_row(series_description="SCOUT CHEST"))
    assert not ok
    assert "scout" in reason


def test_localizer_excluded():
    ok, reason = eligibility.check_series(_row(series_description="Chest Localizer AP"))
    assert not ok


def test_topogram_excluded():
    ok, reason = eligibility.check_series(_row(series_description="TOPOGRAM 120kV"))
    assert not ok


def test_derived_reformat_excluded():
    ok, reason = eligibility.check_series(
        _row(series_description="CHEST CORONAL", image_type="DERIVED,SECONDARY,REFORMATTED")
    )
    assert not ok
    assert "derived" in reason


def test_non_axial_geometry_excluded():
    ok, reason = eligibility.check_series(_row(image_orientation_patient="0,1,0,0,0,-1"))
    assert not ok
    assert "axial" in reason


def test_non_ct_modality_excluded():
    ok, _ = eligibility.check_series(_row(modality="MR"))
    assert not ok


def test_word_boundaries_no_false_positives():
    # "scout" inside a longer word must not trigger the localizer rule.
    ok, reason = eligibility.check_series(_row(series_description="CHEST SCOUTING PROTOCOL"))
    assert ok, reason


def test_apply_marks_scout_row_in_manifest(manifest_df):
    out = eligibility.apply(manifest_df)
    by_uid = {r["series_uid"]: r for _, r in out.iterrows()}
    assert by_uid["1.2.3.4.5.1"]["eligibility_ok"]
    assert not by_uid["1.2.3.4.5.2"]["eligibility_ok"]
    assert not by_uid["1.2.3.4.5.3"]["eligibility_ok"]
    assert "scout" in by_uid["1.2.3.4.5.2"]["eligibility_reason"]
    # every row gets a reason, eligible or not
    assert all(out["eligibility_reason"].astype(bool))


def test_whitelist_override_is_explicit():
    uid = "1.9.9.9"
    eligibility.ELIGIBILITY_OVERRIDES[uid] = "test: hand-verified axial"
    try:
        ok, reason = eligibility.check_series(
            _row(series_uid=uid, series_description="SCOUT CHEST")
        )
        assert ok
        assert "whitelisted" in reason
    finally:
        del eligibility.ELIGIBILITY_OVERRIDES[uid]


def test_unparseable_orientation_rejected():
    ok, reason = eligibility.check_series(_row(image_orientation_patient="not-a-vector"))
    assert not ok
