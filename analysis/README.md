# Analysis code

## `series_eligibility.py`

The series-eligibility rule for the quantification table: only axial diagnostic
CT series are eligible. Scout/localizer, smart-prep, and coronal/sagittal/MIP
reformat series are excluded as code. Uninformative descriptions ("NO DETAILS")
are kept and flagged, not silently dropped.

## `curate_quantification.py`

Applies the eligibility rule to `infection_quantification_by_subject.csv` and
writes `infection_quantification_by_subject_curated.csv` + `curation_log.csv`.
The original CSV is untouched. Run:

```bash
python analysis/curate_quantification.py
```

Latest run: 229 input rows → 154 kept, 75 dropped (non-diagnostic series).
Four kept rows exceed 100% — they were computed under the pre-consolidation
whole-image numerator and are flagged in the `curation_flag` column for
recompute on the next refresh with the consolidated intersection formula
(see `segmentation.py`, `COVIDLungSegmentation.quantify_infection`).
