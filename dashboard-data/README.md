# Dashboard data: COVID lung infection on CT

Small JSON exports of the key results from this repo's notebooks, for building
interactive dashboards. Every number comes from the notebook outputs; nothing
is invented. See `source_notebook` in each file for provenance.

## segmentation_quantification.json
From `FINAL_PRESENTATION.ipynb`.

- `method`: one-line description of the segmentation pipeline.
- `subjects`: array, one object per CT series that was segmented:
  - `subject`, `series_uid` (where applicable), `volume_voxels` [x, y, z]
  - `hu_min`, `hu_max`, `hu_mean`: Hounsfield-unit range of the volume
  - `lung_voxels`, `infected_voxels`, `infection_pct`: mask counts and the percentage
  - `automated_infected_voxels` / `automated_infection_pct`: pipeline result on the annotated series
  - `annotation_infected_voxels` / `annotation_infection_pct`: radiologist-polygon result on the same series
  - `annotation_source`: where the ground-truth labels came from
- `data_note`: what data is (and is not) in the repo.

## pipeline_debugging.json
From `FIRE.ipynb`. The engineering trail, not clinical results.

- `volume_test`: subject, volume shape, HU stats, and the IsoData threshold (`isodata_threshold_hu`) the pipeline settled on.
- `failed_approach`: the connected-components version that overflowed ITK's 255-label limit (`objects_found`: 883236).

## reanalysis_quantification.json
From `web/analysis/analysis.ipynb`: an independent reimplementation on one CT series.

- `series`: subject, slice count, pixel size, HU range/mean.
- `whole_scan`: `lung_voxels`, `infected_voxels_whole_image`, `infected_voxels_inside_lung`, and the infection percentage under the original definition (`original_definition_pct`), the repo's CSV (`repo_csv_pct`), and the fixed definition (`fixed_definition_pct`).
- `per_slice`: how many slices exceeded 100% under the original definition, and the median per-slice infection under the fix.
- `slice_samples`: four representative slices with Wilson 95% CIs (`ci95`) and a note (`heaviest`, `typical`, `mild`, `least affected`).
- `denominator_bug`: plain-language description of the bug the reanalysis fixed.
- `synthetic_method_illustration`: bootstrap ROC-band run on labeled synthetic data (AUC + CI); method demo only.
- `literature_context`: published COVIDNet-CT benchmark numbers, for context.
