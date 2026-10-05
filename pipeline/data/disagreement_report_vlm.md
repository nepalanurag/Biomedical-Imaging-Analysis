# Second-reader disagreement report (vlm:gemini/gemini-2.5-flash)

Series `1.2.826.0.1.3680043.10.474.41963...`, 4 slices sampled across the infection range.
Same slices as the dry-run study, so the two reads are directly comparable.

## Findings

- **vessel_misclassification**: PRESENT (severity moderate) — Numerous small red voxels consistently highlight what appear to be pulmonary vessels in all four images, particularly prominent in Images 1 and 3.
- **partial_volume_edges**: absent (severity low) — No clear evidence of misclassified partial volume artifacts hugging the lung boundaries was observed across any of the images; peripheral red areas appear to represent true pathology.
- **missed_consolidation_above_band**: PRESENT (severity high) — In Image 4, the automated segmentation misses the denser, central portions of the large consolidation in the right lung, stopping at a specific HU threshold.
- **pleural_edge_noise**: absent (severity low) — No red infection-labeled voxels were observed touching the image border in any of the provided images.

## Agreement with human takeaways

- **T1** (not_assessable_from_overlays): method-level point; overlays cannot confirm or refute it
  > threshold over deep segmenter
- **T2** (not_assessable_from_overlays): method-level point; overlays cannot confirm or refute it
  > numerator/denominator population mismatch
- **T3** (not_assessable_from_overlays): method-level point; overlays cannot confirm or refute it
  > Wilson intervals for per-slice percentages
- **T4** (not_assessable_from_overlays): method-level point; overlays cannot confirm or refute it
  > per-patient splitting for future classifier work
- **T5** (not_assessable_from_overlays): method-level point; overlays cannot confirm or refute it
  > bands sit inside the published range
- **T6** (partially_reproduced): flagged ['missed_consolidation_above_band', 'vessel_misclassification'], missed ['partial_volume_edges']
  > documented disagreement modes vs radiologist polygons
- **T7** (not_assessable_from_overlays): method-level point; overlays cannot confirm or refute it
  > single subject, no negative controls

*VLM judgments are uncalibrated; treat as hypotheses.*
