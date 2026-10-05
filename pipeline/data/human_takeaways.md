# Human-written takeaways (ground truth for the second-reader study)

Curated from the repo's own `web/REPORT.md` ("REPORT: method justifications and
sources") and the documented comparison against the radiologist annotations.
Each takeaway has a stable ID. The VLM second-reader study
(`ai_second_reader.py`) reports agreement against these IDs: which
human-noted points the model independently reproduces, and which of its
findings are new.

## T1 — threshold over deep segmenter
"HU-threshold segmentation instead of a deep segmenter... A U-Net would need
voxel-level labels I do not have and would hide its failure modes." Each
counted voxel is one whose density falls in a published band; the method is
fully interpretable.

## T2 — numerator/denominator population mismatch
"Intersecting the infection mask with the lung mask. A fraction whose
numerator and denominator come from different populations is not a fraction."
The whole-image numerator over a lung-only denominator produced values over
100% (36 of 255 slices; 117.73% in the repo's own CSV).

## T3 — Wilson intervals for per-slice percentages
"Each slice percentage is a binomial proportion over tens of thousands of
lung pixels. The Wald interval misbehaves near 0 and 1; Wilson stays in
[0,1] with better coverage for extreme proportions."

## T4 — per-patient splitting for future classifier work
"Splitting by slice leaks highly correlated neighboring slices across train
and test and inflates every metric. Any follow-up must split by patient."

## T5 — bands sit inside the published range
Ground-glass bands cited: -703 to -368 HU (Thoracic VCAR studies of COVID CT)
and -749 to -300 HU for high-attenuation areas (Synapse 3D). The project's
-700 to -200 band sits inside the published range and was kept for
comparability.

## T6 — documented disagreement modes vs radiologist polygons
The automated-vs-radiologist comparison (2.27% automated vs 3.37% radiologist
on the annotated series) is expected to disagree where: small vessels inside
the lung are caught by the infection band; partial-volume voxels at lung
edges inflate the mask; dense consolidation above -200 HU is missed by the
band's upper edge.

## T7 — single subject, no negative controls
Only one subject's DICOM series is local; COVID-negative controls were never
part of release 1a. Sensitivity analysis over the HU bands is the mitigation.
