# COVID lung infection on CT: segmentation and quantification

An interactive demo of COVID-19 lung infection segmentation on chest CT, plus the
analysis behind it. Pick a sample slice and see the infected regions with a
measured percentage and a confidence interval.

## The short version

My earlier project (nepalanurag/Biomedical-Imaging-Analysis) segmented COVID
lung infections from CT with Hounsfield-unit thresholds. It was never a trained
classifier: there are no model weights anywhere in it, so there is nothing to
convert to ONNX and no ROC or calibration curve I can compute honestly. This
repo takes Path B and says so plainly: the demo runs on precomputed
segmentation results from the one real CT series available locally, and the
analysis reimplements the pipeline, checks it against the project's own numbers,
and fixes a real bug I found in it.

## Data

One COVID-positive chest CT series from MIDRC-RICORD-1A (RSNA International
COVID-19 Open Radiology Database, via The Cancer Imaging Archive, CC BY-NC 4.0):
255 axial slices, 512x512, one patient. The full 11 GB collection is not local;
the md.ai ground-truth annotations used in the original notebook are not in the
repo either, so there are no labels to validate against.

## Method

HU-threshold segmentation, reimplemented with pydicom/numpy/scipy (the original
used ITK, which is not installed here; same parameters):

1. Lung mask: keep voxels in [-950, -300] HU, fill holes, keep the largest
   connected region, smooth.
2. Infection mask: keep voxels in [-700, -200] HU (the ground-glass band),
   median-filter to drop specks.
3. Score: infected voxels inside the lung mask divided by lung voxels.

I chose thresholds over a neural network because the task is quantification of
visible opacities, a threshold is fully interpretable, and I have no voxel
labels to train on. The bands match the published COVID CT literature (ground
glass around -703 to -368 HU).

## Results (all computed, none invented)

Whole scan, fixed definition: **9.69%** of lung volume affected.

| | this reimplementation | repo's CSV |
|---|---|---|
| lung voxels | 7,043,692 | 6,627,697 |
| infected voxels | 1,199,890 | 975,156 |
| infection % (original definition) | 17.03% | 14.71% |

Close but not exact, as expected: my scipy approximations of ITK's watershed and
median filter differ in the details. Same magnitude, same conclusion.

The bug: the original code counted infected voxels over the whole image and
divided by lung voxels. On 36 of 255 slices that ratio exceeds 100%, and the
repo's own CSV has a series at 117.73%. Intersecting the infection mask with the
lung mask fixes it: 17.03% becomes 9.69% on this series.

Sample slices in the demo (Wilson 95% intervals):

| sample | slice | infected | 95% CI |
|---|---|---|---|
| S1 (heaviest) | 225 | 29.19% | [28.39, 30.01] |
| S2 (typical) | 198 | 9.13% | [8.82, 9.44] |
| S3 (mild) | 59 | 5.41% | [5.12, 5.71] |
| S4 (least affected) | 83 | 2.56% | [2.41, 2.71] |

## What is not here, and why

ROC with bootstrap bands, calibration curves, and Grad-CAM all need a trained
classifier with labels. I have neither, so I did not fake them. The notebook
shows the bootstrap ROC-band mechanics on plainly labeled synthetic data and
documents what a real follow-up study would need. The infection-mask overlays
serve the same role as Grad-CAM here: they show exactly where the decision
comes from.

## The web demo

`site/` is a static page (no build step). It shows four sample slices with
precomputed infection percentages, confidence intervals, and a CT/overlay
toggle. You can also upload your own lung-window slice; the page gives a rough
on-screen estimate and labels it as approximate, since there is no lung mask in
the browser. Deployed to Vercel; see the repo description for the URL.

## Repo layout

- `analysis/run_analysis.py` - the pipeline: DICOM loading, segmentation, figures, results
- `analysis/analysis.ipynb` - executed notebook with the full analysis and outputs
- `analysis/results.json` - computed numbers
- `figures/` - publication figures (per-slice profile, sample overlays, HU histogram, error analysis)
- `site/` - the static web demo (index.html, app.js, styles.css, samples/)
- `REPORT.md` - method justifications and sources

## Limits

One patient, one series, no labels. These numbers describe this scan, not COVID
CT in general. A threshold also mistakes vessels and pleural edges for
infection and misses dense consolidation above -200 HU. Read the percentages as
estimates, not measurements.
