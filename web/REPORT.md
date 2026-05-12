# REPORT: method justifications and sources

## Path taken

Path B (precomputed predictions). I verified myself that the source project
contains no trained weights of any kind (no .h5, .onnx, .pt, .pth files), and
it was never a classifier: it is a segmentation and quantification pipeline.
Only one subject's DICOM series is local; the full MIDRC-RICORD-1A cohort is
11 GB and COVID-negative controls were never part of release 1a, so training a
classifier is not defensible. The md.ai annotation JSON referenced in the
original notebook is not in the repo, so no Dice or ROC against ground truth
can be recomputed.

## Method choices

**HU-threshold segmentation instead of a deep segmenter.** The clinical
question here is quantification of visible opacities, and a threshold is fully
interpretable: each counted voxel is one whose density falls in a published
band. A U-Net would need voxel-level labels I do not have and would hide its
failure modes. The ground-glass bands I cite: -703 to -368 HU (Thoracic VCAR
studies of COVID CT) and -749 to -300 HU for high-attenuation areas
(Synapse 3D COVID analysis). The original project's -700 to -200 band sits
inside the published range, so I kept it for comparability.

**Intersecting the infection mask with the lung mask.** A fraction whose
numerator and denominator come from different populations is not a fraction.
The original code's whole-image numerator over lung-only denominator produced
values over 100% (36 of 255 slices here; 117.73% in the repo's own CSV), so the
fix is a correctness fix, not a tuning choice.

**Wilson score interval for per-slice percentages.** Each slice percentage is a
binomial proportion over tens of thousands of lung pixels. The Wald interval
misbehaves near 0 and 1; Wilson stays in [0,1] with better coverage for extreme
proportions, which is where the mild slices sit. Standard practice, no tuning.

**Bootstrap ROC bands (documented, not computed).** The nonparametric
two-sample bootstrap with 2,000 replicates is the standard recommendation for
stable ROC uncertainty (NIST studies of bootstrap variability in ROC analysis).
I show the mechanics on synthetic data in the notebook because the method needs
labeled classifier scores I do not have.

**Calibration curves (documented, not computed).** A reliability diagram plots
observed event frequency against mean predicted probability per bin, with the
diagonal as perfect calibration. Needs predicted probabilities and outcomes;
a threshold fraction is not a probability. Method reference: scikit-learn's
probability calibration documentation.

**Grad-CAM (documented, not computed).** Gradient-weighted Class Activation
Mapping backpropagates the class score to the last convolutional layer to
localize what drove the decision. Needs a trained CNN; there is none here.
The infection-mask overlays serve the same explanatory role and are exact.

**Per-patient splitting for any future classifier work.** Splitting by slice
leaks highly correlated neighboring slices across train and test and inflates
every metric. Any follow-up must split by patient.

## Sources

- MIDRC-RICORD-1A collection, The Cancer Imaging Archive:
  https://www.cancerimagingarchive.net/collection/midrc-ricord-1a/
- RICORD dataset description (120 COVID-positive chest CT studies, four sites,
  expert annotations): https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742
- COVIDx-CT benchmark and COVIDNet-CT results (104,009 images, 1,489 patients,
  99.1% accuracy): https://pmc.ncbi.nlm.nih.gov/articles/PMC10263244/
- Bootstrap variability in ROC analysis, 2,000 replicates recommendation (NIST):
  https://www.nist.gov/publications/further-studies-bootstrap-variability-roc-analysis-large-datasets?pub_id=906977
- Calibration curves / reliability diagrams:
  https://scikit-learn.org/1.7/modules/calibration.html
- Grad-CAM paper (Selvaraju et al.):
  https://arxiv.org/pdf/1610.02391v3
- HU bands for ground glass and consolidation in COVID CT:
  https://www.mdpi.com/2075-4426/11/7/641/htm
- Histogram-based threshold quantification of COVID pneumonia:
  https://link.springer.com/article/10.1186/s43055-021-00602-1
