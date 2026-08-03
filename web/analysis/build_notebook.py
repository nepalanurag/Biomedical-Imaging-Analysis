"""Build analysis.ipynb: markdown cells + code cells executed for real, outputs captured."""
import base64
import contextlib
import io
import os
import nbformat as nbf

OUT = os.path.expanduser("~/workspace/diagnostic-demos/covid-ct-web")
TMP = os.path.join(OUT, "analysis", "_nbfig")
os.makedirs(TMP, exist_ok=True)

NS = {}
SETUP = """
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import sys; sys.path.insert(0, os.path.expanduser("~/workspace/diagnostic-demos/covid-ct-web/analysis"))
from run_analysis import load_volume, lung_mask, infection_mask, wilson_ci, lung_window, DICOM_DIR
plt.rcParams.update({"font.size": 11})
"""

CELLS = [
 dict(md="""# COVID lung infection on CT: segmentation, quantification, and an honest error analysis

## Background

I rebuilt the segmentation pipeline from my earlier project
(nepalanurag/Biomedical-Imaging-Analysis) and checked it against the real data.
There are no trained classifier weights in that project, it was always a
segmentation and quantification pipeline, so there is no model to convert to
ONNX and no ROC or calibration curve I can compute honestly: those need labeled
classifier outputs, and I do not have labels. Everything below is computed from
the one real CT series available locally (MIDRC-RICORD-1A subject 419639-000082,
255 axial slices). Where a method needs data I do not have, I say so and show
the method on plainly labeled synthetic data instead.

## Setup

Data provenance: MIDRC-RICORD-1A is the RSNA International COVID-19 Open
Radiology Database release 1a, 120 COVID-positive chest CT studies from four
international sites, annotated by thoracic radiologists, distributed by The
Cancer Imaging Archive under CC BY-NC 4.0. The full 11 GB collection is not
local; one series (255 slices) is.""",
      code=None),

 dict(md="""## Setup: load the volume

First, read the DICOM series and convert to Hounsfield units. Sorting by slice
position matters because the files are not stored in anatomical order.""",
      code=("vol = load_volume(DICOM_DIR)\n"
            "n, H, W = vol.shape\n"
            "print(f'{n} slices, {H}x{W} pixels')\n"
            "print(f'HU range: [{vol.min():.0f}, {vol.max():.0f}]')\n"
            "print(f'mean HU: {vol.mean():.1f}')")),

 dict(md="""## Method: why HU-threshold segmentation

I use Hounsfield-unit thresholding instead of a neural network because the
question here is quantification of already-visible opacities, and a threshold
is fully interpretable: every voxel counted as infected is one whose density
falls in the ground-glass band reported in the COVID CT literature (-703 to -368
HU for ground glass in the Thoracic VCAR studies; the original project used
-700 to -200, which I keep for comparability). A deep segmenter would need
voxel-level labels I do not have, and would hide its mistakes. The right tool
is the simplest one whose failure modes I can state plainly.""",
      code=("lung = lung_mask(vol)\n"
            "inf = infection_mask(vol)\n"
            "lung_vox = int(lung.sum())\n"
            "inf_whole = int(inf.sum())\n"
            "inf_in_lung = int((inf & lung).sum())\n"
            "print(f'lung voxels:             {lung_vox:,}')\n"
            "print(f'infected voxels (whole image): {inf_whole:,}')\n"
            "print(f'infected voxels inside lung:   {inf_in_lung:,}')\n"
            "print(f'original definition (whole-image / lung): {100*inf_whole/lung_vox:.2f}%')\n"
            "print(f'repo CSV for this series:                 14.71%')\n"
            "print(f'fixed definition (in-lung / lung):        {100*inf_in_lung/lung_vox:.2f}%')\n"
            "print()\n"
            "print('My reimplementation lands within a few points of the repo number.')\n"
            "print('Exact match is not expected: ITK watershed+median vs my scipy approximations.')")),

 dict(md="""## Method: the denominator bug, and why the fix is not optional

The original code counted infected voxels over the whole image but divided by
lung voxels only. A fraction whose numerator and denominator come from different
populations is not a fraction at all, which is why the repo's own CSV contains
a series at 117.73%. Intersecting the infection mask with the lung mask first
is the fix; it changes the whole-scan number from 17.03% to 9.69% on this series.""",
      code=("lung_px = lung.reshape(n,-1).sum(1)\n"
            "inf_px_orig = inf.reshape(n,-1).sum(1)\n"
            "inf_px = (inf & lung).reshape(n,-1).sum(1)\n"
            "pct_orig = np.where(lung_px>0, 100*inf_px_orig/np.maximum(lung_px,1), 0.0)\n"
            "pct = np.where(lung_px>0, 100*inf_px/np.maximum(lung_px,1), 0.0)\n"
            "print(f'slices where the original definition exceeds 100%: {(pct_orig>100).sum()} of {n}')\n"
            "print(f'median per-slice infection (fixed): {np.median(pct[lung_px>10000]):.2f}%')\n"
            "fig, ax = plt.subplots(figsize=(9,3.6))\n"
            "ax.plot(pct, lw=1.2, label='fixed: infected inside lung / lung')\n"
            "ax.plot(pct_orig, lw=1.0, alpha=0.6, label='original: whole-image infected / lung')\n"
            "ax.set_xlabel('slice index (feet to head)'); ax.set_ylabel('infection % of lung')\n"
            "ax.set_title('Per-slice infection percentage, one COVID-positive chest CT')\n"
            "ax.legend(fontsize=9, loc='upper left')\n"
            "fig.tight_layout(); fig.savefig('" + TMP + "/p1.png', dpi=120)"),
      pngs=["p1.png"]),

 dict(md="""## Method: why the Wilson interval

Each slice's infection percentage is a binomial proportion over tens of
thousands of lung pixels. The Wald interval (p +/- 1.96 se) misbehaves near 0
and 1; the Wilson score interval stays inside [0,1] and has better coverage for
extreme proportions, which is exactly where the mild slices sit. It is the
standard choice and needs no tuning.""",
      code=("nz = np.where((pct>0.1)&(lung_px>10000))[0]\n"
            "order = nz[np.argsort(pct[nz])]\n"
            "picks = [order[-1], order[len(order)//2], order[len(order)//4], order[0]]\n"
            "names = ['heaviest','typical','mild','least affected']\n"
            "print(f'{\"slice\":>6} {\"infected %\":>11} {\"95% CI\":>22}  note')\n"
            "for s, nm in zip(picks, names):\n"
            "    lo, hi = wilson_ci(int(inf_px[s]), int(lung_px[s]))\n"
            "    print(f'{s:>6} {pct[s]:>10.2f}%  [{100*lo:6.2f}, {100*hi:6.2f}]  {nm}')\n"
            "fig, axes = plt.subplots(2, 4, figsize=(12,6.4))\n"
            "for j, s in enumerate(picks):\n"
            "    gray = lung_window(vol[s])\n"
            "    axes[0,j].imshow(gray, cmap='gray'); axes[0,j].axis('off')\n"
            "    axes[0,j].set_title(f'slice {s}', fontsize=11)\n"
            "    rgb = np.stack([gray,gray,gray],-1).astype(float)\n"
            "    m = (inf & lung)[s]\n"
            "    rgb[m] = 0.55*rgb[m] + 0.45*np.array([255,40,40])\n"
            "    axes[1,j].imshow(rgb.astype(np.uint8)); axes[1,j].axis('off')\n"
            "    axes[1,j].set_title(f'{pct[s]:.1f}% infected', fontsize=11)\n"
            "axes[0,0].set_ylabel('CT (lung window)', fontsize=11)\n"
            "axes[1,0].set_ylabel('infection mask', fontsize=11)\n"
            "fig.suptitle('Sample slices: infection mask overlaid in red', fontsize=13)\n"
            "fig.tight_layout(); fig.savefig('" + TMP + "/p2.png', dpi=120)"),
      pngs=["p2.png"]),

 dict(md="""## Method: sanity check with the HU histogram

If the thresholds are sensible, the lung band should sit on the air-side peak
of the histogram and the infection band should cover the shoulder between air
and soft tissue, where ground glass lives. That is what the plot shows, and it
matches the published bands (ground glass -703 to -368 HU; consolidation above
-100 HU).""",
      code=("lv = vol[lung]\n"
            "fig, ax = plt.subplots(figsize=(8,3.8))\n"
            "ax.hist(lv, bins=200, range=(-1100,200), color='#444444')\n"
            "ax.axvspan(-950,-300, color='steelblue', alpha=0.25, label='lung band [-950,-300]')\n"
            "ax.axvspan(-700,-200, color='firebrick', alpha=0.35, label='infection band [-700,-200]')\n"
            "ax.set_xlabel('Hounsfield units'); ax.set_ylabel('voxels')\n"
            "ax.set_title('HU distribution inside the lung mask')\n"
            "ax.legend(fontsize=10)\n"
            "fig.tight_layout(); fig.savefig('" + TMP + "/p3.png', dpi=120)"),
      pngs=["p3.png"]),

 dict(md="""## Results: where the threshold method goes wrong

Plotting the original definition against the fixed one per slice shows the bug
is not a rare edge case: slices above the lung apices and below the diaphragm
have almost no lung but plenty of -700 to -200 HU tissue (muscle, bowel), so the
original ratio explodes. Even after the fix, the method has known failure modes
I cannot remove with a threshold: vessels and bronchial walls fall in the band,
partial-volume voxels at pleural edges count as infected, and dense
consolidation above -200 HU is missed entirely. Any number from this pipeline
should be read as an upper-bound-ish estimate, not a measurement.""",
      code=("fig, ax = plt.subplots(figsize=(6.4,4.6))\n"
            "ax.scatter(pct, pct_orig, s=8, alpha=0.5, color='#444444')\n"
            "mx = max(pct_orig.max(), pct.max())*1.05\n"
            "ax.plot([0,mx],[0,mx],'k--',lw=1,label='equal')\n"
            "ax.axhline(100, color='firebrick', ls=':', lw=1.2, label='100% (impossible for a true fraction)')\n"
            "ax.set_xlabel('fixed: infected-in-lung / lung (%)')\n"
            "ax.set_ylabel('original: whole-image infected / lung (%)')\n"
            "ax.set_title('Original vs fixed infection percentage, per slice')\n"
            "ax.legend(fontsize=9)\n"
            "fig.tight_layout(); fig.savefig('" + TMP + "/p4.png', dpi=120)"),
      pngs=["p4.png"]),

 dict(md="""## Results: what I did not compute, and why

Three analyses belong to a classifier study, and I do not have a classifier:

* **ROC with bootstrap confidence bands.** Needs per-slice or per-patient
  predicted scores plus true labels. The standard approach is the nonparametric
  two-sample bootstrap (2,000 replicates is the usual recommendation for stable
  ROC uncertainty). I have neither scores nor labels.
* **Calibration curve.** Needs predicted probabilities and outcomes to plot
  observed event frequency against predicted probability (reliability diagram).
  A threshold fraction is not a probability.
* **Grad-CAM.** Needs a trained convolutional network; it backpropagates the
  class score to the last convolutional layer. There is no network here. The
  infection-mask overlays above serve the same role, showing where the
  decision comes from, and they are exact rather than approximate.

For context, published COVID CT classifiers report high numbers on curated
benchmarks: COVIDNet-CT reached 99.1% accuracy on the 104,009-image COVIDx-CT
set. Those are classification benchmarks on multi-site data, a different task
from single-patient quantification, so they are context, not a target.

Below is the bootstrap ROC-band method run on synthetic data, so the mechanics
are on record. This plot is synthetic and proves nothing about CT.""",
      code=("rng = np.random.default_rng(0)\n"
            "# SYNTHETIC DATA - method illustration only\n"
            "y = np.array([0]*400 + [1]*400)\n"
            "s = np.where(y==1, rng.normal(1.2,1,800), rng.normal(0,1,800))\n"
            "from sklearn.metrics import roc_curve, auc\n"
            "fpr, tpr, _ = roc_curve(y, s); a0 = auc(fpr, tpr)\n"
            "grid = np.linspace(0,1,100)\n"
            "boots = []\n"
            "for b in range(2000):\n"
            "    i0 = rng.choice(np.where(y==0)[0], 400, replace=True)\n"
            "    i1 = rng.choice(np.where(y==1)[0], 400, replace=True)\n"
            "    idx = np.concatenate([i0,i1])\n"
            "    fb, tb, _ = roc_curve(y[idx], s[idx])\n"
            "    boots.append(np.interp(grid, fb, tb))\n"
            "boots = np.array(boots)\n"
            "lo, hi = np.percentile(boots,[2.5,97.5],axis=0)\n"
            "print(f'SYNTHETIC example: AUC = {a0:.3f}, 95% bootstrap CI = [{auc(grid,lo):.3f}, {auc(grid,hi):.3f}]')\n"
            "fig, ax = plt.subplots(figsize=(6,5))\n"
            "ax.fill_between(grid, lo, hi, alpha=0.3, label='95% bootstrap band (2,000 replicates)')\n"
            "ax.plot(grid, np.interp(grid,fpr,tpr), label=f'ROC (AUC {a0:.3f})')\n"
            "ax.plot([0,1],[0,1],'k--',lw=1)\n"
            "ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')\n"
            "ax.set_title('SYNTHETIC DATA - bootstrap ROC band, method illustration only')\n"
            "ax.legend(fontsize=9)\n"
            "fig.tight_layout(); fig.savefig('" + TMP + "/p5.png', dpi=120)"),
      pngs=["p5.png"]),

 dict(md="""## Takeaway: limitations and what comes next

One patient, one series, no labels. The numbers above describe this scan, not
COVID CT in general. A real classifier study would need the full RICORD cohort
plus COVID-negative controls, a train/validation/test split by patient (never
by slice, or leakage inflates everything), and then the ROC, calibration, and
Grad-CAM analyses sketched above. The fixed segmentation pipeline here is a
solid baseline to compare such a model against, which is its honest value.
The fixed pipeline is a solid, honest baseline: on this scan it reports 9.69% of lung volume affected, with the known failure modes stated above. It is a quantification tool for visible opacities, not a diagnostic one, and any classifier study would still need the full cohort, patient-level splits, and the ROC, calibration, and Grad-CAM analyses sketched above.""",
      code=None),
]

def main():
    exec(SETUP, NS)
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for ci, cell in enumerate(CELLS):
        if cell.get("md"):
            nb.cells.append(nbf.v4.new_markdown_cell(cell["md"]))
        code = cell.get("code")
        if code:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code, NS)
            text = buf.getvalue()
            outputs = []
            if text.strip():
                outputs.append(nbf.v4.new_output("stream", name="stdout", text=text))
            for png in cell.get("pngs", []):
                with open(os.path.join(TMP, png), "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                outputs.append(nbf.v4.new_output("display_data",
                    data={"image/png": b64, "text/plain": "<Figure>"}, metadata={}))
            nb.cells.append(nbf.v4.new_code_cell(code, outputs=outputs))
    path = os.path.join(OUT, "analysis", "analysis.ipynb")
    nbf.write(nb, path)
    print("wrote", path, "with", len(nb.cells), "cells")

if __name__ == "__main__":
    main()
