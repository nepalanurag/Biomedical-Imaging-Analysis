"""COVID CT segmentation reimplementation and quantification analysis.

Reimplements the HU-threshold pipeline from the source project
(nepalanurag/Biomedical-Imaging-Analysis, streamlit_app.py) with
pydicom + numpy + scipy, because ITK is not installed here.

Original parameters (kept):
  lung mask:      HU in [-950, -300], fill holes, largest region, smooth
  infection mask: HU in [-700, -200], median filter

The one fix: the original divided whole-image infected voxels by lung
voxels, so infection_percentage could exceed 100% (it does in the
repo's own CSV: 117.73% on one series). The fixed version intersects
the infection mask with the lung mask first.
"""
import json
import os
import numpy as np
import pydicom
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.expanduser("~/workspace/diagnostic-demos/sources/Biomedical-Imaging-Analysis")
DICOM_DIR = os.path.join(SRC, "MIDRC-RICORD-1A-419639-000082",
                         "08-02-2002-NA-CT CHEST WITHOUT CONTRAST-04614",
                         "2.000000-ROUTINE CHEST NON-CON-97100")
OUT = os.path.expanduser("~/workspace/diagnostic-demos/covid-ct-web")
FIG = os.path.join(OUT, "figures")
SAMPLES = os.path.join(OUT, "site", "samples")

LUNG_LO, LUNG_HI = -950, -300
INF_LO, INF_HI = -700, -200


def load_volume(dicom_dir):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir)
             if f.endswith(".dcm")]
    ds0 = pydicom.dcmread(files[0])
    slope = float(ds0.get("RescaleSlope", 1))
    inter = float(ds0.get("RescaleIntercept", 0))
    order = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        pos = ds.get("ImagePositionPatient", None)
        key = float(pos[2]) if pos is not None else int(ds.get("InstanceNumber", 0))
        order.append((key, f))
    order.sort()
    n = len(order)
    vol = np.empty((n, ds0.Rows, ds0.Columns), dtype=np.float32)
    for i, (_, f) in enumerate(order):
        ds = pydicom.dcmread(f)
        vol[i] = ds.pixel_array.astype(np.float32) * slope + inter
    return vol


def lung_mask(vol):
    """Threshold -> fill holes -> largest connected component -> smooth."""
    m = (vol >= LUNG_LO) & (vol <= LUNG_HI)
    m = ndi.binary_fill_holes(m)
    lab, nlab = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, nlab + 1))
    m = lab == (np.argmax(sizes) + 1)
    # slice-wise binary closing with a 5x5 square, approximates the
    # original's median filter radius 5 smoothing step
    struct = np.ones((5, 5), dtype=bool)
    out = np.empty_like(m)
    for i in range(m.shape[0]):
        out[i] = ndi.binary_closing(m[i], structure=struct)
    return out


def infection_mask(vol):
    """Threshold -> median filter (slice-wise 5x5, stands in for ITK 3D radius 2)."""
    m = (vol >= INF_LO) & (vol <= INF_HI)
    out = np.empty_like(m)
    for i in range(m.shape[0]):
        out[i] = ndi.median_filter(m[i].astype(np.uint8), size=5) > 0
    return out


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    d = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - d) / den, (c + d) / den


def lung_window(vol, wl=-600, ww=1500):
    lo, hi = wl - ww / 2, wl + ww / 2
    g = np.clip((vol - lo) / (hi - lo), 0, 1)
    return (g * 255).astype(np.uint8)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(SAMPLES, exist_ok=True)
    rng_log = []

    print("loading DICOM series...")
    vol = load_volume(DICOM_DIR)
    n, H, W = vol.shape
    print(f"volume: {n} slices, {H}x{W}, HU range [{vol.min():.0f}, {vol.max():.0f}]")
    rng_log.append(dict(slices=n, hu_min=float(vol.min()), hu_max=float(vol.max())))

    print("segmenting lungs...")
    lung = lung_mask(vol)
    print("segmenting infection...")
    inf = infection_mask(vol)

    lung_vox = int(lung.sum())
    inf_whole = int(inf.sum())
    inf_in_lung = int((inf & lung).sum())
    pct_original = 100.0 * inf_whole / lung_vox
    pct_fixed = 100.0 * inf_in_lung / lung_vox

    print(f"lung voxels:            {lung_vox:,}")
    print(f"infected voxels (whole): {inf_whole:,}")
    print(f"infected voxels in lung: {inf_in_lung:,}")
    print(f"original definition: {pct_original:.2f}%  (repo CSV says 14.71%)")
    print(f"fixed definition:    {pct_fixed:.2f}%")

    # per-slice numbers, both definitions
    lung_px = lung.reshape(n, -1).sum(1).astype(int)
    inf_px_orig = inf.reshape(n, -1).sum(1).astype(int)
    inf_px = (inf & lung).reshape(n, -1).sum(1).astype(int)
    pct_orig = np.where(lung_px > 0, 100 * inf_px_orig / np.maximum(lung_px, 1), 0.0)
    pct = np.where(lung_px > 0, 100 * inf_px / np.maximum(lung_px, 1), 0.0)
    over = int((pct_orig > 100).sum())
    print(f"slices where original definition exceeds 100%: {over} of {n}")

    # sample slices: max, median, lower-quartile, minimum (all distinct, all with real lung)
    nz = np.where((pct > 0.1) & (lung_px > 10000))[0]
    order = nz[np.argsort(pct[nz])]
    picks = [order[-1], order[len(order)//2], order[len(order)//4], order[0]]
    s_max, s_med, s_low, s_zero = (int(p) for p in picks)
    sample_idx = [s_max, s_med, s_low, s_zero]
    print("sample slices:", sample_idx, "pcts:", [round(float(pct[s]), 2) for s in sample_idx])

    samples = []
    for j, s in enumerate(sample_idx):
        k = int(inf_px[s]); nn = int(lung_px[s])
        lo, hi = wilson_ci(k, nn)
        samples.append(dict(
            id=j + 1, slice=int(s),
            infection_pct=round(float(pct[s]), 2),
            ci_low=round(100 * lo, 2), ci_high=round(100 * hi, 2),
            lung_px=nn, infected_px=k,
            label=["heaviest slice", "typical slice", "mild slice", "least affected slice"][j],
        ))
        gray = lung_window(vol[s])
        plt.imsave(os.path.join(SAMPLES, f"sample{j+1}.png"), gray, cmap="gray")
        # overlay: red mask blended over gray
        rgb = np.stack([gray, gray, gray], -1).astype(float)
        m = (inf & lung)[s]
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array([255, 40, 40])
        plt.imsave(os.path.join(SAMPLES, f"sample{j+1}_overlay.png"), rgb.astype(np.uint8))

    # ---- figures ----
    plt.rcParams.update({"font.size": 11})
    # 1. per-slice profile
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(pct, lw=1.2, label="fixed: infected voxels inside lung / lung voxels")
    ax.plot(pct_orig, lw=1.0, alpha=0.6, label="original: whole-image infected / lung voxels")
    for j, s in enumerate(sample_idx):
        ax.axvline(s, ls="--", alpha=0.5)
        ax.text(s + 2, pct[s] + 0.4, f"S{j+1}", fontsize=10)
    ax.set_xlabel("slice index (feet to head)")
    ax.set_ylabel("infection % of lung")
    ax.set_title("Per-slice infection percentage, one COVID-positive chest CT")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "slice_profile.png"), dpi=150); plt.close(fig)

    # 2. sample overlays
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for j, s in enumerate(sample_idx):
        gray = lung_window(vol[s])
        axes[0, j].imshow(gray, cmap="gray"); axes[0, j].axis("off")
        axes[0, j].set_title(f"S{j+1} slice {s}", fontsize=11)
        rgb = np.stack([gray, gray, gray], -1).astype(float)
        m = (inf & lung)[s]
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array([255, 40, 40])
        axes[1, j].imshow(rgb.astype(np.uint8)); axes[1, j].axis("off")
        axes[1, j].set_title(f"{pct[s]:.1f}% infected", fontsize=11)
    axes[0, 0].set_ylabel("CT (lung window)", fontsize=11)
    axes[1, 0].set_ylabel("infection mask", fontsize=11)
    fig.suptitle("Sample slices: infection mask overlaid in red", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "samples.png"), dpi=150); plt.close(fig)

    # 3. HU histogram of lung voxels with threshold bands
    lv = vol[lung]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(lv, bins=200, range=(-1100, 200), color="#444444", alpha=0.9)
    ax.axvspan(LUNG_LO, LUNG_HI, color="steelblue", alpha=0.25, label="lung band [-950, -300]")
    ax.axvspan(INF_LO, INF_HI, color="firebrick", alpha=0.35, label="infection band [-700, -200]")
    ax.set_xlabel("Hounsfield units"); ax.set_ylabel("voxels")
    ax.set_title("HU distribution inside the lung mask")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hu_histogram.png"), dpi=150); plt.close(fig)

    # 4. error analysis: original vs fixed per slice
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(pct, pct_orig, s=8, alpha=0.5, color="#444444")
    mx = max(pct_orig.max(), pct.max()) * 1.05
    ax.plot([0, mx], [0, mx], "k--", lw=1, label="equal")
    ax.axhline(100, color="firebrick", ls=":", lw=1.2, label="100% (impossible for a true fraction)")
    ax.set_xlabel("fixed definition: infected-in-lung / lung (%)")
    ax.set_ylabel("original definition: whole-image infected / lung (%)")
    ax.set_title("Original vs fixed infection percentage, per slice")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "error_analysis.png"), dpi=150); plt.close(fig)

    results = dict(
        subject="MIDRC-RICORD-1A-419639-000082",
        series="2.000000-ROUTINE CHEST NON-CON-97100",
        series_uid="1.2.826.0.1.3680043.10.474.419639.403650391453800318566197197100",
        slices=n,
        lung_voxels=lung_vox,
        infected_voxels_whole_image=inf_whole,
        infected_voxels_in_lung=inf_in_lung,
        pct_original_definition=round(pct_original, 2),
        pct_fixed_definition=round(pct_fixed, 2),
        repo_csv_pct=14.71,
        slices_over_100_original=int(over),
        samples=samples,
        thresholds=dict(lung=[LUNG_LO, LUNG_HI], infection=[INF_LO, INF_HI]),
        note=("No trained classifier weights exist in the source project, so the demo "
              "uses these precomputed segmentation results."),
    )
    with open(os.path.join(OUT, "site", "samples", "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "analysis", "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # per-slice series for the notebook
    np.savez(os.path.join(OUT, "analysis", "per_slice.npz"),
             pct=pct, pct_orig=pct_orig, lung_px=lung_px, inf_px=inf_px)
    print("wrote results.json, figures, sample PNGs")


if __name__ == "__main__":
    main()
