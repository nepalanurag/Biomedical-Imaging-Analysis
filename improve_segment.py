#!/usr/bin/env python3
"""Improved segmentation variants for the COVID CT pipeline.

NEW CODE ONLY -- the repo's original pipeline files (segmentation.py,
pipeline/quantify.py, etc.) are left untouched. These functions are
evaluated by eval_improvements.py against the MD.ai radiologist GT.

Verified facts baked in:
- MD.ai polygon vertices are already in image coordinates (no flipud).
- Fixed bands in the original: infection [-700, -200], lung [-950, -300].
"""

import numpy as np
from scipy import ndimage

INF_LO, INF_HI = -700, -200
LUNG_LO, LUNG_HI = -950, -300


# ---------------------------------------------------------------- base pieces

def infection_mask_fixed(hu, lo=INF_LO, hi=INF_HI):
    m = (hu >= lo) & (hu <= hi)
    m = ndimage.median_filter(m.astype(np.uint8), size=5) > 0
    return m


def lung_mask_base(hu, lo=LUNG_LO, hi=LUNG_HI):
    m = (hu >= lo) & (hu <= hi)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


# ------------------------------------------------- lever 1: lung mask repair

def lung_mask_closed(hu, radius=7):
    """Base lung mask + per-slice 2D morphological closing to repair
    juxtapleural boundary gaps: dense infection near the pleura is
    thresholded out of the [-950,-300] band, leaving concavities in the
    mask. Closing bridges them. Safe for the infection prediction because
    chest wall (~-50..+100 HU) still fails the infection band."""
    base = lung_mask_base(hu)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = (x * x + y * y) <= radius * radius
    closed = np.stack(
        [ndimage.binary_closing(base[z], structure=disk)
         for z in range(base.shape[0])])
    return ndimage.binary_fill_holes(closed)


# -------------------------------------- lever 2: adaptive infection band

def _otsu_threshold_and_separability(vals, nbins=256):
    """Otsu threshold of 1-D values + separability ratio
    (between-class variance / total variance). Low separability means the
    distribution is effectively unimodal (no clear opacity class)."""
    if vals.size < 100:
        return float("-inf"), 0.0
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-6:
        return lo, 0.0
    hist, edges = np.histogram(vals, bins=nbins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist / hist.sum()
    cumw = np.cumsum(w)
    cummean = np.cumsum(w * centers)
    total_mean = cummean[-1]
    total_var = np.sum(w * (centers - total_mean) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        bc = (total_mean * cumw - cummean) ** 2 / (cumw * (1 - cumw))
    bc[~np.isfinite(bc)] = 0
    k = int(np.argmax(bc))
    sep = float(bc[k] / total_var) if total_var > 0 else 0.0
    return float(centers[k]), sep


def infection_mask_adaptive(hu, lung, hi_cap=INF_HI, sep_floor=0.30):
    """Per-slice adaptive band. Within each slice's lung mask, Otsu splits
    aerated parenchyma from denser tissue; infection = [otsu_thr, hi_cap].
    Slices with low separability (no clear second mode, e.g. healthy
    slices) fall back to the fixed lower bound so Otsu cannot invent
    infection out of a unimodal parenchyma distribution."""
    pred = np.zeros(hu.shape, dtype=bool)
    n_adaptive, n_fallback = 0, 0
    for z in range(hu.shape[0]):
        vals = hu[z][lung[z]]
        thr, sep = _otsu_threshold_and_separability(vals)
        if sep >= sep_floor and thr > -800:
            lo = thr
            n_adaptive += 1
        else:
            lo = INF_LO
            n_fallback += 1
        pred[z] = (hu[z] >= lo) & (hu[z] <= hi_cap) & lung[z]
    pred = ndimage.median_filter(pred.astype(np.uint8), size=5) > 0
    print(f"  adaptive slices: {n_adaptive}, fallback slices: {n_fallback}")
    return pred
