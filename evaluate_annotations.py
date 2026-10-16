#!/usr/bin/env python3
"""Evaluate the repo's HU-threshold COVID infection segmentation against
MD.ai radiologist annotations (ground truth) and report Dice / IoU.

Method replicated from segmentation.py (COVIDLungSegmentation):
  infection mask = HU in [-700, -200] + 3D median filter (radius 2)
  lung mask      = HU in [-950, -300] + hole fill + largest connected component
  prediction     = infection mask intersected with lung mask

Ground truth: 'Infectious opacity' freeform polygons from
MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json, rasterized with
cv2.fillPoly in standard image coordinates (no flip needed -- MD.ai vertices
already match pydicom's pixel_array orientation; the flipud in the repo's
old dicom_to_nifti.py was a bug), mapped to slices via SOPInstanceUID.

Metrics are computed over annotated slices only (primary), plus a
full-volume figure for reference.
"""

import glob
import json
import os

import cv2
import numpy as np
import pydicom
from scipy import ndimage

REPO = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR = os.path.join(REPO, "MIDRC-RICORD-1A-419639-000082")
ANN_PATH = os.path.join(
    REPO, "MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json")
LOCAL_SERIES = "1.2.826.0.1.3680043.10.474.419639.403650391453800318566197197100"

# HU thresholds from segmentation.py
INF_LO, INF_HI = -700, -200
LUNG_LO, LUNG_HI = -950, -300


def load_hu_volume(dicom_dir):
    files = sorted(glob.glob(os.path.join(dicom_dir, "**", "*.dcm"),
                             recursive=True))
    assert files, "no DICOM files found"
    slices = []
    for f in files:
        d = pydicom.dcmread(f, stop_before_pixels=False)
        if d.SeriesInstanceUID != LOCAL_SERIES:
            continue
        arr = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1))
        intercept = float(getattr(d, "RescaleIntercept", 0))
        hu = arr * slope + intercept
        z = float(d.ImagePositionPatient[2])
        slices.append((z, d.SOPInstanceUID, hu))
    slices.sort(key=lambda s: s[0])
    sop_to_idx = {s[1]: i for i, s in enumerate(slices)}
    vol = np.stack([s[2] for s in slices], axis=0)  # (z, y, x)
    print(f"loaded {vol.shape[0]} slices, shape {vol.shape}")
    return vol, sop_to_idx


def infection_mask(hu):
    m = (hu >= INF_LO) & (hu <= INF_HI)
    # ITK MedianImageFilter radius 2 -> 5x5x5 neighborhood
    m = ndimage.median_filter(m.astype(np.uint8), size=5) > 0
    return m


def lung_mask(hu):
    m = (hu >= LUNG_LO) & (hu <= LUNG_HI)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    largest = 1 + int(np.argmax(sizes))
    return lab == largest


def ground_truth_mask(ann_path, sop_to_idx, shape):
    with open(ann_path) as f:
        data = json.load(f)
    anns = data["datasets"][0]["annotations"]
    # label name -> ids for the infectious lesion labels
    label_names = {}
    for lg in data["labelGroups"]:
        for lab in lg["labels"]:
            label_names[lab["id"]] = lab["name"]
    gt = np.zeros(shape, dtype=np.uint8)
    used_sops = set()
    skipped_no_slice = 0
    n_poly = 0
    for a in anns:
        if a.get("SeriesInstanceUID") != LOCAL_SERIES:
            continue
        name = label_names.get(a["labelId"], "")
        if name != "Infectious opacity":
            continue
        verts = a.get("data", {}).get("vertices")
        if not verts:
            continue
        idx = sop_to_idx.get(a.get("SOPInstanceUID"))
        if idx is None:
            skipped_no_slice += 1
            continue
        poly = np.int32([np.array(verts).reshape(-1, 2)])
        canvas = np.zeros(shape[1:], dtype=np.uint8)
        cv2.fillPoly(canvas, poly, 1)
        # NOTE: no flipud. MD.ai vertices are already in standard image
        # coordinates (origin top-left), matching pydicom's pixel_array.
        # The flipud in dicom_to_nifti.py was a bug: with it, only 34% of
        # GT falls inside the lung and Dice collapses to ~0.06; without it,
        # 87% of GT is in-lung and Dice is ~0.65.
        gt[idx] = np.maximum(gt[idx], canvas)
        used_sops.add(a["SOPInstanceUID"])
        n_poly += 1
    print(f"GT: {n_poly} infectious-opacity polygons on {len(used_sops)} slices "
          f"({skipped_no_slice} annotations had no matching local slice)")
    return gt, sorted(used_sops)


def dice(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 2 * inter / denom if denom else float("nan")


def iou(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / union if union else float("nan")


def main():
    hu, sop_to_idx = load_hu_volume(DICOM_DIR)
    print("building infection mask ...")
    inf = infection_mask(hu)
    print("building lung mask ...")
    lung = lung_mask(hu)
    pred = inf & lung
    print(f"pred infected voxels (in lung): {pred.sum()}")

    print("building ground-truth mask ...")
    gt, used_sops = ground_truth_mask(ANN_PATH, sop_to_idx, hu.shape)
    print(f"GT infected voxels: {gt.sum()}")

    ann_idx = sorted(sop_to_idx[s] for s in used_sops)
    p_ann, g_ann = pred[ann_idx], gt[ann_idx]

    d_ann = dice(p_ann, g_ann)
    i_ann = iou(p_ann, g_ann)
    d_full = dice(pred, gt)
    i_full = iou(pred, gt)

    print("\n==== RESULTS ====")
    print(f"annotated slices used : {len(ann_idx)} / {hu.shape[0]}")
    print(f"Dice (annotated slices): {d_ann:.4f}")
    print(f"IoU  (annotated slices): {i_ann:.4f}")
    print(f"Dice (full volume)     : {d_full:.4f}")
    print(f"IoU  (full volume)     : {i_full:.4f}")

    # sanity-check overlays on the 3 most-annotated slices
    counts = gt.sum(axis=(1, 2))
    top = np.argsort(counts)[::-1][:3]
    for i, z in enumerate(top):
        overlay = np.zeros((*hu.shape[1:], 3), dtype=np.uint8)
        w = np.clip((hu[z] + 1000) / 1400 * 255, 0, 255).astype(np.uint8)
        overlay[:] = w[:, :, None]
        overlay[gt[z] > 0] = (0, 255, 0)      # GT green
        overlay[(pred[z] > 0) & (gt[z] == 0)] = (255, 0, 0)  # FP red
        overlay[(pred[z] == 0) & (gt[z] > 0)] = (0, 0, 255)   # FN blue
        out = os.path.join(REPO, f"eval_overlay_slice{z}.png")
        cv2.imwrite(out, overlay)
        print(f"overlay written: {out} "
              f"(green=GT, red=FP, blue=FN)")


if __name__ == "__main__":
    main()
