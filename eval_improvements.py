#!/usr/bin/env python3
"""Evaluate improved segmentation variants against the MD.ai GT.

Reuses the verified GT loading from evaluate_annotations.py.
Reports Dice/IoU on annotated slices with deltas vs the 0.6671 baseline.
"""

import time

import numpy as np

import evaluate_annotations as B
import improve_segment as I


def score(pred, gt, ann_idx, name, baseline_dice):
    d = B.dice(pred[ann_idx], gt[ann_idx])
    i = B.iou(pred[ann_idx], gt[ann_idx])
    print(f"{name:28s} Dice {d:.4f} (delta {d - baseline_dice:+.4f})  "
          f"IoU {i:.4f}")
    return d, i


def main():
    t0 = time.time()
    hu, sop_to_idx = B.load_hu_volume(B.DICOM_DIR)
    gt, used_sops = B.ground_truth_mask(B.ANN_PATH, sop_to_idx, hu.shape)
    ann_idx = sorted(sop_to_idx[s] for s in used_sops)

    # --- diagnostic: where does GT live in HU space?
    gt_hu = hu[gt > 0]
    print(f"\nGT HU percentiles: 5th={np.percentile(gt_hu, 5):.0f} "
          f"25th={np.percentile(gt_hu, 25):.0f} "
          f"50th={np.percentile(gt_hu, 50):.0f} "
          f"75th={np.percentile(gt_hu, 75):.0f} "
          f"95th={np.percentile(gt_hu, 95):.0f}")
    print(f"GT voxels above -200 HU (missed by fixed cap): "
          f"{(gt_hu > -200).mean() * 100:.1f}%")
    print(f"GT voxels below -700 HU (missed by fixed floor): "
          f"{(gt_hu < -700).mean() * 100:.1f}%")

    # --- baseline (must reproduce 0.6671)
    inf = B.infection_mask(hu)
    lung = B.lung_mask(hu)
    base_d, _ = score(inf & lung, gt, ann_idx, "baseline (fixed bands)",
                      0.0)
    assert abs(base_d - 0.6671) < 0.002, "baseline drifted!"

    # --- lever 1: lung mask repair
    for r in (5, 9, 14):
        t = time.time()
        lung_c = I.lung_mask_closed(hu, radius=r)
        score(I.infection_mask_fixed(hu) & lung_c, gt, ann_idx,
              f"lever1: lung closing r={r}", base_d)
        print(f"  ({time.time() - t:.0f}s)")

    # --- lever 2: adaptive infection band
    t = time.time()
    lung_c9 = I.lung_mask_closed(hu, radius=9)
    score(I.infection_mask_adaptive(hu, lung) & lung, gt, ann_idx,
          "lever2: otsu adaptive (base lung)", base_d)
    print(f"  ({time.time() - t:.0f}s)")
    t = time.time()
    score(I.infection_mask_adaptive(hu, lung_c9) & lung_c9, gt, ann_idx,
          "lever1+2: closing r=9 + otsu", base_d)
    print(f"  ({time.time() - t:.0f}s)")
    t = time.time()
    score(I.infection_mask_adaptive(hu, lung_c9, hi_cap=100) & lung_c9,
          gt, ann_idx, "lever1+2: closing r=9 + otsu, cap +100",
          base_d)
    print(f"  ({time.time() - t:.0f}s)")

    # --- lever 3: fallback grid tune of fixed bands (with repaired lung)
    # diagnostic showed 26.4% of GT below -700: try lower floors too
    for lo, hi in ((-750, -150), (-650, -250), (-700, -100), (-750, -200),
                   (-800, -200), (-820, -150), (-800, -100)):
        t = time.time()
        score(I.infection_mask_fixed(hu, lo=lo, hi=hi) & lung_c9, gt,
              ann_idx, f"lever3: fixed [{lo},{hi}] + closing r=9",
              base_d)
        print(f"  ({time.time() - t:.0f}s)")

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
