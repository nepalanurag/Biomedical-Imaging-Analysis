#!/usr/bin/env python3
"""Verify the INTEGRATED pipeline (pipeline/quantify.py) reproduces the
validated Dice/IoU numbers against the MD.ai radiologist GT.

Expected: Dice ~0.7115 / IoU ~0.5522 on annotated slices
(lung closing disk r=9, infection band fixed [-700,-200]).

Fails loudly if the numbers drift -- do not push if this fails.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate_annotations as B
from pipeline.quantify import segment_infection, segment_lungs

EXPECTED_DICE = 0.7115
EXPECTED_IOU = 0.5522
TOL = 0.01


def main():
    hu, sop_to_idx = B.load_hu_volume(B.DICOM_DIR)
    gt, used_sops = B.ground_truth_mask(B.ANN_PATH, sop_to_idx, hu.shape)
    ann_idx = sorted(sop_to_idx[s] for s in used_sops)

    print("segmenting with integrated pipeline/quantify.py ...")
    lung = segment_lungs(hu, closing_radius=9)  # integrated default
    inf = segment_infection(hu)
    pred = inf & lung

    d = B.dice(pred[ann_idx], gt[ann_idx])
    i = B.iou(pred[ann_idx], gt[ann_idx])
    print(f"\nINTEGRATED PIPELINE: Dice {d:.4f} (expected ~{EXPECTED_DICE})")
    print(f"INTEGRATED PIPELINE: IoU  {i:.4f} (expected ~{EXPECTED_IOU})")

    assert abs(d - EXPECTED_DICE) < TOL, f"Dice drifted: {d:.4f}"
    assert abs(i - EXPECTED_IOU) < TOL, f"IoU drifted: {i:.4f}"
    print("\nOK: integrated pipeline reproduces the validated metrics.")


if __name__ == "__main__":
    main()
