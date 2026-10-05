"""VLM second-reader study.

For a sample of slices with segmentation overlays, an independent reader
catalogs disagreement modes between the HU-threshold segmentation and what
a radiologist would plausibly mark:

* vessel_misclassification — small vessels caught by the infection band
* partial_volume_edges — infection voxels hugging the lung boundary
* missed_consolidation_above_band — dense consolidation above the band's upper edge
* pleural_edge_noise — infection components touching the image border

Two reader modes:

* ``--mode dryrun`` (default): no API key needed. Rule-based heuristics act
  as a stand-in reader. They are documented proxies, not a model; the point
  of dry-run mode is to exercise the full reporting pipeline.
* ``--mode vlm``: sends the overlay PNGs to a vision-language model
  (``--vlm-provider gemini|openai``) with a structured prompt and parses the
  returned JSON. Needs ``GOOGLE_API_KEY`` (or ``CTPIPE_GOOGLE_API_KEY``) for
  gemini, ``OPENAI_API_KEY`` (or ``CTPIPE_OPENAI_API_KEY``) for openai.

Both modes write ``disagreement_report.json`` + ``disagreement_report.md``
and a takeaway-agreement section comparing the reader's findings against the
human-written takeaways in ``pipeline/data/human_takeaways.md``
(curated from the repo's own web/REPORT.md).

Usage:
    python -m pipeline.ai_second_reader --manifest pipeline_data/manifest.parquet
    python -m pipeline.ai_second_reader --manifest ... --mode vlm --vlm-provider gemini
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage as ndi

from . import contracts, eligibility, quantify
from .config import PipelineSettings, apply_cli_overrides
from .log import get_logger, setup_logging

logger = get_logger(__name__)

# Controlled vocabulary for disagreement modes. The VLM prompt requires the
# model to use these IDs so findings stay comparable across runs.
MODES = (
    "vessel_misclassification",
    "partial_volume_edges",
    "missed_consolidation_above_band",
    "pleural_edge_noise",
)

# takeaway_id -> modes that would reproduce it (from human_takeaways.md)
TAKEAWAY_MODE_MAP = {
    "T6": ["vessel_misclassification", "partial_volume_edges", "missed_consolidation_above_band"],
}

TAKEAWAYS_PATH = os.path.join(os.path.dirname(__file__), "data", "human_takeaways.md")


def load_takeaways() -> dict[str, str]:
    """Parse human_takeaways.md into {id: text}."""
    takeaways = {}
    if not os.path.isfile(TAKEAWAYS_PATH):
        return takeaways
    with open(TAKEAWAYS_PATH) as fh:
        text = fh.read()
    for m in re.finditer(r"^## (T\d+) — (.+?)(?=^## |\Z)", text, re.M | re.S):
        takeaways[m.group(1)] = m.group(2).strip().split("\n")[0]
    return takeaways


def lung_window(vol: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    g = np.clip((vol - lo) / (hi - lo), 0, 1)
    return (g * 255).astype(np.uint8)


def render_overlay(gray: np.ndarray, infection: np.ndarray, lung: np.ndarray, path: str) -> None:
    """Gray slice + red infection overlay + green lung boundary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = np.stack([gray, gray, gray], axis=-1).astype(float)
    rgb[infection] = 0.55 * rgb[infection] + 0.45 * np.array([255, 40, 40])
    boundary = lung ^ ndi.binary_erosion(lung)
    rgb[boundary] = np.array([60, 220, 90])
    plt.imsave(path, rgb.astype(np.uint8))


def sample_slices(vol, lung, inf, n: int) -> list[int]:
    """Pick up to n distinct slices spanning the infection range (quantiles)."""
    lung_px = lung.reshape(vol.shape[0], -1).sum(1)
    inf_px = (inf & lung).reshape(vol.shape[0], -1).sum(1)
    pct = np.where(lung_px > 0, 100 * inf_px / np.maximum(lung_px, 1), 0.0)
    nz = np.where((pct > 0.05) & (lung_px > 10000))[0]
    if len(nz) == 0:
        raise ValueError("no slices with lung tissue and infection found")
    order = nz[np.argsort(pct[nz])]
    fracs = np.linspace(0, 1, min(n, len(order)))
    picks = {int(order[min(int(f * (len(order) - 1)), len(order) - 1)]) for f in fracs}
    return sorted(picks)


def _component_stats(mask: np.ndarray):
    lab, nlab = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, nlab + 1))
    return lab, nlab, sizes


def heuristic_findings(vol, lung, inf, slice_idx: int, inf_hi: int) -> list[dict]:
    """Rule-based stand-in reader. Documented proxies, not a model.

    Returns one finding dict per mode with measured quantities.
    """
    l2, i2 = lung[slice_idx], inf[slice_idx]
    lung_px = int(l2.sum())
    inf_px = int((i2 & l2).sum())
    findings = []

    # partial-volume edges: infection voxels within 2 px of the lung boundary
    boundary = l2 ^ ndi.binary_erosion(l2)
    near_edge = int(((i2 & l2) & ndi.binary_dilation(boundary, iterations=2)).sum())
    findings.append(
        {
            "mode": "partial_volume_edges",
            "present": near_edge > 0.05 * max(inf_px, 1),
            "infection_voxels_near_boundary": near_edge,
            "fraction_of_infection": round(near_edge / max(inf_px, 1), 3),
            "note": "proxy: infection voxels within 2px of the lung boundary",
        }
    )

    # vessel-like: small, round infection components
    lab, nlab, sizes = _component_stats(i2 & l2)
    vessel_like = 0
    for c in range(1, nlab + 1):
        area = float(sizes[c - 1])
        if area >= 80:
            continue
        comp = lab == c
        perim = float(np.sum(comp ^ ndi.binary_erosion(comp)))
        circ = 4 * np.pi * area / max(perim * perim, 1)
        if circ > 0.6:
            vessel_like += 1
    findings.append(
        {
            "mode": "vessel_misclassification",
            "present": vessel_like > 0,
            "small_round_components": vessel_like,
            "note": "proxy: components <80px with circularity >0.6; vessels are "
            "tubular in 3D and this 2D proxy undercounts them",
        }
    )

    # missed consolidation: dense regions inside the lung above the band
    dense = l2 & (vol[slice_idx] > inf_hi) & (vol[slice_idx] < 200)
    _dlab, dnlab, dsizes = _component_stats(dense)
    big_dense = int(np.sum(dsizes > 300))
    dense_px = int(dense.sum())
    findings.append(
        {
            "mode": "missed_consolidation_above_band",
            "present": big_dense > 0,
            "dense_components_over_300px": big_dense,
            "dense_voxels": dense_px,
            "fraction_of_lung": round(dense_px / max(lung_px, 1), 4),
            "note": f"proxy: lung voxels with HU in ({inf_hi}, 200) forming "
            "components >300px; the band's upper edge cannot see these",
        }
    )

    # pleural edge noise: infection components touching the image border
    border = np.zeros_like(i2, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    edge_touch = int(np.sum((i2 & l2) & border))
    findings.append(
        {
            "mode": "pleural_edge_noise",
            "present": edge_touch > 0,
            "infection_voxels_on_border": edge_touch,
            "note": "proxy: infection voxels on the image border",
        }
    )
    return findings


VLM_SYSTEM_PROMPT = """You are a second reader for a lung CT segmentation study.
You are shown axial CT slices in lung window with two overlays: RED = voxels the \
automated HU-threshold method labeled as COVID infection; GREEN = the automated lung boundary.

Catalog disagreement modes between the automated segmentation and what a radiologist \
would plausibly mark, using ONLY these mode IDs:
- vessel_misclassification: small pulmonary vessels caught by the infection HU band
- partial_volume_edges: infection-labeled voxels hugging the lung boundary (partial volume)
- missed_consolidation_above_band: dense consolidation the HU band's upper edge misses
- pleural_edge_noise: infection-labeled voxels touching the image border

Reply with a single JSON object, no other text:
{"findings": [{"mode": "<id>", "present": true|false, "severity": "low|moderate|high",
"evidence": "<one sentence describing what you see>"}]}
Be conservative: mark present=true only if you can point at the evidence."""


def _gemini_call(api_key: str, model: str, png_paths: list[str]) -> dict:
    import requests

    parts = [{"text": VLM_SYSTEM_PROMPT}]
    for p in png_paths:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}" f":generateContent"
    resp = requests.post(
        url, params={"key": api_key}, json={"contents": [{"parts": parts}]}, timeout=120
    )
    resp.raise_for_status()
    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json(text)


def _openai_call(api_key: str, model: str, png_paths: list[str]) -> dict:
    import requests

    content = [{"type": "text", "text": VLM_SYSTEM_PROMPT}]
    for p in png_paths:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1500,
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"VLM did not return JSON: {text[:200]}")
    return json.loads(m.group(0))


def vlm_findings(png_paths: list[str], provider: str, model: str, api_key: str) -> list[dict]:
    """Call the VLM and normalize its findings to the controlled vocabulary."""
    if provider == "gemini":
        raw = _gemini_call(api_key, model, png_paths)
    elif provider == "openai":
        raw = _openai_call(api_key, model, png_paths)
    else:
        raise ValueError(f"unknown vlm provider {provider!r}")
    out = []
    for f in raw.get("findings", []):
        mode = f.get("mode", "")
        if mode not in MODES:
            logger.warning("vlm_unknown_mode", mode=mode)
            continue
        out.append(
            {
                "mode": mode,
                "present": bool(f.get("present")),
                "severity": f.get("severity", "unknown"),
                "evidence": f.get("evidence", ""),
                "reader": f"vlm:{provider}/{model}",
            }
        )
    return out


def agreement_report(findings: list[dict], takeaways: dict[str, str], reader: str) -> list[dict]:
    """Compare reader findings against the human-written takeaways."""
    present = {f["mode"] for f in findings if f.get("present")}
    report = []
    for tid, text in sorted(takeaways.items()):
        modes = TAKEAWAY_MODE_MAP.get(tid)
        if not modes:
            status, detail = (
                "not_assessable_from_overlays",
                "method-level point; overlays cannot confirm or refute it",
            )
        elif all(m in present for m in modes):
            status, detail = "reproduced", f"{reader} independently flagged: {', '.join(modes)}"
        elif any(m in present for m in modes):
            status, detail = (
                "partially_reproduced",
                f"flagged {sorted(set(modes) & present)}, missed {sorted(set(modes) - present)}",
            )
        else:
            status, detail = "not_reproduced", f"none of {modes} flagged by {reader}"
        report.append({"takeaway_id": tid, "takeaway": text, "status": status, "detail": detail})
    novel = sorted(present - {m for ms in TAKEAWAY_MODE_MAP.values() for m in ms})
    if novel:
        report.append(
            {
                "takeaway_id": "NOVEL",
                "takeaway": "modes the reader " "flagged that no human takeaway covers",
                "status": "new_findings",
                "detail": ", ".join(novel),
            }
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="VLM second-reader study: catalog segmentation disagreement "
        "modes on overlay slices and compare against human takeaways."
    )
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--series-uid", default=None, help="series to read (default: first eligible)")
    ap.add_argument("--n-slices", type=int, default=4)
    ap.add_argument(
        "--mode",
        default=None,
        choices=["dryrun", "vlm"],
        help="dryrun = rule-based stand-in (no API key); " "vlm = call a vision-language model",
    )
    ap.add_argument("--vlm-provider", default=None, choices=["gemini", "openai"])
    ap.add_argument("--out-dir", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = apply_cli_overrides(
        PipelineSettings(),
        args,
        ["out_dir", "vlm_provider", "lung_lo", "lung_hi", "inf_lo", "inf_hi"],
    )
    setup_logging(settings.log_level, settings.log_format)
    mode = args.mode or "dryrun"
    out_dir = args.out_dir or settings.second_reader_out_dir
    os.makedirs(out_dir, exist_ok=True)
    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    manifest_path = args.manifest or os.path.join(settings.out_dir, settings.manifest_name)
    if not os.path.isfile(manifest_path):
        logger.error("manifest_not_found", path=manifest_path)
        return 2
    df = contracts.read_manifest(manifest_path)
    df = eligibility.apply(df)
    eligible = df[df["eligibility_ok"]]
    if len(eligible) == 0:
        logger.error("no_eligible_series")
        return 1
    if args.series_uid:
        sel = eligible[eligible["series_uid"] == args.series_uid]
        if len(sel) == 0:
            logger.error("series_not_found_or_ineligible", series_uid=args.series_uid)
            return 2
        row = sel.iloc[0]
    else:
        row = eligible.iloc[0]

    logger.info("second_reader_start", mode=mode, series_uid=str(row["series_uid"])[:32])
    vol = quantify.load_volume(row["nifti_path"] if row.get("nifti_path") else row["source_dir"])
    lung = quantify.segment_lungs(vol, settings.lung_lo, settings.lung_hi)
    inf = quantify.segment_infection(vol, settings.inf_lo, settings.inf_hi)
    slices = sample_slices(vol, lung, inf, args.n_slices or 4)

    png_paths, slice_reports = [], []
    for s in slices:
        p = os.path.join(overlay_dir, f"slice_{s:03d}.png")
        render_overlay(lung_window(vol[s]), (inf & lung)[s], lung[s], p)
        png_paths.append(p)
        k = int(((inf & lung)[s]).sum())
        nn = int(lung[s].sum())
        lo, hi = quantify.wilson_ci(k, nn)
        slice_reports.append(
            {
                "slice": int(s),
                "overlay": os.path.relpath(p, out_dir),
                "infection_pct": round(100 * k / max(nn, 1), 2),
                "ci_low": round(100 * lo, 2),
                "ci_high": round(100 * hi, 2),
            }
        )

    if mode == "vlm":
        provider = args.vlm_provider or settings.vlm_provider
        api_key = (
            settings.google_api_key or os.environ.get("GOOGLE_API_KEY", "")
            if provider == "gemini"
            else settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            logger.error(
                "vlm_key_missing",
                provider=provider,
                hint="set GOOGLE_API_KEY/OPENAI_API_KEY or "
                "CTPIPE_GOOGLE_API_KEY/CTPIPE_OPENAI_API_KEY",
            )
            return 2
        model = settings.vlm_model if provider == "gemini" else "gpt-4o-mini"
        try:
            findings = vlm_findings(png_paths, provider, model, api_key)
        except Exception:
            logger.exception("vlm_call_failed", provider=provider)
            return 1
        reader = f"vlm:{provider}/{model}"
    else:
        findings = []
        for s in slices:
            for f in heuristic_findings(vol, lung, inf, s, settings.inf_hi):
                f["slice"] = int(s)
                f["reader"] = "heuristic-dryrun"
                findings.append(f)
        # collapse per-slice heuristic hits to per-mode verdicts
        collapsed = []
        for m in MODES:
            hits = [f for f in findings if f["mode"] == m and f["present"]]
            rep = next((f for f in findings if f["mode"] == m), {})
            collapsed.append(
                {
                    "mode": m,
                    "present": bool(hits),
                    "slices_flagged": sorted({f["slice"] for f in hits}),
                    "detail": rep.get("note", ""),
                    "reader": "heuristic-dryrun",
                }
            )
        findings = collapsed
        reader = "heuristic-dryrun"

    takeaways = load_takeaways()
    agreement = agreement_report(findings, takeaways, reader)

    report = {
        "reader": reader,
        "series_uid": row["series_uid"],
        "slices": slice_reports,
        "findings": findings,
        "takeaway_agreement": agreement,
        "caveat": (
            "dryrun heuristics are documented proxies, not a model; "
            "re-run with --mode vlm for the actual study."
            if mode == "dryrun"
            else "VLM judgments are uncalibrated; treat as hypotheses."
        ),
    }
    json_path = os.path.join(out_dir, "disagreement_report.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)

    md = [
        f"# Second-reader disagreement report ({reader})",
        "",
        f"Series `{str(row['series_uid'])[:32]}...`, {len(slices)} slices sampled "
        f"across the infection range.",
        "",
        "## Findings",
        "",
    ]
    for f in findings:
        mark = "PRESENT" if f["present"] else "absent"
        md.append(f"- **{f['mode']}**: {mark} — " f"{f.get('evidence') or f.get('detail', '')}")
    md += ["", "## Agreement with human takeaways", ""]
    for a in agreement:
        md.append(f"- **{a['takeaway_id']}** ({a['status']}): {a['detail']}")
        if a["takeaway_id"] != "NOVEL":
            md.append(f"  > {a['takeaway']}")
    md += ["", f"*{report['caveat']}*"]
    md_path = os.path.join(out_dir, "disagreement_report.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md) + "\n")

    n_present = sum(1 for f in findings if f["present"])
    logger.info(
        "second_reader_done",
        reader=reader,
        modes_present=n_present,
        modes_checked=len(findings),
        json_path=json_path,
        md_path=md_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
