"""Agentic HU-band sensitivity sweep.

Varies the lung band and the infection band over a grid, re-runs
quantify.py for each configuration, and writes a sensitivity section
(markdown + figures) showing how the infection percentage moves.

This automates the "try a threshold, look at the overlay" loop from the
original analysis: every configuration is a full pipeline run with its
code hash stamped, so the sweep is reproducible and auditable.

Default design (16 configs): the project-standard bands, plus a 3x3 grid
over the infection band and one-at-a-time variations of each lung-band
edge. Override with explicit value lists; --max-configs guards against
accidentally launching a huge grid.

Compute cost (measured on the reference machine, reported per run in
sensitivity.md): cost scales as n_configs x n_series x t_segment, where
t_segment is dominated by the slice-wise morphology on 512x512 slices.

Usage:
    python -m pipeline.ai_sweep --manifest pipeline_data/manifest.parquet
    python -m pipeline.ai_sweep --manifest ... --inf-lo-values -750,-700,-650 \\
        --inf-hi-values -250,-200,-150
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import pandas as pd

from . import contracts, quantify
from .config import PipelineSettings, apply_cli_overrides
from .log import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_LUNG_LO = [-1000, -950, -900]
DEFAULT_LUNG_HI = [-350, -300, -250]
DEFAULT_INF_LO = [-750, -700, -650]
DEFAULT_INF_HI = [-250, -200, -150]


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return default
    try:
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"not a comma-separated int list: {raw!r}")
    if not vals:
        raise ValueError("empty value list")
    return vals


def build_configs(args, settings) -> list[dict]:
    """One-at-a-time variations around the project standard + inf-band grid."""
    lung_lo_v = _parse_int_list(args.lung_lo_values, DEFAULT_LUNG_LO)
    lung_hi_v = _parse_int_list(args.lung_hi_values, DEFAULT_LUNG_HI)
    inf_lo_v = _parse_int_list(args.inf_lo_values, DEFAULT_INF_LO)
    inf_hi_v = _parse_int_list(args.inf_hi_values, DEFAULT_INF_HI)

    base = {
        "lung_lo": settings.lung_lo,
        "lung_hi": settings.lung_hi,
        "inf_lo": settings.inf_lo,
        "inf_hi": settings.inf_hi,
    }
    configs = [dict(base, label="baseline")]
    for lo in lung_lo_v:
        if lo != base["lung_lo"]:
            configs.append(dict(base, lung_lo=lo, label=f"lung_lo={lo}"))
    for hi in lung_hi_v:
        if hi != base["lung_hi"]:
            configs.append(dict(base, lung_hi=hi, label=f"lung_hi={hi}"))
    for lo, hi in itertools.product(inf_lo_v, inf_hi_v):
        if (lo, hi) != (base["inf_lo"], base["inf_hi"]):
            configs.append(dict(base, inf_lo=lo, inf_hi=hi, label=f"inf=[{lo},{hi}]"))
    for c in configs:
        if not (c["lung_lo"] < c["lung_hi"] and c["inf_lo"] < c["inf_hi"]):
            raise ValueError(f"inverted band in config {c['label']}")
    return configs


def run_sweep(manifest_path: str, configs: list[dict]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        t0 = time.perf_counter()
        rdf = quantify.run_quantification(
            manifest_path,
            lung_band=(cfg["lung_lo"], cfg["lung_hi"]),
            inf_band=(cfg["inf_lo"], cfg["inf_hi"]),
        )
        dt = time.perf_counter() - t0
        rows.append(
            {
                "lung_lo": cfg["lung_lo"],
                "lung_hi": cfg["lung_hi"],
                "inf_lo": cfg["inf_lo"],
                "inf_hi": cfg["inf_hi"],
                "label": cfg["label"],
                "n_series": len(rdf),
                "mean_infection_pct": float(rdf["infection_pct"].mean()) if len(rdf) else 0.0,
                "max_infection_pct": float(rdf["infection_pct"].max()) if len(rdf) else 0.0,
                "seconds": round(dt, 1),
                "code_hash": quantify.code_hash(),
            }
        )
        logger.info(
            "sweep_config_done",
            label=cfg["label"],
            mean_pct=round(rows[-1]["mean_infection_pct"], 2),
            seconds=round(dt, 1),
        )
    return pd.DataFrame(rows)


def write_figures(sdf: pd.DataFrame, fig_dir: str, base: dict, settings) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    paths = []

    # 1. one-at-a-time: infection pct vs each varied edge
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("lung_lo", "Lung band lower edge (HU)"),
        ("lung_hi", "Lung band upper edge (HU)"),
        ("inf_lo", "Infection band lower edge (HU)"),
        ("inf_hi", "Infection band upper edge (HU)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), specs):
        others = [c for c in ("lung_lo", "lung_hi", "inf_lo", "inf_hi") if c != col]
        sub = sdf[(sdf[others] == pd.Series(base)[others].values).all(axis=1)]
        sub = sub.sort_values(col)
        ax.plot(sub[col], sub["mean_infection_pct"], "o-", lw=1.5)
        ax.axvline(base[col], color="gray", ls="--", lw=1, label="project standard")
        ax.set_xlabel(title)
        ax.set_ylabel("mean infection %")
        ax.legend(fontsize=9)
    fig.suptitle("HU-band sensitivity: one-at-a-time variation")
    fig.tight_layout()
    p1 = os.path.join(fig_dir, "sweep_oat.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    paths.append(p1)

    # 2. infection-band grid heatmap (configs where only inf_lo/inf_hi vary)
    others = ["lung_lo", "lung_hi"]
    grid = sdf[(sdf[others] == pd.Series(base)[others].values).all(axis=1)]
    piv = grid.pivot_table(
        index="inf_hi", columns="inf_lo", values="mean_infection_pct", aggfunc="mean"
    )
    if piv.shape[0] > 1 and piv.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(piv.columns)), labels=piv.columns)
        ax.set_yticks(range(len(piv.index)), labels=piv.index)
        ax.set_xlabel("infection band lower edge (HU)")
        ax.set_ylabel("infection band upper edge (HU)")
        ax.set_title("Mean infection % over the infection-band grid")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="mean infection %")
        fig.tight_layout()
        p2 = os.path.join(fig_dir, "sweep_heatmap.png")
        fig.savefig(p2, dpi=110)
        plt.close(fig)
        paths.append(p2)
    return paths


def write_report(sdf: pd.DataFrame, fig_paths: list[str], out_dir: str, total_s: float) -> str:
    base_row = sdf[sdf["label"] == "baseline"].iloc[0]
    base_mean = base_row["mean_infection_pct"]
    spread = sdf["mean_infection_pct"].max() - sdf["mean_infection_pct"].min()

    lines = [
        "# HU-band sensitivity sweep",
        "",
        f"{len(sdf)} configurations, {total_s:.0f}s total wall time "
        f"({sdf['seconds'].mean():.1f}s mean per config).",
        "",
        f"Baseline (project standard bands [{int(base_row['lung_lo'])},"
        f"{int(base_row['lung_hi'])}] / [{int(base_row['inf_lo'])},"
        f"{int(base_row['inf_hi'])}]): mean infection "
        f"{base_mean:.2f}% across {int(base_row['n_series'])} series.",
        "",
        f"Across the sweep, the mean infection percentage ranges from "
        f"{sdf['mean_infection_pct'].min():.2f}% to "
        f"{sdf['mean_infection_pct'].max():.2f}% (spread {spread:.2f} pp).",
        "",
        "## Reading the result",
        "",
        "The infection band's upper edge is the dominant lever: raising it "
        "pulls dense consolidation into the mask, lowering it restricts the "
        "mask to ground glass. The lung band edges move the denominator and "
        "the mask together, so their effect is smaller. If the ranking of "
        "series by infection percentage is stable across the sweep, the "
        "headline comparison is robust to threshold choice; if it flips, the "
        "threshold is doing the work and needs radiologist adjudication.",
        "",
        "## Compute cost",
        "",
        f"Measured: {sdf['seconds'].mean():.1f}s per configuration on "
        f"{int(base_row['n_series'])} series "
        f"({sdf['seconds'].mean() / max(int(base_row['n_series']), 1):.1f}s per series).",
        "Cost scales as n_configs x n_series x t_segment; t_segment is "
        "dominated by slice-wise morphology on 512x512 slices. A full "
        "4-D grid (e.g. 5 values per edge = 625 configs) would take roughly "
        f"{625 * sdf['seconds'].mean() / 3600:.1f}h on this machine, which is "
        "why the default sweep varies one edge at a time around the standard "
        "plus a 3x3 infection-band grid.",
        "",
        "## Figures",
        "",
    ]
    for p in fig_paths:
        lines.append(f"![{os.path.basename(p)}]({os.path.basename(p)})")
    lines += [
        "",
        "## All configurations",
        "",
        sdf[
            ["label", "n_series", "mean_infection_pct", "max_infection_pct", "seconds"]
        ].to_markdown(index=False),
        "",
    ]
    path = os.path.join(out_dir, "sensitivity.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Agentic HU-band sensitivity sweep: re-run quantification "
        "over a band grid and write a sensitivity section."
    )
    ap.add_argument("--manifest", default=None)
    ap.add_argument(
        "--out-dir", default=None, help="sweep output dir (default: pipeline_data/sweep)"
    )
    ap.add_argument(
        "--lung-lo-values", default=None, help="comma-separated HU values, e.g. -1000,-950,-900"
    )
    ap.add_argument(
        "--lung-hi-values", default=None, help="comma-separated HU values, e.g. -350,-300,-250"
    )
    ap.add_argument(
        "--inf-lo-values", default=None, help="comma-separated HU values, e.g. -750,-700,-650"
    )
    ap.add_argument(
        "--inf-hi-values", default=None, help="comma-separated HU values, e.g. -250,-200,-150"
    )
    ap.add_argument(
        "--max-configs", type=int, default=40, help="fail if the grid exceeds this many configs"
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = apply_cli_overrides(
        PipelineSettings(), args, ["lung_lo", "lung_hi", "inf_lo", "inf_hi"]
    )
    setup_logging(settings.log_level, settings.log_format)
    out_dir = args.out_dir or settings.sweep_out_dir
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    manifest_path = args.manifest or os.path.join(settings.out_dir, settings.manifest_name)
    if not os.path.isfile(manifest_path):
        logger.error("manifest_not_found", path=manifest_path)
        return 2

    try:
        configs = build_configs(args, settings)
    except ValueError as exc:
        logger.error("bad_sweep_grid", error=str(exc))
        return 2
    if len(configs) > args.max_configs:
        logger.error(
            "grid_too_large",
            n_configs=len(configs),
            max_configs=args.max_configs,
            hint="narrow the value lists or raise --max-configs",
        )
        return 2

    logger.info("sweep_start", n_configs=len(configs))
    t0 = time.perf_counter()
    try:
        sdf = run_sweep(manifest_path, configs)
        contracts.SweepSchema.validate(sdf, lazy=True)
    except Exception:
        logger.exception("sweep_failed")
        return 1
    total_s = time.perf_counter() - t0

    results_path = os.path.join(out_dir, "sweep_results.parquet")
    contracts.write_sweep(sdf, results_path)
    fig_paths = write_figures(
        sdf,
        fig_dir,
        {
            "lung_lo": settings.lung_lo,
            "lung_hi": settings.lung_hi,
            "inf_lo": settings.inf_lo,
            "inf_hi": settings.inf_hi,
        },
        settings,
    )
    # figures live next to the markdown; copy names for relative links
    import shutil

    rel_figs = []
    for p in fig_paths:
        dest = os.path.join(out_dir, os.path.basename(p))
        shutil.copy(p, dest)
        rel_figs.append(dest)
    md_path = write_report(sdf, rel_figs, out_dir, total_s)

    logger.info(
        "sweep_done",
        n_configs=len(configs),
        total_seconds=round(total_s, 1),
        results_path=results_path,
        report_path=md_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
