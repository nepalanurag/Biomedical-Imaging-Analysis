# DICOM ingestion pipeline

Production ingestion pipeline for the CT lung-infection analysis in this
repo. It walks a DICOM tree, converts series to NIfTI, and writes a
versioned Parquet manifest that every downstream step reads. Validation
gates enforce a schema contract, a de-identification pattern, and a
series-eligibility rule; quantification uses the fixed
intersection-based infection metric as its single source of truth.

This is new capability, not a re-analysis: the pipeline is designed so the
two failure modes this project has historically hit cannot recur —
non-diagnostic series (SCOUT/localizer, reformatted/derived) leaking into
the analysis table, and an infection numerator counted over the whole
image instead of inside the lung mask.

```
                        ┌──────────────┐
                        │  DICOM tree  │
                        └──────┬───────┘
                               │ ingest.py (pydicom → nibabel)
                               ▼
                  ┌────────────────────────┐
                  │ manifest.parquet       │  contract manifest/1.0
                  │ one row per series UID │  (stamped in file metadata)
                  └────────┬───────────────┘
                           │ validate.py (pandera + eligibility)
              ┌────────────┴────────────┐
              ▼                         ▼
   eligible series            quarantined series
   (analysis table)           (validation_report.json)
              │
              ▼
   quantify.py ──► quantification.parquet   contract results/1.0
              │     (infection % bounded [0,100] by construction)
              ▼
   refresh.py ──► results/quantification_<ts>.parquet + .csv
                  (versioned, timestamped, code-hash stamped)

   AI components (separate, read-only over pipeline outputs):
   ai_second_reader.py ──► disagreement_report.{json,md}
   ai_sweep.py         ──► sensitivity.md + figures
```

## Quickstart

Prerequisites: Python 3.12, pip. About 2 GB free for the NIfTI volumes.

```bash
# 1. environment
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r pipeline/requirements.txt

# 2. ingest the DICOM tree (the repo ships the 255-slice RICORD series)
python -m pipeline.ingest \
    --dicom-root "MIDRC-RICORD-1A-419639-000082" \
    --out-dir pipeline_data

# 3. validate (exits 1 on failure; quarantine report on ineligible series)
python -m pipeline.validate \
    --manifest pipeline_data/manifest.parquet \
    --out-dir pipeline_data

# 4. quantify every eligible series
python -m pipeline.quantify \
    --manifest pipeline_data/manifest.parquet \
    --out-dir pipeline_data

# 5. tests
pytest tests/ -q
```

Or with DVC (installs from the same requirements file):

```bash
dvc repro            # ingest → validate → quantify
```

Or with Docker:

```bash
docker build -t ctpipe .
docker run --rm -v $PWD/pipeline_data:/app/pipeline_data ctpipe \
    python -m pipeline.ingest --dicom-root /data/dicom --out-dir pipeline_data
```

## Configuration

Every setting has a default and can be set three ways (later wins):

1. defaults in `pipeline/config.py`
2. a `.env` file in the repo root
3. environment variables prefixed `CTPIPE_` (e.g. `CTPIPE_OUT_DIR`)
4. explicit CLI flags (each CLI only applies flags you actually pass)

Secrets are env-only, never CLI flags:

| Variable | Purpose |
|---|---|
| `CTPIPE_DICOM_ROOT` | DICOM tree to ingest |
| `CTPIPE_OUT_DIR` | pipeline output dir (default `pipeline_data`) |
| `CTPIPE_LUNG_LO/HI`, `CTPIPE_INF_LO/HI` | HU bands (defaults -950/-300, -700/-200) |
| `CTPIPE_LOG_LEVEL`, `CTPIPE_LOG_FORMAT` | `INFO`, `json`/`text` |
| `GOOGLE_API_KEY` / `CTPIPE_GOOGLE_API_KEY` | VLM second reader (gemini) |
| `OPENAI_API_KEY` / `CTPIPE_OPENAI_API_KEY` | VLM second reader (openai) |

Run any module with `--help` for its flags. Exit codes: `0` ok,
`1` processing/validation failure, `2` bad input (missing path, bad flags).

## Data contracts

Two artifacts cross stage boundaries; both carry an explicit contract
version stamped into the Parquet metadata, checked on read
(`pipeline/contracts.py`). A reader that sees a version it does not know
fails loudly instead of misreading a stale file.

* `manifest/1.0` — one row per DICOM series UID: patient/study/series IDs,
  series description, modality, ImageType, orientation cosines, slice
  count, geometry, HU range, NIfTI path. Bump the version if a column is
  added, removed, or redefined.
* `results/1.0` — one row per quantified series: bands used, voxel counts,
  infection %, Wilson CI, source, code hash, run timestamp, git SHA.
  `infection_pct` is schema-bounded to [0, 100] and
  `infected_voxels <= lung_voxels` is asserted: the >100% bug class is
  unrepresentable in a valid results file.
* `sweep/1.0` — one row per HU-band configuration from `ai_sweep.py`.

## Series eligibility (the rule as code)

`pipeline/eligibility.py` decides whether a series may enter the analysis
table. Eligible = axial diagnostic CT only: modality is CT, the series
description does not name a localizer/scout/topogram survey, ImageType is
ORIGINAL (not DERIVED/SECONDARY reformats), and the direction cosines are
axial within tolerance. The rule is deliberately conservative — anything
that cannot be proven axial and diagnostic is rejected. Ineligible series
are quarantined with a machine-readable reason in
`validation_report.json`, never silently dropped. Explicit whitelists live
in `ELIGIBILITY_OVERRIDES` with a required comment.

## Quantification

`pipeline/quantify.py` is the single source of truth for the metric: the
infection numerator is infected voxels **intersected with the lung mask**,
matching the fixed `quantify_infection` in the repo's `segmentation.py`.
The segmentation itself (threshold → hole fill → largest region → smooth;
infection band → median filter) is reimplemented in numpy/scipy with the
project's standard HU bands, so the pipeline has no ITK dependency.

## Scheduled refresh

`pipeline/refresh.py` re-runs quantification over the manifest and writes
versioned results (`results/quantification_<UTC-ts>.parquet`/`.csv`) with
run timestamp, `quantify.py` code hash, and git SHA. Old versions are
kept; `quantification_latest.*` symlinks point at the newest run.
Intended as a cron job:

```cron
0 3 * * * cd /path/to/repo && .venv/bin/python -m pipeline.refresh
```

## AI components

### VLM second reader (`ai_second_reader.py`)

For a sample of slices with segmentation overlays, an independent reader
catalogs disagreement modes (vessels misclassified, partial-volume edges,
missed consolidation above the band, pleural edge noise) and writes
`disagreement_report.json` + `.md`, including agreement against the
human-written takeaways in `pipeline/data/human_takeaways.md` (curated
from the repo's own `web/REPORT.md`).

```bash
# no API key needed: rule-based stand-in reader exercises the full pipeline
python -m pipeline.ai_second_reader --manifest pipeline_data/manifest.parquet

# real study: needs GOOGLE_API_KEY (gemini) or OPENAI_API_KEY (openai)
python -m pipeline.ai_second_reader --manifest pipeline_data/manifest.parquet \
    --mode vlm --vlm-provider gemini
```

### HU-band sensitivity sweep (`ai_sweep.py`)

Varies the lung and infection bands over a grid (default: one-at-a-time
around the project standard plus a 3x3 infection-band grid, 16 configs),
re-runs quantification per config, and writes `sensitivity.md` with
figures. Compute cost is measured and reported per run; cost scales as
n_configs × n_series × t_segment. Guarded by `--max-configs` (default 40).

```bash
python -m pipeline.ai_sweep --manifest pipeline_data/manifest.parquet
```

## DVC

`dvc.yaml` defines the `ingest → validate → quantify` stages;
`params.yaml` holds the tunable parameters. No DVC remote is configured —
data stays local by default. To add one:

```bash
dvc remote add -d storage /path/to/shared/storage   # or s3://bucket/path
dvc push
```

`pipeline_data/` is git-ignored; DVC tracks the artifacts.

## Development

```bash
pip install pre-commit && pre-commit install   # black + ruff on pipeline/ and tests/
pytest tests/ -q                               # 30 tests: eligibility, contracts,
                                               # ingest on synthetic DICOMs, metric bounds
```

CI (`.github/workflows/pipeline-ci.yml`) runs the test suite and an
end-to-end ingest → validate → quantify pass on the committed RICORD
series.

## Logging

Structured logs on stderr: JSON lines by default (`CTPIPE_LOG_FORMAT=text`
for human reading). Every stage logs per-series records with UIDs, counts,
and timings; warnings mark quarantined series and skipped files.
