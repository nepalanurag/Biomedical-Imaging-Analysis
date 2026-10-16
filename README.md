# COVID-19 Lung Segmentation and Quantification

This project provides a complete pipeline for segmenting COVID-19 infected lung regions from DICOM images, quantifying infection percentage, and visualizing results. The web front end is a static site deployed on Vercel (see Live demo below); there is no Streamlit app.

## Live demo

Try it in your browser: https://covid-ct-web.vercel.app/

The full demo package (analysis and the site source) lives in [`web/`](web/).

## Method (honest version)

HU-thresholding plus morphological operations. No machine learning, no neural
network, no trained weights anywhere:

1. **Lung mask**: keep voxels in [-950, -300] HU, fill holes, keep the largest
   connected region, then morphological closing (disk, radius 9) to repair
   juxtapleural boundary gaps where dense infection is thresholded out of the
   lung band.
2. **Infection mask**: keep voxels in [-700, -200] HU (the ground-glass band),
   3D median filter (radius 2) to drop specks.
3. **Score**: infected voxels intersected with the lung mask, divided by lung
   voxels. The numerator counts only voxels inside the lung, so the percentage
   can never exceed 100.

## Evaluation against radiologist annotations

The method is scored against real ground truth: radiologist-drawn "infectious
opacity" annotations from the TCIA MIDRC-RICORD-1A collection (MD.ai format,
annotator Team 1). 142 of the 255 slices in series 419639-000082 carry 346
annotation polygons.

| Metric | Value |
|---|---|
| Dice | 0.7115 |
| IoU | 0.5522 |
| Annotated slices | 142 / 255 |
| Series | MIDRC-RICORD-1A-419639-000082 |

The annotations also drove a real fix: the original lung mask lost infection
near the pleura (Dice 0.667); adding the morphological closing repair lifted it
to 0.712. An earlier version of the repo's annotation-loading code flipped the
polygons upside down, which is why the integration "never worked" before.

To reproduce:

1. Download the annotations JSON from the TCIA
   [MIDRC-RICORD-1A collection page](https://www.cancerimagingarchive.net/collection/midrc-ricord-1a/)
   (file `MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json`, ~47 MB,
   CC BY-NC 4.0) and place it in the repo root.
2. Run `python evaluate_annotations.py` (needs the DICOM series folder
   `MIDRC-RICORD-1A-419639-000082/` in the repo root).
3. Run `python verify_integration.py` to confirm the production pipeline
   (`pipeline/quantify.py`) reproduces the same numbers.

Caveats: one CT series, one annotator team. These numbers describe this scan,
not COVID CT in general.

## Features

- **DICOM Series Reading and Processing**: Reads DICOM series and converts them to ITK images.
- **COVID-19 Infection Segmentation**: Segments infected lung regions using Hounsfield Unit (HU) thresholds and morphological operations.
- **Lung Segmentation**: Segments the entire lung region for accurate quantification, with morphological closing to repair juxtapleural gaps.
- **Infection Quantification**: Calculates the percentage of infected lung tissue.
- **Visualization**: Slice overlays and the static web demo in [`web/`](web/).
- **Annotation Integration**: Loads MD.ai radiologist annotation files as ground truth and reports Dice/IoU (see Evaluation above).
- **Patient Data Analysis**: Visualizes infection statistics and patient grouping from CSV files.

## DICOM ingestion pipeline

`pipeline/` is a production ingestion pipeline for the CT analysis above:
DICOM → NIfTI + a versioned Parquet manifest, Pandera validation gates, a
series-eligibility rule as code (axial diagnostic CT only; scout/localizer
and reformatted series are quarantined, never silently dropped), and
quantification using the fixed intersection-based infection metric as the
single source of truth. It also ships two AI components: a VLM
second-reader study that catalogs segmentation disagreement modes against
the human-written takeaways, and an agentic HU-band sensitivity sweep.

```bash
pip install -r pipeline/requirements.txt
python -m pipeline.ingest --dicom-root "MIDRC-RICORD-1A-419639-000082" --out-dir pipeline_data
python -m pipeline.validate --manifest pipeline_data/manifest.parquet --out-dir pipeline_data
python -m pipeline.quantify --manifest pipeline_data/manifest.parquet --out-dir pipeline_data
pytest tests/ -q
```

See [`pipeline/README.md`](pipeline/README.md) for the full documentation:
architecture, configuration, versioned data contracts, DVC stages, Docker,
and CI.

## Quickstart

1. **Clone the repository**
   ```bash
   git clone https://github.com/nepalanurag/Biomedical-Imaging-Analysis.git
   cd Biomedical-Imaging-Analysis
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Upload or update your DICOM and CSV files as needed.**

## File Structure

- `segmentation.py`: core segmentation logic (ITK-based, if used separately).
- `pipeline/`: production ingestion pipeline (DICOM to NIfTI, validation, quantification).
- `evaluate_annotations.py`: scores the segmentation against MD.ai radiologist annotations (Dice/IoU).
- `eval_improvements.py`: evaluates segmentation variants against the annotations.
- `improve_segment.py`: improved segmentation variants (lung-mask closing repair).
- `verify_integration.py`: confirms the production pipeline reproduces the validated metrics.
- `web/`: static web demo source (deployed on Vercel).
- `grouped_by_subject_id.csv`: grouped patient DICOM metadata.
- `infection_quantification_by_subject.csv`: infection quantification per subject.
- `FINAL_PRESENTATION.ipynb`: Jupyter notebook with the full workflow.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Example Data

- Place your DICOM folders and CSV files in the project directory as described in the notebook and app.

## Acknowledgements

- COVID-19 CT images: MIDRC-RICORD-1A dataset
- Public images: Unsplash, NYU Langone Health
- Libraries: ITK, Numpy, Matplotlib, Pandas, Seaborn, Nibabel, scikit-image, tqdm, dicom2nifti, mdai

## License

This project is for academic and research use.
