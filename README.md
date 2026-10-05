# COVID-19 Lung Segmentation and Quantification

This project provides a complete pipeline for segmenting COVID-19 infected lung regions from DICOM images, quantifying infection percentage, and visualizing results. It includes a Streamlit web app for interactive exploration and patient data analysis.

## Live demo

Try it in your browser: https://covid-ct-web.vercel.app/

The full demo package (analysis and the site source) lives in [`web/`](web/).

## Features

- **DICOM Series Reading and Processing**: Reads DICOM series, handles series details, and converts to ITK images.
- **COVID-19 Infection Segmentation**: Segments infected lung regions using Hounsfield Unit (HU) thresholds and morphological operations.
- **Lung Segmentation**: Segments the entire lung region for accurate quantification.
- **Infection Quantification**: Calculates the percentage of infected lung tissue.
- **Visualization**: Interactive slice viewing and overlays using Matplotlib and Streamlit.
- **Annotation Integration**: Supports MD.ai annotation files for ground truth comparison.
- **Patient Data Analysis**: Visualizes infection statistics and patient grouping from CSV files.

## Streamlit Web App

The app provides:

- Step-by-step visualization of the segmentation workflow for a sample DICOM folder.
- Interactive overlays for original, lung mask, and infection mask.
- Patient data analysis with infection percentage distribution and top infected cases.

## DICOM ingestion pipeline

`pipeline/` is a production ingestion pipeline for the CT analysis above:
DICOM → NIfTI + a versioned Parquet manifest, Pandera validation gates, a
series-eligibility rule as code (axial diagnostic CT only — scout/localizer
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
3. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```
4. **Upload or update your DICOM and CSV files as needed.**

## File Structure

- `streamlit_app.py`: main Streamlit web application.
- `segmentation.py`: core segmentation logic (if used separately).
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
- Libraries: ITK, Numpy, Matplotlib, Pandas, Seaborn, Streamlit, Nibabel, scikit-image, tqdm, dicom2nifti, mdai

## License

This project is for academic and research use.
