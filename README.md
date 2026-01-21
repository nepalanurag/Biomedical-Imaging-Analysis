# COVID-19 Lung Segmentation and Quantification

This project provides a complete pipeline for segmenting COVID-19 infected lung regions from DICOM images, quantifying infection percentage, and visualizing results. It includes a Streamlit web app for interactive exploration and patient data analysis.

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

- `streamlit_app.py` — Main Streamlit web application.
- `segmentation.py` — Core segmentation logic (if used separately).
- `grouped_by_subject_id.csv` — Grouped patient DICOM metadata.
- `infection_quantification_by_subject.csv` — Infection quantification per subject.
- `FINAL_PRESENTATION.ipynb` — Jupyter notebook with the full workflow.

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

This project is for academic and research use. See LICENSE for details.
