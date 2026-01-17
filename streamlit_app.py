import streamlit as st
import os
import numpy as np
import itk
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import seaborn as sns

# Helper to display matplotlib figures in Streamlit
def st_display_fig(fig, caption=None):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches='tight')
    st.image(buf.getvalue(), caption=caption, use_container_width=True)
    plt.close(fig)

# --- Workflow Classes and Functions (adapted from your notebook) ---
class COVIDLungSegmentation:
    def __init__(self, dicom_directory):
        self.dicom_directory = dicom_directory
        self.lower_threshold = -700
        self.upper_threshold = -200
        self.PixelType = itk.ctype("short")
        self.Dimension = 3
        self.ImageType = itk.Image[self.PixelType, self.Dimension]
        self.setup_dicom_reader()

    def setup_dicom_reader(self):
        self.names_generator = itk.GDCMSeriesFileNames.New()
        self.names_generator.SetUseSeriesDetails(True)
        self.names_generator.AddSeriesRestriction("0008|0021")
        self.names_generator.SetGlobalWarningDisplay(False)
        self.names_generator.SetDirectory(self.dicom_directory)
        self.series_uids = self.names_generator.GetSeriesUIDs()
        if not self.series_uids:
            raise ValueError(f"No DICOM series found in {self.dicom_directory}")
        self.series_identifier = self.series_uids[0]
        self.filenames = self.names_generator.GetFileNames(self.series_identifier)
        self.image_reader = itk.ImageSeriesReader[self.ImageType].New()
        self.dicom_io = itk.GDCMImageIO.New()
        self.image_reader.SetImageIO(self.dicom_io)
        self.image_reader.SetFileNames(self.filenames)
        self.image_reader.ForceOrthogonalDirectionOff()

    def read_dicom_series(self):
        self.image_reader.Update()
        return self.image_reader.GetOutput()

    def segment_covid_lungs(self, lower_threshold=None, upper_threshold=None):
        lt = lower_threshold if lower_threshold is not None else self.lower_threshold
        ut = upper_threshold if upper_threshold is not None else self.upper_threshold
        input_image = self.read_dicom_series()
        FloatImageType = itk.Image[itk.F, self.Dimension]
        castFilter = itk.CastImageFilter[self.ImageType, FloatImageType].New()
        castFilter.SetInput(input_image)
        thresholdFilter = itk.BinaryThresholdImageFilter[FloatImageType, FloatImageType].New()
        thresholdFilter.SetInput(castFilter.GetOutput())
        thresholdFilter.SetLowerThreshold(lt)
        thresholdFilter.SetUpperThreshold(ut)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        kernel = itk.FlatStructuringElement[self.Dimension].Ball(2)
        medianFilter = itk.MedianImageFilter[FloatImageType, FloatImageType].New()
        medianFilter.SetInput(thresholdFilter.GetOutput())
        medianFilter.SetRadius(2)
        medianFilter.Update()
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        binaryCastFilter = itk.CastImageFilter[FloatImageType, BinaryImageType].New()
        binaryCastFilter.SetInput(medianFilter.GetOutput())
        binaryCastFilter.Update()
        return input_image, binaryCastFilter.GetOutput()

    def segment_lungs(self, input_image):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        castFilter = itk.CastImageFilter[self.ImageType, FloatImageType].New()
        castFilter.SetInput(input_image)
        thresholdFilter = itk.BinaryThresholdImageFilter[FloatImageType, FloatImageType].New()
        thresholdFilter.SetInput(castFilter.GetOutput())
        thresholdFilter.SetLowerThreshold(-950)
        thresholdFilter.SetUpperThreshold(-300)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()
        binaryCastFilter = itk.CastImageFilter[FloatImageType, BinaryImageType].New()
        binaryCastFilter.SetInput(thresholdFilter.GetOutput())
        binaryCastFilter.Update()
        holeFillFilter = itk.BinaryFillholeImageFilter[BinaryImageType].New()
        holeFillFilter.SetInput(binaryCastFilter.GetOutput())
        holeFillFilter.SetForegroundValue(1)
        holeFillFilter.Update()
        castFilter2 = itk.CastImageFilter[BinaryImageType, FloatImageType].New()
        castFilter2.SetInput(holeFillFilter.GetOutput())
        castFilter2.Update()
        distanceFilter = itk.SignedMaurerDistanceMapImageFilter[FloatImageType, FloatImageType].New()
        distanceFilter.SetInput(castFilter2.GetOutput())
        distanceFilter.SetInsideIsPositive(True)
        distanceFilter.SetUseImageSpacing(True)
        distanceFilter.Update()
        distance_image = distanceFilter.GetOutput()
        watershedFilter = itk.WatershedImageFilter[FloatImageType].New()
        watershedFilter.SetInput(distance_image)
        watershedFilter.SetThreshold(0.001)
        watershedFilter.SetLevel(0.01)
        watershedFilter.Update()
        watershed_np = itk.GetArrayViewFromImage(watershedFilter.GetOutput())
        labels, counts = np.unique(watershed_np[watershed_np != 0], return_counts=True)
        largest_region_label = labels[np.argmax(counts)]
        binary_mask = np.where(watershed_np == largest_region_label, 1, 0).astype(np.uint8)
        binary_image = itk.GetImageFromArray(binary_mask)
        binary_image.CopyInformation(input_image)
        medianFilter = itk.MedianImageFilter[BinaryImageType, BinaryImageType].New()
        medianFilter.SetInput(binary_image)
        medianFilter.SetRadius(5)
        medianFilter.Update()
        return medianFilter.GetOutput()

    def quantify_infection(self, lung_mask, segmentation_mask):
        mask_array = itk.GetArrayFromImage(segmentation_mask)
        lung_array = itk.GetArrayFromImage(lung_mask)
        total_voxels = np.sum(lung_array > 0)
        infected_voxels = np.sum(mask_array > 0)
        infection_percentage = (infected_voxels / total_voxels) * 100
        return {
            'total_voxels': total_voxels,
            'infected_voxels': infected_voxels,
            'infection_percentage': infection_percentage
        }

# --- Streamlit App ---
st.set_page_config(page_title="COVID-19 Lung Segmentation Workflow", layout="wide")


# Sidebar with a more visually appealing public image
st.sidebar.image(
    "https://nyulangone.org/news/sites/default/files/styles/hero/public/2020-11/nyul-fall-2020-covid-lung.jpg?h=4321480b&itok=KMcRwc0w",
    caption=" ",
    use_container_width=True
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Segmentation Workflow", "Patient Data Analysis"])
st.sidebar.markdown("---")
st.sidebar.info("COVID-19 Lung Segmentation")

# Main Title with a public hero image
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://nyulangone.org/news/sites/default/files/styles/inset/public/2021-05/press-release-artificial-intelligence-tool-uses-chest-x-ray.jpeg?itok=n2B18qzW' width='60%' style='border-radius: 10px; margin-bottom: 20px;'/>
        <h1 style='color: #2c3e50;'>COVID-19 Lung Segmentation and Quantification Workflow</h1>
        <p style='color: #555;'>This app demonstrates the COVID-19 lung segmentation pipeline step-by-step, and provides data analysis of patient infection quantification.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Path to sample DICOM folder (update as needed)
dicom_dir = '/Users/anurag/Downloads/CSC821 Files/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-419639-000082/08-02-2002-NA-CT CHEST WITHOUT CONTRAST-04614/3.000000-0.625mm bone alg-26970'

if page == "Segmentation Workflow":
    st.header("Segmentation Workflow")
    segmenter = COVIDLungSegmentation(dicom_dir)
    input_image = segmenter.read_dicom_series()
    input_np = itk.GetArrayViewFromImage(input_image)
    mid_slice = input_np.shape[0] // 2
    # Step 1 & 2: Show original and COVID mask side by side
    st.subheader("1. Original CT Slice and COVID-19 Segmentation Mask")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.imshow(input_np[mid_slice], cmap='gray')
        ax1.axis('off')
        ax1.set_title(f"Original CT Slice {mid_slice}")
        st_display_fig(fig1)
    _, covid_mask = segmenter.segment_covid_lungs()
    covid_np = itk.GetArrayViewFromImage(covid_mask)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.imshow(input_np[mid_slice], cmap='gray')
        ax2.imshow(np.ma.masked_where(covid_np[mid_slice]==0, covid_np[mid_slice]), alpha=0.5, cmap='autumn')
        ax2.axis('off')
        ax2.set_title("COVID-19 Segmentation Overlay")
        st_display_fig(fig2)

    # Step 3: Lung Mask
    st.subheader("2. Lung Mask Segmentation")
    lung_mask = segmenter.segment_lungs(input_image)
    lung_np = itk.GetArrayViewFromImage(lung_mask)
    col3, col4 = st.columns(2)
    with col3:
        fig3, ax3 = plt.subplots()
        ax3.imshow(input_np[mid_slice], cmap='gray')
        ax3.imshow(np.ma.masked_where(lung_np[mid_slice]==0, lung_np[mid_slice]), alpha=0.4, cmap='Blues')
        ax3.axis('off')
        ax3.set_title("Lung Mask Overlay")
        st_display_fig(fig3)
    with col4:
        fig4, ax4 = plt.subplots()
        overlay = np.copy(lung_np[mid_slice])
        overlay[covid_np[mid_slice] > 0] = 2
        cmap = plt.cm.get_cmap('gray', 3)
        ax4.imshow(overlay, cmap=cmap, vmin=0, vmax=2)
        ax4.axis('off')
        ax4.set_title("Overlay: Infection on Lung Mask")
        st_display_fig(fig4)

    # Step 4: Quantification
    st.subheader("3. Infection Quantification")
    stats = segmenter.quantify_infection(lung_mask, covid_mask)
    col5, col6, col7 = st.columns(3)
    col5.metric("Total Lung Voxels", f"{stats['total_voxels']:,}")
    col6.metric("Infected Voxels", f"{stats['infected_voxels']:,}")
    col7.metric("Infection %", f"{stats['infection_percentage']:.2f}%")
    st.success("Workflow complete! You can change the DICOM folder path in the code to try other cases.")

elif page == "Patient Data Analysis":
    st.header("Patient Data Analysis")
    # Load grouped_by_subject_id.csv
    grouped_path = os.path.join(os.path.dirname(__file__), 'grouped_by_subject_id.csv')
    quant_path = os.path.join(os.path.dirname(__file__), 'infection_quantification_by_subject.csv')
    if os.path.exists(grouped_path):
        grouped_df = pd.read_csv(grouped_path)
        st.subheader("Grouped by Subject ID")
        st.dataframe(grouped_df.head(20))
        st.markdown(f"<small>Showing first 20 rows of {grouped_df.shape[0]} subjects.</small>", unsafe_allow_html=True)
    else:
        st.warning("grouped_by_subject_id.csv not found.")
    # Infection quantification
    if os.path.exists(quant_path):
        quant_df = pd.read_csv(quant_path)
        # Remove outliers: infection_percentage > 100
        quant_df = quant_df[quant_df['infection_percentage'] <= 100]
        st.subheader("Infection Quantification by Subject (Outliers Removed)")
        st.dataframe(quant_df.head(20))
        st.markdown(f"<small>Showing first 20 rows of {quant_df.shape[0]} subjects (outliers removed).</small>", unsafe_allow_html=True)
        # Plot histogram and bar chart
        st.markdown("### Infection Percentage Distribution")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(quant_df['infection_percentage'], bins=20, kde=True, color='crimson', ax=ax)
        ax.set_xlabel('Infection Percentage (%)')
        ax.set_ylabel('Number of Patients')
        st_display_fig(fig)
    else:
        st.warning("infection_quantification_by_subject.csv not found.")
