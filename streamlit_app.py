
import streamlit as st
import os
import numpy as np
import itk
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import seaborn as sns
import gc
import psutil

# Helper to display matplotlib figures in Streamlit
# Helper to display matplotlib figures in Streamlit
def memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    return mem

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


    # --- Sub-functions for COVID segmentation ---
    def cast_to_float(self, image):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        castFilter = itk.CastImageFilter[self.ImageType, FloatImageType].New()
        castFilter.SetInput(image)
        castFilter.Update()
        return castFilter.GetOutput()

    def threshold_image(self, image, lower, upper):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        thresholdFilter = itk.BinaryThresholdImageFilter[FloatImageType, FloatImageType].New()
        thresholdFilter.SetInput(image)
        thresholdFilter.SetLowerThreshold(lower)
        thresholdFilter.SetUpperThreshold(upper)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()
        return thresholdFilter.GetOutput()

    def median_filter(self, image, radius=2):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        medianFilter = itk.MedianImageFilter[FloatImageType, FloatImageType].New()
        medianFilter.SetInput(image)
        medianFilter.SetRadius(radius)
        medianFilter.Update()
        return medianFilter.GetOutput()

    def cast_to_binary(self, image):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        binaryCastFilter = itk.CastImageFilter[FloatImageType, BinaryImageType].New()
        binaryCastFilter.SetInput(image)
        binaryCastFilter.Update()
        return binaryCastFilter.GetOutput()

    def segment_covid_lungs(self, lower_threshold=None, upper_threshold=None):
        lt = lower_threshold if lower_threshold is not None else self.lower_threshold
        ut = upper_threshold if upper_threshold is not None else self.upper_threshold
        input_image = self.read_dicom_series()
        float_img = self.cast_to_float(input_image)
        thresh_img = self.threshold_image(float_img, lt, ut)
        median_img = self.median_filter(thresh_img, radius=2)
        binary_img = self.cast_to_binary(median_img)
        return input_image, binary_img

    # --- Sub-functions for Lung segmentation ---
    def threshold_lung(self, float_img):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        thresholdFilter = itk.BinaryThresholdImageFilter[FloatImageType, FloatImageType].New()
        thresholdFilter.SetInput(float_img)
        thresholdFilter.SetLowerThreshold(-950)
        thresholdFilter.SetUpperThreshold(-300)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()
        return thresholdFilter.GetOutput()

    def binary_cast(self, float_img):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        binaryCastFilter = itk.CastImageFilter[FloatImageType, BinaryImageType].New()
        binaryCastFilter.SetInput(float_img)
        binaryCastFilter.Update()
        return binaryCastFilter.GetOutput()

    def fill_holes(self, binary_img):
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        holeFillFilter = itk.BinaryFillholeImageFilter[BinaryImageType].New()
        holeFillFilter.SetInput(binary_img)
        holeFillFilter.SetForegroundValue(1)
        holeFillFilter.Update()
        return holeFillFilter.GetOutput()

    def cast_for_distance(self, binary_img):
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        FloatImageType = itk.Image[itk.F, self.Dimension]
        castFilter2 = itk.CastImageFilter[BinaryImageType, FloatImageType].New()
        castFilter2.SetInput(binary_img)
        castFilter2.Update()
        return castFilter2.GetOutput()

    def distance_map(self, float_img):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        distanceFilter = itk.SignedMaurerDistanceMapImageFilter[FloatImageType, FloatImageType].New()
        distanceFilter.SetInput(float_img)
        distanceFilter.SetInsideIsPositive(True)
        distanceFilter.SetUseImageSpacing(True)
        distanceFilter.Update()
        return distanceFilter.GetOutput()

    def watershed(self, float_img):
        FloatImageType = itk.Image[itk.F, self.Dimension]
        watershedFilter = itk.WatershedImageFilter[FloatImageType].New()
        watershedFilter.SetInput(float_img)
        watershedFilter.SetThreshold(0.001)
        watershedFilter.SetLevel(0.01)
        watershedFilter.Update()
        return watershedFilter.GetOutput()

    def extract_largest_region(self, watershed_img, input_image):
        watershed_np = itk.GetArrayViewFromImage(watershed_img)
        labels, counts = np.unique(watershed_np[watershed_np != 0], return_counts=True)
        if len(labels) == 0:
            print("[LungSeg] WARNING: No regions found in watershed output. Returning empty mask.")
            # Return an empty mask with the same shape as input
            shape = watershed_np.shape
            binary_mask = np.zeros(shape, dtype=np.uint8)
            binary_image = itk.GetImageFromArray(binary_mask)
            binary_image.CopyInformation(input_image)
            return binary_image
        largest_region_label = labels[np.argmax(counts)]
        binary_mask = np.where(watershed_np == largest_region_label, 1, 0).astype(np.uint8)
        binary_image = itk.GetImageFromArray(binary_mask)
        binary_image.CopyInformation(input_image)
        return binary_image

    def median_filter_binary(self, binary_img, radius=5):
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        medianFilter = itk.MedianImageFilter[BinaryImageType, BinaryImageType].New()
        medianFilter.SetInput(binary_img)
        medianFilter.SetRadius(radius)
        medianFilter.Update()
        return medianFilter.GetOutput()

    def segment_lungs(self, input_image):
        float_img = self.cast_to_float(input_image)
        thresh_img = self.threshold_lung(float_img)
        binary_img = self.binary_cast(thresh_img)
        filled_img = self.fill_holes(binary_img)
        float_img2 = self.cast_for_distance(filled_img)
        dist_img = self.distance_map(float_img2)
        watershed_img = self.watershed(dist_img)
        largest_region = self.extract_largest_region(watershed_img, input_image)
        median_img = self.median_filter_binary(largest_region, radius=5)
        return median_img

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
dicom_dir = 'MIDRC-RICORD-1A-419639-000082/08-02-2002-NA-CT CHEST WITHOUT CONTRAST-04614/2.000000-ROUTINE CHEST NON-CON-97100'

if page == "Segmentation Workflow":

    st.header("Segmentation Workflow")
    segmenter = COVIDLungSegmentation(dicom_dir)
    mem_placeholder = st.empty()
    def update_mem():
        mem_placeholder.info(f"**Current memory usage:** {memory_usage_mb():.1f} MB")

    input_image = segmenter.read_dicom_series()
    update_mem()
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
    update_mem()
    covid_np = itk.GetArrayViewFromImage(covid_mask)
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.imshow(input_np[mid_slice], cmap='gray')
        ax2.imshow(np.ma.masked_where(covid_np[mid_slice]==0, covid_np[mid_slice]), alpha=0.5, cmap='autumn')
        ax2.axis('off')
        ax2.set_title("COVID-19 Segmentation Overlay")
        st_display_fig(fig2)

    # Step 3: Lung Mask
    # --- Stepwise Lung Segmentation to avoid Streamlit crash ---
    st.subheader("2. Lung Mask Segmentation")
    progress = st.progress(0, text="Starting lung segmentation...")
    status = st.empty()
    print("[LungSeg] Step 1: Cast to float")
    status.info("Casting to float...")
    FloatImageType = itk.Image[itk.F, segmenter.Dimension]
    BinaryImageType = itk.Image[itk.UC, segmenter.Dimension]
    image = input_image
    # Step 1: Cast to float
    image = segmenter.cast_to_float(image)
    del input_image
    gc.collect()
    update_mem()
    progress.progress(10, text="Thresholding lung tissue...")
    status.info("Thresholding lung tissue...")
    print("[LungSeg] Step 2: Thresholding")
    # Step 2: Thresholding
    prev_image = image
    image = segmenter.threshold_lung(image)
    del prev_image
    gc.collect()
    update_mem()
    progress.progress(30, text="Converting to binary image...")
    status.info("Converting to binary image......")
    print("[LungSeg] Step 3: Convert to binary")
    # Step 3: Convert to binary
    prev_image = image
    image = segmenter.binary_cast(image)
    del prev_image
    gc.collect()
    update_mem()
    progress.progress(45, text="Filling holes...")
    status.info("Filling holes......")
    print("[LungSeg] Step 4: Hole filling")
    # Step 4: Hole filling
    prev_image = image
    image = segmenter.fill_holes(image)
    del prev_image
    gc.collect()
    update_mem()
    progress.progress(60, text="Casting for distance map...")
    status.info("Casting for distance map....")
    print("[LungSeg] Step 5: Cast for distance map")
    # Step 5: Cast for distance map
    prev_image = image
    image = segmenter.cast_for_distance(image)
    del prev_image
    gc.collect()
    update_mem()
    progress.progress(70, text="Computing distance map...")
    status.info("Computing distance map....")
    print("[LungSeg] Step 6: Distance map")
    # Step 6: Distance map
    prev_image = image
    image = segmenter.distance_map(image)
    del prev_image
    gc.collect()
    update_mem()
    progress.progress(80, text="Watershed segmentation...")
    status.info("Watershed segmentation...")
    print("[LungSeg] Step 7: Watershed")
    # Delete all images and clear cache before watershed
    try:
        del prev_image
    except Exception:
        pass
    try:
        del covid_np
    except Exception:
        pass
    try:
        del lung_np
    except Exception:
        pass
    gc.collect()
    update_mem()
    st.cache_resource.clear()
    # Step 7: Watershed
    watershed_img = segmenter.watershed(image)
    del image
    gc.collect()
    update_mem()
    st.cache_resource.clear()
    # Convert watershed ITK image to numpy as early as possible to free memory
    watershed_np = itk.GetArrayViewFromImage(watershed_img)
    del watershed_img
    gc.collect()
    update_mem()
    progress.progress(90, text="Extract Largest Region...")
    status.info("Extract Largest Region....")
    print("[LungSeg] Step 8: Extract largest region")
    labels, counts = np.unique(watershed_np[watershed_np != 0], return_counts=True)
    if len(labels) == 0:
        st.warning("No lung region found in watershed output. The mask is empty. Please check your input data.")
        print("[LungSeg] WARNING: No lung region found in watershed output. The mask is empty.")
        binary_mask = np.zeros(watershed_np.shape, dtype=np.uint8)
    else:
        largest_region_label = labels[np.argmax(counts)]
        binary_mask = np.where(watershed_np == largest_region_label, 1, 0).astype(np.uint8)
    del watershed_np
    gc.collect()
    # Convert back to ITK image for median filtering
    image = itk.GetImageFromArray(binary_mask)
    image.CopyInformation(segmenter.read_dicom_series())
    del binary_mask
    gc.collect()
    update_mem()
    progress.progress(90, text="Median filtering...")
    print("[LungSeg] Step 9: Median filtering")
    status.info("Median filtering...")
    prev_image = image
    image = segmenter.median_filter_binary(image, radius=5)
    del prev_image
    gc.collect()
    update_mem()
    st.cache_resource.clear()
    lung_mask = image
    del image
    gc.collect()
    progress.progress(100, text="Lung segmentation complete!")
    status.success("Lung segmentation complete!")
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
    st.success("Workflow complete!")

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
