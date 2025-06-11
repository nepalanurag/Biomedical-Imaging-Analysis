import itk
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt

def view_nifti_slices(nifti_path, slice_indices=None, cmap='gray'):
    # Load the NIfTI file
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Get dimensions
    x, y, z = data.shape
    
    # If no slice indices provided, take middle slices
    if slice_indices is None:
        slice_indices = [x//2, y//2, z//2]
    
    # Create a figure with three subplots (one for each orientation)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sagittal view (YZ plane)
    ax1.imshow(np.rot90(data[slice_indices[0], :, :]), cmap=cmap)
    ax1.set_title(f'Sagittal Slice {slice_indices[0]}')
    
    # Coronal view (XZ plane)
    ax2.imshow(np.rot90(data[:, slice_indices[1], :]), cmap=cmap)
    ax2.set_title(f'Coronal Slice {slice_indices[1]}')
    
    # Axial view (XY plane)
    ax3.imshow(np.rot90(data[:, :, slice_indices[2]]), cmap=cmap)
    ax3.set_title(f'Axial Slice {slice_indices[2]}')
    
    plt.tight_layout()
    plt.show()

def view_nifti_interactive(nifti_path):

    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    class IndexTracker:
        def __init__(self, ax, X):
            self.ax = ax
            self.X = X
            self.slices = X.shape[2]
            self.ind = self.slices//2
            
            self.im = ax.imshow(np.rot90(self.X[:, :, self.ind]), cmap='gray')
            self.update()
    
        def onscroll(self, event):
            if event.key == 'up':
                self.ind = (self.ind + 1) % self.slices
            elif event.key == 'down':
                self.ind = (self.ind - 1) % self.slices
            self.update()
    
        def update(self):
            self.im.set_data(np.rot90(self.X[:, :, self.ind]))
            self.ax.set_title(f'Slice {self.ind}/{self.slices}')
            self.im.axes.figure.canvas.draw()
    
    fig, ax = plt.subplots()
    tracker = IndexTracker(ax, data)
    fig.canvas.mpl_connect('key_press_event', tracker.onscroll)
    plt.show()

def get_dicom_series_filenames(dicom_directory):
    # Find all .dcm files, including in subdirectories
    dicom_files = []
    for root, _, files in os.walk(dicom_directory):
        dicom_files.extend([
            os.path.join(root, f) for f in files 
            if f.lower().endswith(('.dcm', '.dicom'))
        ])
    
    # Sort files to ensure correct slice order
    dicom_files = sorted(dicom_files)
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_directory}")
    
    return dicom_files

class COVIDLungSegmentation:
    
    def __init__(self, dicom_directory):

        self.dicom_directory = dicom_directory
        
        # Specific HU ranges for COVID-19 lung involvement
        self.lower_threshold = -700  # Lung air/ground-glass opacity lower bound
        self.upper_threshold = -200  # Consolidation upper bound
        
        # Initialize ITK image types
        self.PixelType = itk.ctype("short") 
        self.Dimension = 3
        self.ImageType = itk.Image[self.PixelType, self.Dimension]
        
        # Set up DICOM series reader
        self.setup_dicom_reader()
    
    def setup_dicom_reader(self):
        # Create DICOM name generator
        self.names_generator = itk.GDCMSeriesFileNames.New()
        self.names_generator.SetUseSeriesDetails(True)
        self.names_generator.AddSeriesRestriction("0008|0021")
        self.names_generator.SetGlobalWarningDisplay(False)
        self.names_generator.SetDirectory(self.dicom_directory)
        
        # Get series UIDs
        self.series_uids = self.names_generator.GetSeriesUIDs()
        
        if not self.series_uids:
            raise ValueError(f"No DICOM series found in {self.dicom_directory}")
        
        # Get filenames for the first series
        self.series_identifier = self.series_uids[0]
        self.filenames = self.names_generator.GetFileNames(self.series_identifier)
        
        # Set up image reader
        self.image_reader = itk.ImageSeriesReader[self.ImageType].New()
        self.dicom_io = itk.GDCMImageIO.New()
        self.image_reader.SetImageIO(self.dicom_io)
        self.image_reader.SetFileNames(self.filenames)
        self.image_reader.ForceOrthogonalDirectionOff()
    
    def read_dicom_series(self):
        try:
            self.image_reader.Update()
            return self.image_reader.GetOutput()
        except Exception as e:
            print(f"Error reading DICOM series: {e}")
            return None
    
    def segment_covid_lungs(self,output_directory=None,lower_threshold=None,upper_threshold=None):
        # Use custom thresholds if provided
        lt = lower_threshold if lower_threshold is not None else self.lower_threshold
        ut = upper_threshold if upper_threshold is not None else self.upper_threshold
        
        # Prepare output directory
        if output_directory is None:
            output_directory = os.path.join(self.dicom_directory, 'covid_segmentation')
        os.makedirs(output_directory, exist_ok=True)
        
        # Read the image series
        input_image = self.read_dicom_series()
        if input_image is None:
            return None
            
        # Convert to float for processing
        FloatImageType = itk.Image[itk.F, self.Dimension]
        castFilter = itk.CastImageFilter[self.ImageType, FloatImageType].New()
        castFilter.SetInput(input_image)
        
        # Threshold segmentation
        thresholdFilter = itk.BinaryThresholdImageFilter[FloatImageType, FloatImageType].New()
        thresholdFilter.SetInput(castFilter.GetOutput())
        thresholdFilter.SetLowerThreshold(lt)
        thresholdFilter.SetUpperThreshold(ut)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        
        # Morphological operations to clean up the segmentation
        kernel = itk.FlatStructuringElement[self.Dimension].Ball(2)
        
        # median filter to remove small noise salt and pepper
        medianFilter = itk.MedianImageFilter[FloatImageType, FloatImageType].New()
        medianFilter.SetInput(thresholdFilter.GetOutput())
        medianFilter.SetRadius(2)
        medianFilter.Update()
        
        # Convert back to binary image
        BinaryImageType = itk.Image[itk.UC, self.Dimension]
        binaryCastFilter = itk.CastImageFilter[FloatImageType, BinaryImageType].New()
        binaryCastFilter.SetInput(medianFilter.GetOutput())
        binaryCastFilter.Update()
        
        if output_directory:
            writer = itk.ImageFileWriter[BinaryImageType].New()
            writer.SetFileName(os.path.join(output_directory, "covid_segmentation.nii.gz"))
            writer.SetInput(binaryCastFilter.GetOutput())
            writer.Update()
        
        print("Starting interactive viewer (use up/down arrow keys to navigate slices)...")
        view_nifti_slices(os.path.join(output_directory, "covid_segmentation.nii.gz"))
        
        return input_image,binaryCastFilter.GetOutput()
    
    def segment_lungs(self,input_image,output_directory=None):
        
        if output_directory is None:
            output_directory = os.path.join(self.dicom_directory, 'lung')
        os.makedirs(output_directory, exist_ok=True)
                
        # print(itk.size(input_image))  # Check the dimensions of the original image
        # print(itk.GetArrayViewFromImage(input_image).shape)  # Validate dimensions
        # image_array = itk.GetArrayViewFromImage(input_image)
        # plt.hist(image_array.flatten(), bins=256, range=(-2000, 2000))
        # plt.show()
        # image_array = itk.GetArrayFromImage(input_image)
        # print("Image HU Range:")
        # print(f"Minimum: {image_array.min()}")
        # print(f"Maximum: {image_array.max()}")
        # print(f"Mean: {image_array.mean()}")
        
        if input_image is None:
            print('returning none')
            return None  

        # Convert to float for initial processing
        FloatImageType = itk.Image[itk.F, self.Dimension]
        BinaryImageType = itk.Image[itk.UC, self.Dimension]

        # Initial thresholding to get lung tissue
        castFilter = itk.CastImageFilter[self.ImageType, FloatImageType].New()
        castFilter.SetInput(input_image)

        thresholdFilter = itk.BinaryThresholdImageFilter[FloatImageType, FloatImageType].New()
        thresholdFilter.SetInput(castFilter.GetOutput())
        thresholdFilter.SetLowerThreshold(-950)
        thresholdFilter.SetUpperThreshold(-300)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.Update()

        # Convert to binary image
        binaryCastFilter = itk.CastImageFilter[FloatImageType, BinaryImageType].New()
        binaryCastFilter.SetInput(thresholdFilter.GetOutput())
        binaryCastFilter.Update()  


        # # Hole filling
        holeFillFilter = itk.BinaryFillholeImageFilter[BinaryImageType].New()
        holeFillFilter.SetInput(binaryCastFilter.GetOutput())
        holeFillFilter.SetForegroundValue(1)
        holeFillFilter.Update()

        
        FloatImageType = itk.Image[itk.F, self.Dimension]  # Define explicitly here
        castFilter = itk.CastImageFilter[BinaryImageType, FloatImageType].New() # Cast from binary TO float
        castFilter.SetInput(holeFillFilter.GetOutput())
        castFilter.Update()


        # Now use the casted image as input to distance map:
        distanceFilter = itk.SignedMaurerDistanceMapImageFilter[FloatImageType, FloatImageType].New()
        distanceFilter.SetInput(castFilter.GetOutput()) 
        distanceFilter.SetInsideIsPositive(True)
        distanceFilter.SetUseImageSpacing(True)
        distanceFilter.Update()
        distance_image=distanceFilter.GetOutput()


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
        
        if output_directory:
            writer = itk.ImageFileWriter[BinaryImageType].New()
            writer.SetFileName(os.path.join(output_directory, "lungs.nii.gz"))
            writer.SetInput(medianFilter.GetOutput())
            writer.Update()
        
        print("Starting interactive viewer (use up/down arrow keys to navigate slices)...")
        view_nifti_slices(os.path.join(output_directory, "lungs.nii.gz"))
        
        return medianFilter.GetOutput()

    def quantify_infection(self,lung_mask,segmentation_mask):
        # Convert mask to numpy for analysis
        mask_array = itk.GetArrayFromImage(segmentation_mask)
        lung_array= itk.GetArrayFromImage(lung_mask)
        # Calculate infection metrics
        total_voxels = np.sum(lung_array > 0)
        infected_voxels = np.sum(mask_array > 0)
        infection_percentage = (infected_voxels / total_voxels) * 100
        
        return {
            'total_voxels': total_voxels,
            'infected_voxels': infected_voxels,
            'infection_percentage': infection_percentage
        }
        


def main():
    dicom_dir = '/Users/anurag/Downloads/CSC821 Files/manifest-1608266677008/MIDRC-RICORD-1A/MIDRC-RICORD-1A-660042-000107/11-20-2008-NA-NA-25678/4.000000-NA-25679'
    
    try:
        segmenter = COVIDLungSegmentation(dicom_dir)
        
        print("Performing segmentation...")
        input_volume,mask = segmenter.segment_covid_lungs(
            lower_threshold=-700,  
            upper_threshold=-200   
        )
        lung_mask= segmenter.segment_lungs(input_volume)
        if mask:
            print("Calculating infection statistics...")
            infection_stats = segmenter.quantify_infection(lung_mask,mask)
            print("\nInfection Statistics:")
            for key, value in infection_stats.items():
                print(f"{key}: {value}")
        else:
            print("Segmentation failed.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    main()