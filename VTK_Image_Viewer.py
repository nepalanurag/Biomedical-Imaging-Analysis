import os
import pydicom
import numpy as np
import vtkmodules.vtkRenderingContextOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionImage import vtkImageViewer2
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import (
    vtkActor2D,
    vtkRenderWindowInteractor,
    vtkTextMapper,
    vtkTextProperty
)
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtk.util import numpy_support


# Helper class to format slice status message
class StatusMessage:
    @staticmethod
    def format(slice: int, max_slice: int):
        return f'Slice Number {slice + 1}/{max_slice + 1}'


# Define own interaction style. Taken straight from the VTK Docs
class MyVtkInteractorStyleImage(vtkInteractorStyleImage):
    def __init__(self, parent=None):
        super().__init__()
        self.AddObserver('KeyPressEvent', self.key_press_event)
        self.AddObserver('MouseWheelForwardEvent', self.mouse_wheel_forward_event)
        self.AddObserver('MouseWheelBackwardEvent', self.mouse_wheel_backward_event)
        self.image_viewer = None
        self.status_mapper = None
        self.slice = 0
        self.min_slice = 0
        self.max_slice = 0

    def set_image_viewer(self, image_viewer):
        self.image_viewer = image_viewer
        self.min_slice = image_viewer.GetSliceMin()
        self.max_slice = image_viewer.GetSliceMax()
        self.slice = self.min_slice
        print(f'Slicer: Min = {self.min_slice}, Max= {self.max_slice}')

    def set_status_mapper(self, status_mapper):
        self.status_mapper = status_mapper

    def move_slice_forward(self):
        if self.slice < self.max_slice:
            self.slice += 1
            print(f'MoveSliceForward::Slice = {self.slice}')
            self.image_viewer.SetSlice(self.slice)
            msg = StatusMessage.format(self.slice, self.max_slice)
            self.status_mapper.SetInput(msg)
            self.image_viewer.Render()

    def move_slice_backward(self):
        if self.slice > self.min_slice:
            self.slice -= 1
            print(f'MoveSliceBackward::Slice = {self.slice}')
            self.image_viewer.SetSlice(self.slice)
            msg = StatusMessage.format(self.slice, self.max_slice)
            self.status_mapper.SetInput(msg)
            self.image_viewer.Render()

    def key_press_event(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'Up':
            self.move_slice_forward()
        elif key == 'Down':
            self.move_slice_backward()

    def mouse_wheel_forward_event(self, obj, event):
        self.move_slice_forward()

    def mouse_wheel_backward_event(self, obj, event):
        self.move_slice_backward()

#Taking the DICOM folder and then converting every image in it to a numpy array. Axis=-1 means a 3D image.
def load_dicom_series(folder):
    dicom_files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".dcm")]
    )
    slices = []
        
    for file in dicom_files:
        dicom_data = pydicom.dcmread(file)
        slices.append(dicom_data.pixel_array)
    
    volume_3d = np.stack(slices, axis=-1)
    
    volume_3d = np.interp(volume_3d, (volume_3d.min(), volume_3d.max()), (0, 255))
    return volume_3d.astype(np.uint8)

# Takes the numpy array to flatten it and then convert it to the shape of 3D image shape(x,y,z)
def numpy_to_vtk_image(numpy_array):
    vtk_image = vtkImageData()
    vtk_image.SetDimensions(numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2])
    
    flat_array = numpy_array.ravel(order='F')
    vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True, array_type=vtkmodules.vtkCommonCore.VTK_UNSIGNED_CHAR)
    
    vtk_image.GetPointData().SetScalars(vtk_array)
    return vtk_image

#This is same from the VTK docs change stuff if you like to tweak anything
def vtk_image_view(vtk_image):
    colors = vtkNamedColors()
    # Visualilze
    image_viewer = vtkImageViewer2()
    image_viewer.SetInputData(vtk_image)
    # Slice status message
    slice_text_prop = vtkTextProperty()
    slice_text_prop.SetFontFamilyToCourier()
    slice_text_prop.SetFontSize(20)
    slice_text_prop.SetVerticalJustificationToBottom()
    slice_text_prop.SetJustificationToLeft()
    # Slice status message
    slice_text_mapper = vtkTextMapper()
    msg = StatusMessage.format(image_viewer.GetSliceMin(), image_viewer.GetSliceMax())
    slice_text_mapper.SetInput(msg)
    slice_text_mapper.SetTextProperty(slice_text_prop)

    slice_text_actor = vtkActor2D()
    slice_text_actor.SetMapper(slice_text_mapper)
    slice_text_actor.SetPosition(15, 10) 
    
    # Usage hint message
    usage_text_prop = vtkTextProperty()
    usage_text_prop.SetFontFamilyToCourier()
    usage_text_prop.SetFontSize(14)
    usage_text_prop.SetVerticalJustificationToTop()
    usage_text_prop.SetJustificationToLeft()
    usage_text_mapper = vtkTextMapper()
    usage_text_mapper.SetInput(
        'Move slice with mouse wheel\n  or Up/Down-Key\n- Zoom with pressed right\n '
        ' mouse button while dragging'
    )
    usage_text_mapper.SetTextProperty(usage_text_prop)

    usage_text_actor = vtkActor2D()
    usage_text_actor.SetMapper(usage_text_mapper)
    usage_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    usage_text_actor.GetPositionCoordinate().SetValue(0.05, 0.95)
    
    # Create an interactor with our own style (inherit from
    # vtkInteractorStyleImage in order to catch mousewheel and key events.
    render_window_interactor = vtkRenderWindowInteractor()
    my_interactor_style = MyVtkInteractorStyleImage()

    # Make imageviewer2 and sliceTextMapper visible to our interactorstyle
    # to enable slice status message updates when  scrolling through the slices.
    my_interactor_style.set_image_viewer(image_viewer)
    my_interactor_style.set_status_mapper(slice_text_mapper)

    # Make the interactor use our own interactor style
    # because SetupInteractor() is defining it's own default interator style
    # this must be called after SetupInteractor().
    # renderWindowInteractor.SetInteractorStyle(myInteractorStyle);
    image_viewer.SetupInteractor(render_window_interactor)
    render_window_interactor.SetInteractorStyle(my_interactor_style)
    render_window_interactor.Render()

    # Add slice status message and usage hint message to the renderer.
    image_viewer.GetRenderer().AddActor2D(slice_text_actor)
    image_viewer.GetRenderer().AddActor2D(usage_text_actor)

    # Initialize rendering and interaction.
    image_viewer.Render()
    image_viewer.GetRenderer().ResetCamera()
    image_viewer.GetRenderer().SetBackground(colors.GetColor3d('Black'))
    image_viewer.GetRenderWindow().SetSize(800, 800)
    image_viewer.GetRenderWindow().SetWindowName('DICOM Image Output')
    image_viewer.Render()
    image_viewer.GetRenderer().ResetCamera()
    render_window_interactor.Initialize()
    try:
        render_window_interactor.Start()
    finally:
        render_window_interactor.TerminateApp()
    
def vtk_image_show_folder(folder):
    #Load image as an numpy array
    volume_3d = load_dicom_series(folder)
    #Convert numpy to image
    vtk_image = numpy_to_vtk_image(volume_3d)
    #View the image
    vtk_image_view(vtk_image)
    