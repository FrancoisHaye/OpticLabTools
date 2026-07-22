"""

Usage of SimpleImaging class to view the gaussian beam in direct.

"""

from ThorlabsGaussianTools import VisualizationParameters, CameraParameters, SimpleImaging

myCamParams = CameraParameters(exposure_time_us=1)

myVisParams = VisualizationParameters(
    fontsize=16,
    magnification=23.,
    lengthscale_um=10,
    zoom_bool=True,
    zoom_width=100
)

myAnim = SimpleImaging(myCamParams, myVisParams, verbosity=2)
myAnim.rich_print_params()
myAnim.run(number_of_frames=100)

del myAnim