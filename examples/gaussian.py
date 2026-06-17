from ThorlabsGaussianTools import VisualizationGaussianParameters, CameraParameters, GaussianFitImaging

myCamParams = CameraParameters(exposure_time_us=1)

myVisParams = VisualizationGaussianParameters(
    fontsize=16,
    magnification=23.,
    lengthscale_um=10,
    zoom_bool=True,
    zoom_width=100,
    downscale_bool=True,
    downscale_order=4,
    gaussian_filter_sigma=2,
    gaussian_fitting=True
)

myAnim = GaussianFitImaging(myCamParams, myVisParams, verbosity=2)
myAnim.rich_print_params()
myAnim.run(number_of_frames=100)

del myAnim