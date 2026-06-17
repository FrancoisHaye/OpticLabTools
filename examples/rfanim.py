from ThorlabsGaussianTools import CameraParameters, VisualizationGaussianParameters, RFanim
import numpy as np

myCamParams = CameraParameters(exposure_time_us=1)

myVisParams = VisualizationGaussianParameters(
    fontsize=16,
    magnification=23.,
    lengthscale_um=10,
    zoom_bool=False,
    zoom_width=100,
    downscale_bool=True,
    downscale_order=5,
    gaussian_filter_sigma=3,
    gaussian_fitting=True
)

myFreqRF = np.linspace(70, 90, 50)
myPowRF = np.linspace(10, 30, 50)

myDumbFreqRF = np.ones_like(myFreqRF)*80
myDumbPowRF = np.ones_like(myPowRF)*30

myAnim = RFanim(
    mogPort=7,
    freqRF1=myFreqRF,
    powRF1=myPowRF,
    freqRF2=myDumbFreqRF,
    powRF2=myDumbPowRF,
    camParams=myCamParams,
    visParams=myVisParams,
    verbosity=4
)
myAnim.rich_print_params()
myAnim.run()
myAnim.get_results()

del myAnim