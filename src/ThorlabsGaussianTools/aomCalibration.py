"""
``aomCalibration``
==================

Providing CLI tool for the movement of the point with only ONE AOM (select channel 1 or 2).

"""



import click
import numpy as np
import matplotlib.pyplot as plt
from ._real_time_imaging import CameraParameters, VisualizationGaussianParameters, RFanim
from .utils.mogdevice import MOGDevice
from rich.console import Console
from rich.traceback import install

cns = Console()
install(console=cns, show_locals=True)


@click.command()
@click.option('-v','--verbosity', type=int, default=2, show_default=True, help="How much information you want the code to give, between 1 and 4.")
@click.option('--exposure-time', type=int, default=1, show_default=True, help="Exposure time of the camera, in µs.")
@click.option('--magnification', type=float, default=23., show_default=True, help="The magnification of the imaging system.")
@click.option("--lengthscale", type=int, default=None, show_default=True, help="The desired size for the scalebar, in µm.")
@click.option("-z","--zoom-width", type=int, default=None, show_default=True, help="Half width of the zooming window around the point. If None, no zooming is performed.")
@click.option("-ds", "--downscale", type=int, default=5, show_default=True, help="Order of downscaling, in pixels. If None, no downscaling")
@click.option("--mog-port", type=int, default=7, show_default=True, help="Port of the USB MogDevice. Use Mogrf.exe to get this value.")
@click.option("-s", "--start-frequency", type=float, default=50, show_default=True, help="Starting frequency for the AOM frequency span.")
@click.option("-e", "--end-frequency", type=float, default=110, show_default=True, help="End frequency of the span.")
@click.option("-n", "--number-frequencies", type=int, default=1000, show_default=True, help="Number of frequencies for the span. A greater value means more precision but a longer execution time.")
def main(
    verbosity: int,
    exposure_time: int,
    magnification: float,
    lengthscale: int | None,
    zoom_width: int,
    downscale: int | None,
    mog_port: int,
    start_frequency: float,
    end_frequency: float,
    number_frequencies: int
):
    cns.print("\n\n")
    cns.rule("[bold]Calibration of AOMs")
    
    doZoom = False
    if zoom_width:
        doZoom = True
    
    doDownscale = False
    if downscale:
        doDownscale = True

    myCamParams = CameraParameters(exposure_time_us=exposure_time)

    myVisParams = VisualizationGaussianParameters(
        fontsize=12,
        magnification=magnification,
        lengthscale_um=lengthscale,
        zoom_bool=doZoom,
        zoom_width=zoom_width,
        downscale_bool=doDownscale,
        downscale_order=downscale,
        gaussian_fitting=True,
        gaussian_filter_sigma=1
        )
    
    myFreq = np.linspace(start_frequency, end_frequency, number_frequencies)
    myDumbFreq = np.ones_like(myFreq) * 80

    myAnimX = RFanim(
        mogPort=mog_port,
        freqRF1=myFreq,
        freqRF2=myDumbFreq,
        camParams=myCamParams,
        visParams=myVisParams,
        console=cns,
        verbosity=verbosity
    )

    myAnimX.rich_print_params()
    myAnimX.run()
    myFreqRF, _, myIntensity, myPositionx, myPositiony, myWx, myWy, myTheta = myAnimX.get_results()
    np.savez_compressed("AOMx",frequency=myFreqRF, intensity = myIntensity, positionx = myPositionx, positiony = myPositiony, wx = myWx, wy = myWy, theta = myTheta)

    cns.print(f"Results saved in {'AOMx.npz':>5}\nNow proceeding to the second AOM.")

    myAnimX.mogdevice.close()
    del myAnimX

    myAnimY = RFanim(
        mogPort=mog_port,
        freqRF1=myDumbFreq,
        freqRF2=myFreq,
        camParams=myCamParams,
        visParams=myVisParams,
        console=cns,
        verbosity=verbosity
    )

    myAnimY.rich_print_params()
    myAnimY.run()
    _, myFreqRF, myIntensity, myPositionx, myPositiony, myWx, myWy, myTheta = myAnimY.get_results()
    np.savez_compressed("AOMy",frequency=myFreqRF, intensity = myIntensity, positionx = myPositionx, positiony = myPositiony, wx = myWx, wy = myWy, theta = myTheta)

    del myAnimY

    cns.print(f"Results saved in {'AOMy.npz':>5}\n\n")
    cns.rule()
    cns.print("\n\n")


if __name__=="__main__":
    main()