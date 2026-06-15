import click
import numpy as np
import matplotlib.pyplot as plt
from realTimeImaging import VisualizationGaussianParameters, RFexpImaging
from utils.mogdevice import MOGDevice
from rich.console import Console
from rich.traceback import install

cns = Console()
install(console=cns, show_locals=True)


@click.command()
@click.option('-v','--verbosity', type=int, default=2, show_default=True, help="How much information you want the code to give, between 1 and 4.")
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
    magnification: float,
    lengthscale: int | None,
    zoom_width: int,
    downscale: int | None,
    mog_port: int,
    start_frequency: float,
    end_frequency: float,
    number_frequencies: int
):
    
    doZoom = False
    if zoom_width:
        doZoom = True
    
    doDownscale = False
    if downscale:
        doDownscale = True

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

    myMogDevice = MOGDevice("COM", mog_port)
    myFreq = np.linspace(start_frequency, end_frequency, number_frequencies)
    myAnimX = RFexpImaging(myMogDevice, visParams=myVisParams, verbosity=verbosity, freqRF=myFreq, channel=1)
    myAnimX.rich_print_params()
    myAnimX.run()
    myFreqRF, myIntensity, myPositionx, myPositiony, myWx, myWy, myTheta = myAnimX.get_results()
    np.savez_compressed("AOMx",frequency=myFreqRF, intensity = myIntensity, positionx = myPositionx, positiony = myPositiony, wx = myWx, wy = myWy, theta = myTheta)

    cns.print(f"Results saved in {'AOMx.npz':>5}\nNow proceeding to the second AOM.")


    myAnimY = RFexpImaging(myMogDevice, visParams=myVisParams, verbosity=verbosity, freqRF=myFreq, channel=2)
    myAnimY.rich_print_params()
    myAnimY.run()
    myFreqRF, myIntensity, myPositionx, myPositiony, myWx, myWy, myTheta = myAnimY.get_results()
    np.savez_compressed("AOMy", frequency = myFreqRF, intensity = myIntensity, positionx = myPositionx, positiony = myPositiony, wx = myWx, wy = myWy, theta = myTheta)

    cns.print(f"Results saved in {'AOMy.npz':>5}\n\nEnd of program.")



if __name__=="__main__":
    main()