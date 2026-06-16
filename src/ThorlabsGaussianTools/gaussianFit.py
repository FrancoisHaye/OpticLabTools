"""
``gaussianFit``
===============

Providing CLI tool for imaging and gaussian fitting, with various parameters.

"""

import click
from rich.console import Console
from rich.traceback import install
from ._real_time_imaging import GaussianFitImaging, CameraParameters, VisualizationGaussianParameters

console = Console()
install(console=console, show_locals=False)


@click.command()
@click.option('-v','--verbosity', type=int, default=2, show_default=True, help="How much information you want the code to give, between 1 and 4.")
@click.option('--exposure-time', type=int, default=1, show_default=True, help="Exposure time of the camera in µs.")
@click.option('--magnification', type=float, default=23., show_default=True, help="The magnification of the imaging system.")
@click.option("--lengthscale", type=int, default=10, show_default=True, help="The desired size for the scalebar, in µm.")
@click.option("-z","--zoom-width", type=int, default=100, show_default=True, help="Half width of the zooming window around the point. If None, no zooming is performed.")
@click.option("-ds", "--downscale", type=int, default=None, show_default=True, help="Order of downscaling, in pixels. If None, no downscaling.")
@click.option("--gaussian-fit/--no-gaussian-fit", type=bool, default=True, show_default=True, help="Whether to perform a lstsq fit of the gaussian or only a small calculation.")
def main(
    verbosity: int,
    exposure_time: int,
    magnification: float,
    lengthscale: int | None,
    zoom_width: int,
    downscale: int | None,
    gaussian_fit: bool
    ):

    assert verbosity>0 and verbosity<5
    do_zoom: bool = False
    if zoom_width:
        do_zoom=True

    do_downscale: bool = False
    if downscale:
        do_downscale = True

    console.rule('Real Time Imaging')

    myCamParams = CameraParameters(exposure_time_us=exposure_time)
    myVisParams = VisualizationGaussianParameters(
        magnification=magnification,
        lengthscale_um=lengthscale,
        zoom_bool=do_zoom,
        zoom_width=zoom_width,
        downscale_bool=do_downscale,
        downscale_order=downscale,
        gaussian_filter_sigma=3,
        gaussian_fitting=gaussian_fit
    )

    myAnim = GaussianFitImaging(
        camParams=myCamParams,
        visParams=myVisParams,
        console=console,
        verbosity=verbosity
        )

    if verbosity>1: myAnim.rich_print_params()

    myAnim.run()

    console.rule('End of program')


if __name__ == "__main__":

    main()