"""
``directImage``
===============

Providing the CLI tool for direct imaging without fitting of freq modification.

"""

import click
from rich.console import Console
from rich.traceback import install
from src.ThorlabsGaussianTools.realTimeImaging import SimpleImaging, CameraParameters, VisualizationParameters

console = Console()
install(console=console, show_locals=True)


@click.command()
@click.option('-v','--verbosity', type=int, default=2, show_default=True, help="How much information you want the code to give, between 1 and 4.")
@click.option('--exposure-time', type=int, default=1, show_default=True, help="Exposure time of the camera in µs.")
@click.option('--magnification', type=float, default=23., show_default=True, help="The magnification of the imaging system.")
@click.option("--lengthscale", type=int, default=None, show_default=True, help="The desired size for the scalebar.")
@click.option("-z","--zoom-width", type=int, default=100, show_default=True, help="Half width of the zooming window around the point. If None, no zooming is performed.")
def main(
    verbosity: int,
    exposure_time: int,
    magnification: float,
    lengthscale: int | None,
    zoom_width: int
    ):

    assert verbosity>0 and verbosity<5
    do_zoom: bool = False
    if zoom_width:
        do_zoom=True

    console.rule('Real Time Imaging')

    myAnim = SimpleImaging(
        camParams=CameraParameters(exposure_time_us=exposure_time),
        visParams=VisualizationParameters(magnification=magnification, lengthscale_um=lengthscale, zoom_bool=do_zoom, zoom_width=zoom_width),
        console=console,
        verbosity=verbosity
        )

    if verbosity>1: myAnim.rich_print_params()

    myAnim.run()

    console.rule('End of program')


if __name__ == "__main__":

    main()