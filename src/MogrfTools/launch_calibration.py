from ._aom_calibration import do_calibration
import numpy as np
import click

@click.command()
@click.option("-n", "--frequencies-number", type = int, default = 50, show_default=True, help = "Half number of frequencies on which to perform the calibration")
@click.option("-fi", "--f-min", type = float, default = 50, show_default=True, help="Minimal frequency for the calibration.")
@click.option("-ff", "--f-max", type=float, default=110, show_default=True, help="Maximal frequency for the calibration")
@click.option('-ai', "--amplitude-min", type=float, default=5, show_default=True, help="Minimal amplitude for the calibration.")
@click.option("-af", "--amplitude-max", type=float, default=30, show_default=True, help="Maximal amplitude for the calibration.")
@click.option('-l', "--lam", type=float, default=100, show_default=True, help="Lagrande multiplier for smoothing. Greater lambda means smoother curve.")
@click.option('-fw', "--frequency-window", type=list[float, float], default=[70, 90], show_default=True, help="Frequency window where you want the intensity to be constant. Smaller ones gives better results.")
def main(
    n,
    f_min,
    f_max,
    amplitude_min,
    amplitude_max,
    lam,
    frequency_window
    ):

    freq = np.linspace(f_min, f_max, 2*n)
    amplitudes = np.linspace(amplitude_min, amplitude_max, 2*n)

    do_calibration(n, freq, amplitudes, lam, frequency_window)