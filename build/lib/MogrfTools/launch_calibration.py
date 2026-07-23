"""
CLI tool to launch calibration experiment.

"""

from ._aom_calibration import do_calibration, do_quick_calibration
import numpy as np
import click

@click.command()
@click.option("-n", "--frequencies-number", type = int, default = 50, show_default=True, help = "Half number of frequencies on which to perform the calibration")
@click.option("-fi", "--f-min", type = float, default = 50, show_default=True, help="Minimal frequency for the calibration.")
@click.option("-ff", "--f-max", type=float, default=110, show_default=True, help="Maximal frequency for the calibration")
@click.option('-ai', "--amplitude-min", type=float, default=5, show_default=True, help="Minimal amplitude for the calibration.")
@click.option("-af", "--amplitude-max", type=float, default=30, show_default=True, help="Maximal amplitude for the calibration.")
@click.option('-l', "--lam", type=float, default=100, show_default=True, help="Lagrande multiplier for smoothing. Greater lambda means smoother curve.")
@click.option('-fwi', "--fw-min", type=float, default=70, show_default=True, help="Minimal frequency for the optimization window.")
@click.option('-fwf', "--fw-max", type=float, default=90, show_default=True, help="Maximal frequency for the optimization window.")
@click.option('--fit/--no-fit', type=bool, default=True, show_default=True, help="Whether to perform a gaussian fit for more precise results.")
@click.option('-nd', '--noise-duration', type=float, default=60, show_default=True, help='Duration of the noise measurement, in s.')
@click.option('-ns', "--noise-step", type=float, default=0.5, show_default=True, help='Step for measurement of noise, in s.')
def main(
    frequencies_number,
    f_min,
    f_max,
    amplitude_min,
    amplitude_max,
    lam,
    fw_min,
    fw_max,
    fit,
    noise_duration,
    noise_step
    ):

    frequency_window = [fw_min, fw_max]
    freq = np.linspace(f_min, f_max, 2*frequencies_number)
    amplitudes = np.linspace(amplitude_min, amplitude_max, 2*frequencies_number)

    if fit:
        do_calibration(frequencies_number, freq, amplitudes, lam, frequency_window, noise_duration, noise_step)
    else:
        do_quick_calibration(frequencies_number, freq, amplitudes, lam, frequency_window, plot=True, noise_duration=noise_duration, noise_step=noise_step)