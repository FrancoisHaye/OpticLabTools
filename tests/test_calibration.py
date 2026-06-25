import os
os.chdir('./tests')

import numpy as np
from MogrfTools import Path
from MogrfTools import do_calibration, do_calibration_from_exp

n = 50
freq = np.linspace(65, 95, 2*n)
amp = np.linspace(25, 30, 2*n)
window = [70, 90]

# The path I want to draw: a straigth line in the x direction, between 0 and 5 µm.
xpos = np.linspace(0, 5, 100)
ypos = np.zeros_like(xpos)
time_ms = np.linspace(1, 1000, 100)

polX, polY, compX, compY, alpha, beta = do_calibration(n, freq, amp, lam=100, window=window)
os.chdir('..')

myPath = Path.from_calibration(compX, compY, polX, polY, alpha, beta, xpos, ypos, time_ms)

myPath.ani_plot()