import os
os.chdir('./tests')

import numpy as np
from MogrfTools import Path, Line, Circle, MovePoint
from MogrfTools import do_calibration, do_calibration_from_exp

n = 500
freq = np.linspace(40, 120, 2*n)
amp = np.linspace(20, 30, 2*n)
window = [70, 90]

# The path I want to draw: a straigth line in the x direction, between -5 and 5 µm.
xpos = np.linspace(-5, 5, 100)
ypos = np.zeros_like(xpos)
time_ms = np.linspace(1, 1000, 100)

polX, polY, compX, compY, alpha, beta = do_calibration(n, freq, amp, lam=1000, window=window)
os.chdir('..')

myPath = Circle.from_calibration(compX, compY, polX, polY, alpha, beta, R_um=10, x0_um=0, y0_um=0, speed_um_ms=0.1, dt_ms=1)
myPath.ani_plot()
myMove = MovePoint(myPath)

myMove.run(repeat=True)