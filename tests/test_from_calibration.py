import os
os.chdir('./tests')

import numpy as np
from MogrfTools import Path, Line, Circle, MovePoint
from MogrfTools import do_calibration_from_exp

n = 500
freq = np.linspace(40, 120, 2*n)
amp = np.linspace(20, 30, 2*n)
window = [70, 90]

polX, polY, compX, compY, alpha, beta = do_calibration_from_exp("260626-1006",n, freq, amp, lam=2000, window=window)
os.chdir('..')

#myPath = Circle.from_calibration(compX, compY, polX, polY, alpha, beta, R_um=5, x0_um=0, y0_um=0, speed_um_ms=0.01, dt_ms=50) # Caution: if dt*speed is too small, Time
myPath = Line.from_calibration(compX, compY, polX, polY, alpha, beta, theta_deg=10, length_um=20, speed_um_ms=0.001, dt_ms=500, backwards=True)
myPath.plot()

myMove = MovePoint(myPath)

myMove.run(repeat=False)