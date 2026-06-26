import os
import numpy as np
from MogrfTools import Circle, MovePoint
from MogrfTools import do_quick_calibration, do_calibration_from_exp

n = 500
freq = np.linspace(50, 110, 2*n)
amp = np.linspace(20, 30, 2*n)
window = [70, 90]

os.chdir('./tests')
#polX, polY, compX, compY, alpha, beta = do_quick_calibration(n, freq, amp, lam=2000, window=window, plot=True)
polX, polY, compX, compY, alpha, beta = do_calibration_from_exp("260626-1312", n, freq, amp, lam=2000, window=window, plot=True)
os.chdir('..')

myPath = Circle.from_calibration(compX, compY, polX, polY, alpha, beta, R_um=5, x0_um=0, y0_um=0, speed_um_ms=-0.05, dt_ms=5) # Caution: if dt*speed is too small, TimeOut Error from MogRF and need to disconnect and reconnect.

#myPath = Line.from_calibration(compX, compY, polX, polY, alpha, beta, theta_deg=10, length_um=20, speed_um_ms=0.001, dt_ms=500, backwards=False)
#myPath.ani_plot()

myMove = MovePoint(myPath)

myMove.run(repeat=True, pause_ms=10) # Look on ThorImageCam to see the point moving.