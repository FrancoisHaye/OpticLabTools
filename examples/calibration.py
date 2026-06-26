"""

Usage of do_calibration functions and Path classes.
subprocess calls the executable ``imaging`` to run the imaging system during the movement of the point.

"""

import os, sys
import numpy as np
import subprocess

from MogrfTools import Path, Line, Circle, MovePoint
from MogrfTools import do_calibration, do_calibration_from_exp, do_quick_calibration

good_dir = False
if os.getcwd().endswith('examples'):
    good_dir = True

n = 250
freq = np.linspace(50, 110, 2*n)
amp = np.linspace(20, 30, 2*n)
window = [70, 90]

if not good_dir: os.chdir('./examples')

# 3 possibilities, uncomment the one you wish to use

#polX, polY, compX, compY, alpha, beta = do_calibration(n, freq, amp, lam=1000, window=window)
#polX, polY, compX, compY, alpha, beta = do_quick_calibration(n, freq, amp, lam=1000, window=window, plot=True)
polX, polY, compX, compY, alpha, beta = do_calibration_from_exp("260626-1006",n, freq, amp, lam=1000, window=window, plot=False)

if not good_dir: os.chdir('..')

myPath = Circle.from_calibration(compX, compY, polX, polY, alpha, beta, R_um=5, x0_um=0, y0_um=0, speed_um_ms=0.01, dt_ms=10)
#myPath.ani_plot()
myMove = MovePoint(myPath)

imaging_cmd = sys.prefix + os.sep + 'Scripts' + os.sep + 'imaging'
subprocess.Popen([imaging_cmd, '--magnification', '23', "--lengthscale", "10", "-z", "0"])

myMove.run(repeat=True)