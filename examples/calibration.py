"""

Usage of do_calibration functions and Path classes.
subprocess calls the executable ``imaging`` to run the imaging system during the movement of the point.

"""

import os, sys
import numpy as np
import subprocess

from MogrfTools import Path, Line, Circle, Lissajous, MovePoint
from MogrfTools import do_calibration, do_calibration_from_exp, do_quick_calibration, do_complete_calibration_from_exp
from rich.console import Console
cns = Console()

good_dir = False
if os.getcwd().endswith('examples'):
    good_dir = True

n = 250
freq = np.linspace(50, 110, 2*n)
amp = np.linspace(20, 30, 2*n)
window = [70, 90]

if not good_dir: os.chdir('./examples')

# 4 possibilities, uncomment the one you wish to use

#polX, polY, compX, compY, alpha, beta = do_calibration(n, freq, amp, lam=1000, window=window)
#polX, polY, compX, compY, alpha, beta = do_quick_calibration(n, freq, amp, lam=1000, window=window, plot=True)
#polX, polY, compX, compY, alpha, beta = do_calibration_from_exp("260626-1006",n, freq, amp, lam=1000, window=window, plot=False)
polX, polY, compX, compY, alpha, beta = do_complete_calibration_from_exp("260626-1006",n, freq, amp, lam=1000, window=window, plot=False)

cns.rule("Parameters", align='left')
cns.print('\n')
cns.print(f"alpha = {np.rad2deg(alpha):>10.2f}°")
cns.print(f"beta =  {np.rad2deg(beta):>10.2f}°")
cns.print('\n')
cns.print(f"x = {polX.convert().coef[1]:.2f} * f + {polX.convert().coef[0]:.2f}")
cns.print(f"y = {polY.convert().coef[1]:.2f} * f + {polY.convert().coef[0]:.2f}")
cns.print('\n')
cns.print(f"a_x =   {polX.convert().coef[1]:>10.2f} µm/MHz")
cns.print(f"a_y =   {polY.convert().coef[1]:>10.2f} µm/MHz")
cns.print(f"f0_x =  {-polX.convert().coef[0]/polX.convert().coef[1]:>10.1f} MHz")
cns.print(f"f0_y =  {-polY.convert().coef[0]/polY.convert().coef[1]:>10.1f} MHz")
cns.print("\n\n")

if not good_dir: os.chdir('..') 

myPath = Lissajous.from_calibration(compX, compY, polX, polY, alpha, beta, Lx_um=30, Ly_um=20, delta_rad=np.pi/2, speed_ratio=4, x0_um=0, y0_um=0, duration_ms=5000, n=500)
#myPath.ani_plot()
myMove = MovePoint(myPath)

#imaging_cmd = sys.prefix + os.sep + 'Scripts' + os.sep + 'imaging'
#subprocess.Popen([imaging_cmd, '--magnification', '23', "--lengthscale", "10", "-z", "0"])

myMove.run(repeat=True, pause_ms=1)