"""
Program to launch the experiment oof the AOMs linear displacement, using the RealTimeAnimation class from realTimeImaging.

author: Francois Haye
date: 26-06-08
"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from functools import partial
from scipy.ndimage import gaussian_filter
from utils.pygauss import gaussianFitnoPlot
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera
from utils.mogdevice import MOGDevice
from rich.console import Console
from rich.traceback import install
from rich.table import Table
from rich.columns import Columns
try:
    from utils.windows_setup import configure_path
    configure_path("./utils/thorlabs_dlls")
except ImportError:
    configure_path = None

cns = Console()
install(console=cns, show_locals=False)

cns.rule('\nTesting of MogDevice python API')

#%% Connection to XRF device
devXRF = MOGDevice('COM',7)

with cns.status("Initializing the XRF device"):

    time.sleep(1)

    devXRF.cmd('MODE,1,NSB')
    devXRF.cmd('MODE,2,NSB')

    devXRF.cmd('FREQ,1,80MHz')
    devXRF.cmd('FREQ,2,80MHz')

    devXRF.cmd('POWER,1,30dBm')
    devXRF.cmd('POWER,2,30dBm')

    devXRF.cmd('PHASE,1,0')
    devXRF.cmd('PHASE,2,0')

    devXRF.cmd('ON,1')
    devXRF.cmd("ON,2")

    devXRF.cmd('SYNC,on')

    cns.print("All done :v:")

cns.print(f"XRF device status: [bold green]{devXRF.cmd('STATUS')}")

#%% Creating frequencies lines

central_freq = 80   #MHz
bandwidth = 30  #MHz
freqSize = 100

freqX = np.linspace(central_freq-bandwidth, central_freq+bandwidth, freqSize)
freqY = np.linspace(central_freq-bandwidth, central_freq+bandwidth, freqSize)

real_freqX = np.zeros(freqX.shape)
real_freqY = np.zeros(freqY.shape)

xmX, ymX = np.zeros(freqX.shape), np.zeros(freqX.shape)
xmY, ymY = np.zeros(freqY.shape), np.zeros(freqY.shape)

intensityX = np.zeros(freqX.shape)
intensityY = np.zeros(freqY.shape)

sigma1X, sigma2X, thetaX = np.zeros(freqX.shape), np.zeros(freqX.shape), np.zeros(freqX.shape)
sigma1Y, sigma2Y, thetaY = np.zeros(freqY.shape), np.zeros(freqY.shape), np.zeros(freqY.shape)