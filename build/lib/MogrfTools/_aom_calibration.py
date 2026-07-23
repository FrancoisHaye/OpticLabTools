"""
---------------
aom_calibration
---------------

Gives the functions ``do_calibration`` to run calibrations in the current directory.

"""
#%% Preamble

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import minimize_scalar, root_scalar
from ThorlabsGaussianTools.utils.tl_camera import TLCameraSDK
from ThorlabsGaussianTools.utils.mogdevice import MOGDevice
from ThorlabsGaussianTools.utils.pygauss import gaussianFit
from ThorlabsGaussianTools import CameraParameters, VisualizationGaussianParameters, RFanim
from MogrfTools import Path
import os
import time
import serial
from datetime import datetime
from rich.console import Console
from rich.traceback import install

import matplotlib as mpl
mpl.rcParams=mpl.rcParamsDefault
mpl.rcParams.update({
    "font.family": "sans-serif",
    #"font.sans-serif": ["CMU Sans Serif"],
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.weight": 400.
})
import matplotlib.pyplot as plt

cns = Console()
install(console=cns, show_locals=False)


#%% Constants & Parameters

N = 50
FREQ = np.linspace(50, 110, 2*N)
AMPLITUDES = np.linspace(15, 30, 2*N)
LAM = 100.
WINDOW = [70, 90]
DOWNSCALE_ORDER = 4
GAUSSIAN_SIGMA = 2

CAM_PARAMS = CameraParameters(
    exposure_time_us = 1,
    poll_timeout_ms = 60
)

VIS_PARAMS = VisualizationGaussianParameters(
    fontsize = 12,
    magnification = 23.,
    lengthscale_um = None,
    zoom_bool = False,
    downscale_bool = True,
    downscale_order = DOWNSCALE_ORDER,
    gaussian_filter_sigma = GAUSSIAN_SIGMA,
    gaussian_fitting = True

)

#%% Utilities

def normalize_spline(spline, bracket = None, bounds=None):

    x0 = minimize_scalar(lambda x: -spline(x), bracket=bracket, bounds=bounds).x

    return lambda x: spline(x)/spline(x0)

def invert_spline(func, y, bracket=None):

    sol = root_scalar(
        lambda x: func(x) - y,
        bracket=bracket
    )

    return sol.root

def init_mog_device(mogdevice):

    mogdevice.cmd('FREQ,1,80')
    mogdevice.cmd('FREQ,2,80')
    mogdevice.cmd('POW,1,30')
    mogdevice.cmd('POW,2,30')

def make_annotation(axd: dict[str, mpl.axes.Axes]):
    for label, ax in axd.items():
        ax.annotate(
            label,
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='large', va='top', ha="left", fontfamily='sans-serif',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='white')
        )
#%% Fitting functions

def get_intensities_functions(lam, window):
    """
    Parameters
    ----------
    window : array like, dim = 2, optional
        The frequency window where you need the intensity to be constant, in MHz
        default = [60, 100]
    
    Returns
    -------
    compensate_x, compensate_y : ``function``
        The array function giving A = compensate_x(freq_x) (resp. A = compensate_y(freq_y)) such that the intensity is constant in ``window``.
        
    """

    with np.load('./freq_x.npz') as data:
        interpX = make_smoothing_spline(data['frequencies'], data['intensities'], lam=lam)
        fX = normalize_spline(interpX, bracket=[70, 90])

    with np.load('./freq_y.npz') as data:
        interpY = make_smoothing_spline(data['frequencies'], data['intensities'], lam=lam)
        fY = normalize_spline(interpY, bracket=[70,90])

    with np.load('./amplitudes.npz') as data:
        interp_amp_x = make_smoothing_spline(data['amplitudes'], data['intensities_x'], lam=lam)
        interp_amp_y = make_smoothing_spline(data['amplitudes'], data['intensities_y'], lam=lam)

        gX = lambda x: interp_amp_x(x)/interp_amp_x(30)
        gY = lambda y: interp_amp_y(y)/interp_amp_y(30)

        gXinv = np.vectorize(lambda int_norm: invert_spline(gX, int_norm, bracket=[20,30]))
        gYinv = np.vectorize(lambda int_norm: invert_spline(gY, int_norm, bracket=[20,30]))

    i0x = fX(window).min()
    i0y = fX(window).min()

    compensate_x = lambda frequencies: gXinv(np.clip(i0x/fX(frequencies), 0.3, 1.))
    compensate_y = lambda frequencies: gYinv(np.clip(i0y/fY(frequencies), 0.3, 1.))

    return (compensate_x, compensate_y), (interpX, interpY), (interp_amp_x, interp_amp_y)

def get_position_functions():
    """
    Returns
    -------
    fitX, fitY : ``np.polynomial.Polynomial``
        The 1st order polynomial giving the variation of position with frequency of the resp. AOM.
    
    """

    with np.load("./freq_x.npz") as data:
        fitX = np.polynomial.Polynomial.fit(data['frequencies'], data['position_x'], 1)
    with np.load("./freq_y.npz") as data:
        fitY = np.polynomial.Polynomial.fit(data['frequencies'], data['position_y'], 1)

    return fitX, fitY

def get_angles():
    """
    Returns
    -------
    alpha: ``float``
        Angle between the two lines.

    beta : ``float``
        Angle between the x-axis of the AOMs and the x-axis of the camera.
        
    regX, regY : ``np.polynomial.Polynomial``
        regressions giving y=regX(x) (resp. x=regY(y)) for the AOMs.
        
    """

    with np.load("./freq_x.npz") as data:
        regX = np.polynomial.Polynomial.fit(data['position_x'], data["position_y"], 1)
    with np.load('./freq_y.npz') as data:
        regY = np.polynomial.Polynomial.fit(data['position_y'], data['position_x'], 1)

    vX = np.array([
        [1],
        [regX.convert().coef[1]]
        ])
    vY = np.array([
        [regY.convert().coef[1]], 
        [1]])

    eX = np.array([
        [1],
        [0]
    ])

    alpha = np.arccos(np.vdot(vX, vY) / (np.linalg.norm(vX) * np.linalg.norm(vY)))
    beta = -np.arccos(np.vdot(vX, eX) / np.linalg.norm(vX))

    return float(alpha), float(beta), (regX, regY)

#%% Running functions

def frequency_calibration(n, freq):

    myAnimX = RFanim(
        mogPort = 7,
        freqRF1 = freq,
        powRF1 = 30 * np.ones_like(freq),
        freqRF2 = 80 * np.ones_like(freq),
        powRF2 = 30 * np.ones_like(freq),
        camParams = CAM_PARAMS,
        visParams = VIS_PARAMS,
        console = cns,
        verbosity = 2
    )

    myAnimX.rich_print_params()

    try:
        myAnimX.run()
    except:
        cns.print("End of x measurements")

    f, _, i, x0, y0, wx, wy, theta = myAnimX.get_results()
    x0 = x0 - x0[n]
    y0 = y0 - y0[n]

    np.savez_compressed('./freq_x.npz', frequencies = f, intensities = i, position_x = x0, position_y = y0, waist_x = wx, waist_y = wy, theta = theta)

    myAnimX.mogdevice.close()

    myAnimY = RFanim(
        mogPort = 7,
        freqRF1 = 80 * np.ones_like(freq),
        powRF1 = 30 * np.ones_like(freq),
        freqRF2 = freq,
        powRF2 = 30 * np.ones_like(freq),
        camParams = CAM_PARAMS,
        visParams = VIS_PARAMS,
        console = cns,
        verbosity = 2
    )
    myAnimY.rich_print_params()

    try:
        myAnimY.run()
    except:
        cns.print('End of y measurements')

    _, f, i, x0, y0, wx, wy, theta = myAnimY.get_results()
    x0 = x0 - x0[n]
    y0 = y0 - y0[n]

    np.savez_compressed('./freq_y.npz', frequencies = f, intensities = i, position_x = x0, position_y = y0, waist_x = wx, waist_y = wy, theta = theta)

    myAnimY.mogdevice.close()

def quick_frequency_calibration(n, freq, g=23.):

    intensities_x = np.zeros_like(freq)
    intensities_y = np.zeros_like(freq)
    x0_X, y0_X = np.zeros_like(freq), np.zeros_like(freq)
    x0_Y, y0_Y = np.zeros_like(freq), np.zeros_like(freq)

    try:
        aomDevice = MOGDevice("COM", port=7)
    except serial.SerialException, RuntimeError:
        print("AOM device already connected.")
        
    
    init_mog_device(aomDevice)

    with TLCameraSDK() as sdk:

        cameras = sdk.discover_available_cameras()
        with sdk.open_camera(cameras[0]) as cam:

            cam.frames_per_trigger_zero_for_unlimited = 0
            cam.exposure_time_us = 1
            cam.image_poll_timeout_ms = 60
            cam.is_frame_rate_control_enabled = False
            scale = cam.sensor_pixel_height_um / g

            cam.arm(8)
            cam.issue_software_trigger()

            try:

                for i,f in enumerate(freq):

                    aomDevice.cmd(f'FREQ,1,{f} MHz')
                    time.sleep(1)
                
                    imageCam = cam.get_pending_frame_or_null()
                    if imageCam is not None:
                        image = np.copy(imageCam.image_buffer).reshape(cam.image_height_pixels, cam.image_width_pixels)
                        intensities_x[i] = image.max()
                        y0_X[i], x0_X[i] = np.unravel_index(image.argmax(), image.shape)
                        y0_X[i] *= scale
                        x0_X[i] *= scale
                    else:
                        print(f"unable to acquire image {i}")
                        cam.disarm()
                        exit()

                time.sleep(1)
                init_mog_device(aomDevice)
                time.sleep(1)

                for i,f in enumerate(freq):

                    aomDevice.cmd(f'FREQ,2,{f} MHz')
                    time.sleep(1)
                
                    imageCam = cam.get_pending_frame_or_null()
                    if imageCam is not None:
                        image = np.copy(imageCam.image_buffer).reshape(cam.image_height_pixels, cam.image_width_pixels)
                        intensities_y[i] = image.max()
                        y0_Y[i], x0_Y[i] = np.unravel_index(image.argmax(), image.shape)
                        y0_Y[i] *= scale
                        x0_Y[i] *= scale
                    else:
                        print(f"unable to acquire image {i}")
                        cam.disarm()
                        exit()

            except KeyboardInterrupt:
                pass

            finally:
                x0_X, y0_X = x0_X - x0_X[n], y0_X - y0_X[n]
                x0_Y, y0_Y = x0_Y - x0_Y[n], y0_Y - y0_Y[n]
                np.savez_compressed("./freq_x.npz", frequencies = freq, intensities = intensities_x, position_x = x0_X, position_y = y0_X)
                np.savez_compressed("./freq_y.npz", frequencies = freq, intensities = intensities_y, position_x = x0_Y, position_y = y0_Y)
                init_mog_device(aomDevice)
                aomDevice.close()
                cam.disarm()

def amplitude_calibration(amplitudes):

    intensities_x = np.zeros_like(amplitudes)
    intensities_y = np.zeros_like(amplitudes)

    try:
        aomDevice = MOGDevice("COM", port=7)
    except serial.SerialException, RuntimeError:
        print("AOM device already connected.")
        
    
    init_mog_device(aomDevice)

    with TLCameraSDK() as sdk:

        cameras = sdk.discover_available_cameras()
        with sdk.open_camera(cameras[0]) as cam:

            cam.frames_per_trigger_zero_for_unlimited = 0
            cam.exposure_time_us = 1
            cam.image_poll_timeout_ms = 60
            cam.is_frame_rate_control_enabled = False

            cam.arm(8)
            cam.issue_software_trigger()

            try:

                for i,amp in enumerate(amplitudes):

                    aomDevice.cmd(f'POW,1,{amp} dBm')
                    time.sleep(1)
                
                    imageCam = cam.get_pending_frame_or_null()
                    if imageCam is not None:
                        image = np.copy(imageCam.image_buffer).reshape(cam.image_height_pixels, cam.image_width_pixels)
                        intensities_x[i] = image.max()
                    else:
                        print(f"unable to acquire image {i}")
                        cam.disarm()
                        exit()

                time.sleep(1)
                init_mog_device(aomDevice)
                time.sleep(1)

                for i,amp in enumerate(amplitudes):

                    aomDevice.cmd(f'POW,2,{amp} dBm')
                    time.sleep(1)
                
                    imageCam = cam.get_pending_frame_or_null()
                    if imageCam is not None:
                        image = np.copy(imageCam.image_buffer).reshape(cam.image_height_pixels, cam.image_width_pixels)
                        intensities_y[i] = image.max()
                    else:
                        print(f"unable to acquire image {i}")
                        cam.disarm()
                        exit()

            except KeyboardInterrupt:
                pass

            finally:
                np.savez_compressed("./amplitudes.npz", amplitudes=amplitudes, intensities_x=intensities_x, intensities_y=intensities_y)
                init_mog_device(aomDevice)
                aomDevice.close()
                cam.disarm()

def noise_calibration(duration_s: float = 30, step: float = 0.5):

    t0 = time.time()
    t = []
    i = []

    with TLCameraSDK() as sdk:

        cameras = sdk.discover_available_cameras()
        with sdk.open_camera(cameras[0]) as cam:

            cam.frames_per_trigger_zero_for_unlimited = 0
            cam.exposure_time_us = 1
            cam.image_poll_timeout_ms = 60
            cam.is_frame_rate_control_enabled = False

            cam.arm(8)
            cam.issue_software_trigger()

            try:

                while time.time()-t0 < duration_s:
                    imageCam = cam.get_pending_frame_or_null()
                    if imageCam is not None:
                        image = np.copy(imageCam.image_buffer).reshape(cam.image_height_pixels, cam.image_width_pixels)
                        i.append(image.max())
                        t.append(time.time()-t0)
                        time.sleep(step)
                    else:
                        print(f"unable to acquire image {i}")
                        cam.disarm()
                        exit()

            except:
                ...
            finally:
                i, t = np.array(i), np.array(t)
                np.savez_compressed("./noise.npz", intensity = i, time = t)
                cam.disarm()

def test_calibration(freq, compensate_x: function, compensate_y: function):
    
    intensities_x = np.zeros_like(freq)
    intensities_y = np.zeros_like(freq)
    amplitudes_x = compensate_x(freq)
    amplitudes_y = compensate_y(freq)

    try:
        aomDevice = MOGDevice("COM", port=7)
    except serial.SerialException, RuntimeError:
        print("AOM device already connected.")

    init_mog_device(aomDevice)

    with TLCameraSDK() as sdk:

        cameras = sdk.discover_available_cameras()
        with sdk.open_camera(cameras[0]) as cam:

            cam.frames_per_trigger_zero_for_unlimited = 0
            cam.exposure_time_us = 1
            cam.image_poll_timeout_ms = 60
            cam.is_frame_rate_control_enabled = False

            cam.arm(8)
            cam.issue_software_trigger()

            try:

                for i in range(len(freq)):

                    aomDevice.cmd(f'POW,1,{amplitudes_x[i]} dBm')
                    aomDevice.cmd(f'FREQ,1,{freq[i]} MHz')
                    time.sleep(1)
                
                    imageCam = cam.get_pending_frame_or_null()
                    if imageCam is not None:
                        image = np.copy(imageCam.image_buffer).reshape(cam.image_height_pixels, cam.image_width_pixels)
                        intensities_x[i] = image.max()
                    else:
                        print(f"unable to acquire image {i}")
                        cam.disarm()
                        exit()

                time.sleep(1)
                init_mog_device(aomDevice)
                time.sleep(1)

                for i in range(len(freq)):

                    aomDevice.cmd(f'POW,2,{amplitudes_y[i]} dBm')
                    aomDevice.cmd(f'FREQ,2,{freq[i]} MHz')
                    time.sleep(1)
                
                    imageCam = cam.get_pending_frame_or_null()
                    if imageCam is not None:
                        image = np.copy(imageCam.image_buffer).reshape(cam.image_height_pixels, cam.image_width_pixels)
                        intensities_y[i] = image.max()
                    else:
                        print(f"unable to acquire image {i}")
                        cam.disarm()
                        exit()

            except KeyboardInterrupt:
                pass

            finally:
                np.savez_compressed("./compensation.npz", frequencies=freq, amplitudes_x=amplitudes_x, amplitudes_y=amplitudes_y, intensities_x=intensities_x, intensities_y=intensities_y)
                init_mog_device(aomDevice)
                aomDevice.close()
                cam.disarm()

#%% Plotting functions

def plot_frequency(interpX, interpY, regX, regY, polX, polY, colorx = 'b', colory = 'r'):

    fX = normalize_spline(interpX, bracket=[70, 90])
    fY = normalize_spline(interpY, bracket=[70, 80])
    ### Axes preparation
    fig = plt.figure(figsize=(30,15,"cm"))
    left, right = fig.subfigures(nrows=1, ncols=2, width_ratios=[1,2], wspace=-0.1)
    axl = left.subplot_mosaic(
        [["a", "b"],
        ["c", "."]],
        height_ratios=[2,1],
        width_ratios=[2,1],
        gridspec_kw=dict(hspace=0.05, wspace=0.05)
    )
    axl['b'].tick_params(labelleft=False)
    axl['b'].sharey(axl['a'])
    axl['c'].sharex(axl['a'])
    axl['a'].tick_params(labelbottom=False)

    axr = right.subplot_mosaic(
        [["d", "f"],
        ["e", "f"]],
        sharex=True,
        gridspec_kw=dict(hspace=0.05, wspace=0.05)
    )

    axr["f"].tick_params(left=False, labelleft=False, right=True, labelright=True)
    axr["f"].yaxis.set_label_position('right')

    ### Plotting
    with np.load("./freq_x.npz") as xData:
        freqsim = np.linspace(xData["frequencies"].min()-1, xData["frequencies"].max()+1, 1000)
        axl['a'].plot(xData["position_x"], xData['position_y'], '.', color = colorx) # (c,b) line of AOM c
        axl['c'].plot(xData['position_x'], xData['frequencies'], '.', color = colorx)
        axr["d"].plot(xData['frequencies'], xData['waist_x'], xData['frequencies'], xData['waist_y'], '--', color = colorx)
        axr["e"].plot(xData['frequencies'], xData["theta"], color = colorx)
        axr["f"].plot(xData["frequencies"], xData["intensities"]/interpX(freqsim).max(), color = colorx, label = 'AOM x')

    with np.load("./freq_y.npz") as yData:
        axl['a'].plot(yData['position_x'], yData['position_y'], '.', color = colory) # (c,b) line of AOM b
        axl['b'].plot(yData['frequencies'], yData['position_y'], '.', color = colory)
        axr["d"].plot(yData['frequencies'], yData['waist_x'], yData['frequencies'], yData['waist_y'], '--', color = colory)
        axr["e"].plot(yData["frequencies"], yData["theta"], color = colory)
        axr["f"].plot(yData["frequencies"], yData['intensities']/interpY(freqsim).max(), color = colory, label = 'AOM y')
    
    
    axl["a"].plot(polX(freqsim), regX(polX(freqsim)), 'k-', regY(polY(freqsim)), polY(freqsim), 'k-')
    axl['c'].plot(polX(freqsim), freqsim, 'k-')
    axl['b'].plot(freqsim, polY(freqsim), 'k-')
    axr["f"].plot(freqsim, fX(freqsim), "k-")
    axr["f"].plot(freqsim, fY(freqsim), "k--")

    ### Axes parameters
    axl['a'].set(
        ylabel=r"$\Delta y$ [µm]"
    )
    axl['c'].set(
        xlabel=r"$\Delta x$ [µm]",
        ylabel=r"$f$ [MHz]"
    )
    axl["b"].set(
        xlabel=r"$f$ [MHz]"
    )

    axr["d"].set(
        ylabel=r"$w$ [µm]"
    )
    axr["e"].set(
        xlabel = r"$f$ [MHz]",
        ylabel = r"$\theta$ [°]"
    )
    axr['f'].set(
        xlabel = r"$f$ [MHz]",
        ylabel = r"$I/I_0$"
    )

    make_annotation(axl)
    make_annotation(axr)

    axr['f'].legend()

    fig.savefig('./Frequency.png', dpi=600)
    plt.show()

def plot_simple_frequency(interpX, interpY, regX, regY, polX, polY, colorx = 'b', colory = 'r'):

    fX = normalize_spline(interpX, bracket=[70, 90])
    fY = normalize_spline(interpY, bracket=[70, 80])
    ### Axes preparation
    fig = plt.figure(figsize=(30,15,"cm"))
    left, right = fig.subfigures(nrows=1, ncols=2, width_ratios=[1,2], wspace=-0.1)
    axl = left.subplot_mosaic(
        [["a", "b"],
        ["c", "."]],
        height_ratios=[2,1],
        width_ratios=[2,1],
        gridspec_kw=dict(hspace=0.05, wspace=0.05)
    )
    axl['b'].tick_params(labelleft=False)
    axl['b'].sharey(axl['a'])
    axl['c'].sharex(axl['a'])
    axl['a'].tick_params(labelbottom=False)

    axr = right.subplot_mosaic(
        [["d"]]
    )

    axr["d"].tick_params(left=False, labelleft=False, right=True, labelright=True)
    axr["d"].yaxis.set_label_position('right')

    ### Plotting
    with np.load("./freq_x.npz") as xData:
        freqsim = np.linspace(xData["frequencies"].min()-1, xData["frequencies"].max()+1, 1000)
        axl['a'].plot(xData["position_x"], xData['position_y'], '.', color = colorx) # (c,b) line of AOM c
        axl['c'].plot(xData['position_x'], xData['frequencies'], '.', color = colorx)
        axr["d"].plot(xData["frequencies"], xData["intensities"]/interpX(freqsim).max(), color = colorx, label = 'AOM x')

    with np.load("./freq_y.npz") as yData:
        axl['a'].plot(yData['position_x'], yData['position_y'], '.', color = colory) # (c,b) line of AOM b
        axl['b'].plot(yData['frequencies'], yData['position_y'], '.', color = colory)
        axr["d"].plot(yData["frequencies"], yData['intensities']/interpY(freqsim).max(), color = colory, label = 'AOM y')
    
    
    axl["a"].plot(polX(freqsim), regX(polX(freqsim)), 'k-', regY(polY(freqsim)), polY(freqsim), 'k-')
    axl['c'].plot(polX(freqsim), freqsim, 'k-')
    axl['b'].plot(freqsim, polY(freqsim), 'k-')
    axr["d"].plot(freqsim, fX(freqsim), "k-")
    axr["d"].plot(freqsim, fY(freqsim), "k--")

    ### Axes parameters
    axl['a'].set(
        ylabel=r"$\Delta y$ [µm]"
    )
    axl['c'].set(
        xlabel=r"$\Delta x$ [µm]",
        ylabel=r"$f$ [MHz]"
    )
    axl["b"].set(
        xlabel=r"$f$ [MHz]"
    )
    axr['d'].set(
        xlabel = r"$f$ [MHz]",
        ylabel = r"$I/I_0$"
    )

    make_annotation(axl)
    make_annotation(axr)

    axr['d'].legend()

    fig.savefig('./Frequency.png', dpi=600)
    plt.show()

def plot_test_comp(freq, window, lam, colorx = 'b', colory = 'r'):
    
    fig, ax = plt.subplots()
    with np.load('./compensation.npz') as data:
        spl_x = make_smoothing_spline(data['frequencies'], data['intensities_x'], lam=lam)
        spl_y = make_smoothing_spline(data['frequencies'], data['intensities_y'], lam=lam)

        ax.plot(data['frequencies'], data['intensities_x'] / spl_x(freq).max(), color=colorx, label='AOM x')
        ax.plot(data['frequencies'], data['intensities_y'] / spl_y(freq).max(), color=colory, label='AOM y')
        
        ax.plot(freq, normalize_spline(spl_x, bounds=window)(freq), 'k-')
        ax.plot(freq, normalize_spline(spl_y, bounds=window)(freq), 'k--')

    ax.set(
        xlabel = r"$f$ [MHz]",
        ylabel = r"$I$ [a.u.]"
    )
    ax.legend(loc="best")

    fig.savefig('./Compensation.png', dpi=600)
    plt.show()

def plot_noise():

    fig, ax = plt.subplots()
    with np.load('./noise.npz') as data:
        ax.plot(data['time'], data['intensity']/data['intensity'].mean(), 'b.')
        ax.text(
        .95, .95, f"sigma = {data['intensity'].std()/data['intensity'].mean():.1%}",
        ha = 'right', va = 'top',
        transform = ax.transAxes,
        bbox = dict(boxstyle="round", edgecolor='k', facecolor='white')
        )
        ax.hlines(1, -1, data['time'][-1]+1, colors='k')
        ax.set(
            xlabel = r"$t$ [s]",
            ylabel = r"$I/I_0$",
            ylim=(0.95,1.05),
            xlim=(-1, data['time'][-1]+1)
        )
    
    fig.savefig('./Noise.png', dpi=600)
    plt.show()

#%% Main

def do_calibration(n = N, freq = FREQ, amplitudes = AMPLITUDES, lam = LAM, window = WINDOW, noise_duration: float = 60, noise_step: float = 0.5):
    """
    Launch all AOM calibration experiment and computes the necessary paramters.
    
    Parameters
    ----------
    n : ``int``, optional
        2*n is the number of frequencies and amplitudes used for the calibration

    freq : ``np.ndarray``, optional
        Frequencies to investigate, in MHz. of dim (2*n,).

    amplitudes : ``np.ndarray``, optional
        Amplitudes to investigate, in MHz. of dim (2*n,).

    lam : ``float``, optional
        Lagrange multiplier for the smoothing, a greater lambda means a smoother curve.
        default = 100

    window : ``list[float, float]``
        Frequency window where the intensity must remain constant, in MHz.
        default = [70, 90]

    Returns
    -------
    polX : ``np.polynomial.Polynomial``
        The polynomail giving the x-axis position (brt the AOMs frame) brt the frequency of the x AOM.

    polY : ``np.polynomial.Polynomial``
        The polynomial giving the y-axis position (brt the AOMs frame ) brt the frequency of the y AOM.

    compX : ``function``
        The function giving the compensation of intensity loss by change of amplitude, for any frequency. AmpX = compX(freqX).

    compY : ``function``
        The function giving the compensation of intensity loss by change of amplitude, for any frequency. AmpY = compY(freqY).

    alpha : ``float``
        The angle between the two AOM directions, in rad.

    beta : ``float``
        The angle between the camera x-axis and the AOMs x-axis, in rad.
    
    """
    cns.print('\n\n')
    cns.rule("[bold]Calibration of AOMs", align='left')

    DATE = str(datetime.now().strftime("%y%m%d-%H%M"))
    try:
        os.mkdir(f"./{DATE}")
    except FileExistsError:
        cns.print(f"dir {DATE} already exists.")
    os.chdir(f'./{DATE}')
    
    cns.print("\nThe following steps will execute, please wait until the end of the program.\n")
    cns.print(f"{1:<5d} Calibration of frequencies with gaussian fitting.")
    cns.print(f"{2:<5d} Calibration of amplitudes.")
    cns.print(f"{3:<5d} Noise measurement.")
    cns.print(f"{4:<5d} Testing the quality of the calibration.\n")
    

    with cns.status('frequency calibration'): frequency_calibration(n, freq)
    cns.print(':thumbsup: frequency calibration')
    with cns.status('amplitude calibration'): amplitude_calibration(amplitudes)
    cns.print(':thumbsup: amplitude calibration')
    with cns.status('noise measurement'): noise_calibration(duration_s=noise_duration, step=noise_step)
    cns.print(':thumbsup: noise measurement')
    polX, polY = get_position_functions()
    alpha, beta, (regX, regY) = get_angles()
    (compX, compY), (interpX, interpY), (interp_amp_x, interp_amp_y)  = get_intensities_functions(lam, window)
    plot_frequency(interpX, interpY, regX, regY, polX, polY)
    plot_noise()
    with cns.status('calibration testing'): test_calibration(freq, compX, compY)
    cns.print(':thumbsup: calibration testing')
    plot_test_comp(freq, window, lam)

    os.chdir('../..')

    cns.print('\n\n')
    cns.rule("End of calibration", align='left')
    cns.print('\n\n')

    return polX, polY, compX, compY, alpha, beta

def do_calibration_from_exp(path: str, n = N, freq = FREQ, amplitudes = AMPLITUDES, lam = LAM, window = WINDOW, plot=False):
    """
    Computes the necessary parameters from a calibration experiment already existing.
    
    Parameters
    ----------
    path : ``str``
        The path of the experiment, giving access to all data stored in the path dir.

    n : ``int``, optional
        2*n is the number of frequencies and amplitudes used for the calibration

    freq : ``np.ndarray``, optional
        Frequencies to investigate, in MHz. of dim (2*n,).

    amplitudes : ``np.ndarray``, optional
        Amplitudes to investigate, in MHz. of dim (2*n,).

    lam : ``float``, optional
        Lagrange multiplier for the smoothing, a greater lambda means a smoother curve.
        default = 100

    window : ``list[float, float]``
        Frequency window where the intensity must remain constant, in MHz.
        default = [70, 90]

    Returns
    -------
    polX : ``np.polynomial.Polynomial``
        The polynomail giving the x-axis position (brt the AOMs frame) brt the frequency of the x AOM.

    polY : ``np.polynomial.Polynomial``
        The polynomial giving the y-axis position (brt the AOMs frame ) brt the frequency of the y AOM.

    compX : ``function``
        The function giving the compensation of intensity loss by change of amplitude, for any frequency. AmpX = compX(freqX).

    compY : ``function``
        The function giving the compensation of intensity loss by change of amplitude, for any frequency. AmpY = compY(freqY).

    alpha : ``float``
        The angle between the two AOM directions, in rad.

    beta : ``float``
        The angle between the camera x-axis and the AOMs x-axis, in rad.
    
    """

    cns.print('\n\n')
    cns.rule("[bold]Calibration of AOMs", align='left')

    cns.print("\nThe following steps must have been executed, if not, an error will occur.\n")
    cns.print(f"{1:<5d} Calibration of frequencies with gaussian fitting.")
    cns.print(f"{2:<5d} Calibration of amplitudes.")
    cns.print(f"{3:<5d} Noise measurement.")
    cns.print(f"{4:<5d} Testing the quality of the calibration.\n")

    os.chdir(f'./{path}')

    with cns.status('analysis of data') :
        polX, polY = get_position_functions()
        alpha, beta, (regX, regY) = get_angles()
        (compX, compY), (interpX, interpY), (interp_amp_x, interp_amp_y)  = get_intensities_functions(lam, window)
        if plot: plot_simple_frequency(interpX, interpY, regX, regY, polX, polY)
        if plot: plot_noise()
        if plot: plot_test_comp(freq, window, lam)

    cns.print(':thumbsup: data analysed, calibration ready for usage.')

    os.chdir('..')

    cns.print('\n\n')
    cns.rule("End of calibration", align='left')
    cns.print('\n\n')

    return polX, polY, compX, compY, alpha, beta

def do_complete_calibration_from_exp(path: str, n = N, freq = FREQ, amplitudes = AMPLITUDES, lam = LAM, window = WINDOW, plot=False):
    """
    Computes the necessary parameters from a calibration experiment already existing.
    
    Parameters
    ----------
    path : ``str``
        The path of the experiment, giving access to all data stored in the path dir.

    n : ``int``, optional
        2*n is the number of frequencies and amplitudes used for the calibration

    freq : ``np.ndarray``, optional
        Frequencies to investigate, in MHz. of dim (2*n,).

    amplitudes : ``np.ndarray``, optional
        Amplitudes to investigate, in MHz. of dim (2*n,).

    lam : ``float``, optional
        Lagrange multiplier for the smoothing, a greater lambda means a smoother curve.
        default = 100

    window : ``list[float, float]``
        Frequency window where the intensity must remain constant, in MHz.
        default = [70, 90]

    Returns
    -------
    polX : ``np.polynomial.Polynomial``
        The polynomail giving the x-axis position (brt the AOMs frame) brt the frequency of the x AOM.

    polY : ``np.polynomial.Polynomial``
        The polynomial giving the y-axis position (brt the AOMs frame ) brt the frequency of the y AOM.

    compX : ``function``
        The function giving the compensation of intensity loss by change of amplitude, for any frequency. AmpX = compX(freqX).

    compY : ``function``
        The function giving the compensation of intensity loss by change of amplitude, for any frequency. AmpY = compY(freqY).

    alpha : ``float``
        The angle between the two AOM directions, in rad.

    beta : ``float``
        The angle between the camera x-axis and the AOMs x-axis, in rad.
    
    """

    cns.print('\n\n')
    cns.rule("[bold]Calibration of AOMs", align='left')

    cns.print("\nThe following steps must have been executed, if not, an error will occur.\n")
    cns.print(f"{1:<5d} Calibration of frequencies with gaussian fitting.")
    cns.print(f"{2:<5d} Calibration of amplitudes.")
    cns.print(f"{3:<5d} Noise measurement.")
    cns.print(f"{4:<5d} Testing the quality of the calibration.\n")

    os.chdir(f'./{path}')

    with cns.status('analysis of data') :
        polX, polY = get_position_functions()
        alpha, beta, (regX, regY) = get_angles()
        (compX, compY), (interpX, interpY), (interp_amp_x, interp_amp_y)  = get_intensities_functions(lam, window)
        if plot: plot_frequency(interpX, interpY, regX, regY, polX, polY)
        if plot: plot_noise()
        if plot: plot_test_comp(freq, window, lam)

    cns.print(':thumbsup: data analysed, calibration ready for usage.')

    os.chdir('..')

    cns.print('\n\n')
    cns.rule("End of calibration", align='left')
    cns.print('\n\n')

    return polX, polY, compX, compY, alpha, beta

def do_quick_calibration(n = N, freq = FREQ, amplitudes = AMPLITUDES, lam = LAM, window = WINDOW, plot=False, noise_duration: float = 60, noise_step: float = 0.5):

    cns.print('\n\n')
    cns.rule("[bold]Calibration of AOMs", align='left')

    DATE = str(datetime.now().strftime("%y%m%d-%H%M"))
    try:
        os.mkdir(f"./{DATE}")
    except FileExistsError:
        cns.print(f"dir {DATE} already exists.")
    os.chdir(f'./{DATE}')  

    cns.print("\nThe following steps will execute, please wait until the end of the program.\n")
    cns.print(f"{1:<5d} Calibration of frequencies.")
    cns.print(f"{2:<5d} Calibration of amplitudes.")
    cns.print(f"{3:<5d} Noise measurement.")
    cns.print(f"{4:<5d} Testing the quality of the calibration.\n")
    

    with cns.status('frequency calibration'): quick_frequency_calibration(n, freq)
    cns.print(":thumbsup: frequency calibration")
    with cns.status('amplitude calibration'): amplitude_calibration(amplitudes)
    cns.print(":thumbsup: amplitude calibration")
    with cns.status('noise measurement'): noise_calibration(duration_s=noise_duration, step=noise_step)
    cns.print(":thumbsup: noise measurement")
    polX, polY = get_position_functions()
    alpha, beta, (regX, regY) = get_angles()
    (compX, compY), (interpX, interpY), (interp_amp_x, interp_amp_y)  = get_intensities_functions(lam, window)
    if plot: plot_simple_frequency(interpX, interpY, regX, regY, polX, polY)
    if plot: plot_noise()
    with cns.status("calibration testing"): test_calibration(freq, compX, compY)
    cns.print(":thumbsup: calibration testing\n\n")
    if plot: plot_test_comp(freq, window, lam)

    os.chdir('../..')

    cns.rule("End of calibration", align='left')
    cns.print('\n\n')

    return polX, polY, compX, compY, alpha, beta

