"""
# Pygauss

Python module to compute gaussian beam fits
"""

import numpy as np
from scipy.optimize import curve_fit

def gaussian2D(xdata: tuple[np.ndarray, np.ndarray], A: float, sigmax: float, sigmay: float, theta: float, x0: float, y0: float, offset: float):
    """
    # gaussian2D
    Return the value of the gaussian centered on (`x0`,`y0`), of amplitude `A` and standard deviation `sigmax` (resp `sigmay`) in the x (resp y) direction, on point `xdata`=(x,y).\\
    The `offset` parameter is because experimentaly the zero is not perfect.
    """

    x, y = xdata
    x0 = float(x0)
    y0 = float(y0)
    a = np.cos(theta)**2/(2*sigmax**2)+np.sin(theta)**2/(2*sigmay**2)
    b = -np.sin(theta)*np.cos(theta)/(2*sigmax**2)+np.cos(theta)*np.sin(theta)/(2*sigmay**2)
    c = np.sin(theta)**2/(2*sigmax**2)+np.cos(theta)**2/(2*sigmay**2)

    arg = a*(x-x0)**2+2*b*(x-x0)*(y-y0)+c*(y-y0)**2
    g = A*np.exp(-arg)+offset
    return g.ravel()

def gaussianFit(Z: np.ndarray, scale: float, sigma_guess: float, x0_guess: float, y0_guess: float, theta0: float):
    """
    # gaussiantFit
    
    ## Parameters
    * `Z`: numpy array of dim 2. The image of the gaussian beam to fit.
    * `scale`: float. The scale of the picture in µm/pixel. Typically l/n with l the size of the picture and n the number of pixels. Beware to add the magnification in case an imaging system is used!
    * `sigma_guess`: The guessed waist of the beam in pixels.
    * `x0_guess`, `y0_guess`: The guessed position of the center of the beam in pixels.

    ## Returns
    * `wx`, `wy`: calculated waists
    * `theta`: calculated angle of the profile
    * `Zsim`: calculated gaussian beam profile
    """

    ### Calculation with curve_fit
    ny, nx = Z.shape
    x, y = np.arange(0,nx,1), np.arange(0,ny,1)
    xv, yv = np.meshgrid(x,y)
    xdata = np.vstack((xv.ravel(), yv.ravel()))
    guess_param = (Z.max(),sigma_guess,sigma_guess,theta0*np.pi/180,x0_guess,y0_guess,10) # (A,sigmax,sigmay,theta,x0,y0,offset)
    popt, pcov = curve_fit(
        gaussian2D,
        xdata,
        Z.ravel(),
        guess_param,
        bounds=([0,0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.pi/4,np.inf,np.inf,np.inf])
        )

    ### Visualizing the gaussian and getting the waist
    Zsim = gaussian2D(xdata,*popt).reshape(xv.shape)
    sigmax, sigmay, theta, x0, y0 = popt[1], popt[2], popt[3]*180/np.pi, popt[4], popt[5]
    wx = 2*scale*sigmax
    wy = 2*scale*sigmay


    return (wx, wy, theta, x0, y0), (xv, yv, Zsim)