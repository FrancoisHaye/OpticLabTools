"""
``utils.pygauss``
=================


Python module to compute gaussian beam fits.

Available functions
---------
gaussian2D
    Two dimensional gaussian distribution.
gaussianFit
    Two dimensional gaussian fitting by least square method.
gaussianCompute
    Computes the parameters of a 2D gaussian by definition of average and standard deviation of a probability distribution.

"""

import numpy as np
from scipy.optimize import curve_fit

def gaussian2D(xdata: tuple[np.ndarray, np.ndarray], x0: float, y0: float, sigmax: float, sigmay: float, theta: float, amplitude: float, offset: float):
    """
    Return the value of the gaussian centered on (`x0`,`y0`), of amplitude `A` and standard deviation `sigmax` (resp `sigmay`) in the x (resp y) direction, on point `xdata`=(x,y).

    The `offset` parameter is because experimentaly the zero is not perfect.

    """

    x, y = xdata
    x0 = float(x0)
    y0 = float(y0)
    a = np.cos(theta)**2/(2*sigmax**2)+np.sin(theta)**2/(2*sigmay**2)
    b = -np.sin(theta)*np.cos(theta)/(2*sigmax**2)+np.cos(theta)*np.sin(theta)/(2*sigmay**2)
    c = np.sin(theta)**2/(2*sigmax**2)+np.cos(theta)**2/(2*sigmay**2)

    arg = a*(x-x0)**2+2*b*(x-x0)*(y-y0)+c*(y-y0)**2
    g = amplitude*np.exp(-arg)+offset
    return g.ravel()

def gaussianFit(Z: np.ndarray, x0_guess: float, y0_guess: float, sigmax_guess: float, sigmay_guess: float, theta_guess: float):
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
    guess_params = (x0_guess,y0_guess,sigmax_guess,sigmay_guess,theta_guess*np.pi/180,Z.max(),Z.min()) # (x0,y0,sigmax,sigmay,theta,amplitude,offset)
    popt, pcov = curve_fit(
        f = gaussian2D,
        xdata = xdata,
        ydata = Z.ravel(),
        p0 = guess_params,
        bounds = ([0,0,0,0,0,0,0],[nx, ny, nx, ny, np.pi/4, 2*Z.max(), 2*Z.min()])
        )

    ### Visualizing the gaussian and getting the waist
    Zsim = gaussian2D(xdata,*popt).reshape(xv.shape)
    x0, y0, sigmax, sigmay = popt[:-3]
    theta = popt[-3]*180/np.pi

    return (x0, y0, sigmax, sigmay, theta), (xv, yv, Zsim)

def gaussianCompute(Z: np.ndarray):
    """
    Calculates the position and stdev of a gaussian distribution, without fitting.
    
    Parameters
    ----------
    Z : ``np.ndarray``
        The 2D array containing the gaussian beam.
    
    Returns
    -------
    params : ``tuple[float, float, float, float]``
        The parameters of the gaussian in the following order: x0, y0, sigmax, sigmay, theta

    Raises
    ------
    AssertionError
        if the array dimension is not 2

    """
    
    assert len(Z.shape) == 2

    Zx = Z.sum(axis=0)
    Zy = Z.sum(axis=1)

    Zx = (Zx - Zx.min()) / Zx.sum()
    Zy = (Zy - Zy.min()) / Zx.sum()

    x = np.arange(0, *Zx.shape, 1)
    y = np.arange(0, *Zy.shape, 1)

    x0 = np.average(x, weights=Zx)
    y0 = np.average(y, weights=Zy)
    sigmax = np.sqrt(np.average((x-x0)**2, weights=Zx))
    sigmay = np.sqrt(np.average((y-y0)**2, weights=Zy))

    params = (float(x0), float(y0), float(sigmax), float(sigmay), 0.)

    ny, nx = Z.shape
    x, y = np.arange(0,nx,1), np.arange(0,ny,1)
    xv, yv = np.meshgrid(x,y)
    xdata = np.vstack((xv.ravel(), yv.ravel()))
    Zsim = gaussian2D(xdata,*params, Z.max(), Z.min()).reshape(xv.shape)

    return params, (xv, yv, Zsim)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from rich.console import Console
    from rich.table import Table, Column
    from rich.traceback import install

    cns = Console()
    install(console=cns, show_locals=False)

    inputs = (60, 50, 15, 5, 0, 100, 10)

    x = np.arange(0,150,1)
    y = np.arange(0,150,1)

    xmesh, ymesh = np.meshgrid(x, y)
    xdata = np.vstack((xmesh.ravel(), ymesh.ravel()))

    distribution = gaussian2D(xdata,*inputs).reshape(xmesh.shape)
    paramsComp, simComp = gaussianCompute(distribution)
    paramsFit, simFit = gaussianFit(distribution, *inputs[:-2])

    tab = Table("method",Column("x0", justify='right', highlight=True), Column("y0", justify='right', highlight=True), Column("sigma_x", justify='right', highlight=True), Column("sigma_y", justify='right', highlight=True), caption="Results of gaussian fitting", title_justify='center', highlight=True, show_edge=True)
    tab.add_row('input', *map(lambda x: f"{x:.1f}", inputs[:-3]))
    tab.add_row('gaussianFit', *map(lambda x: f"{x:.1f}", paramsFit[:-1]))
    tab.add_row('gaussian_compute', *map(lambda x: f"{x:.1f}", paramsComp[:-1]))

    cns.print(tab, justify='center')

    gs = GridSpec(2,2,width_ratios=[1,3], height_ratios=[3,1])
    fig = plt.figure(layout='tight')
    axIm = plt.subplot(gs[1])
    plt.imshow(distribution)
    plt.contour(*simFit, colors="white", levels=5)
    plt.contour(*simComp, colors="red", levels=5)
    plt.axis('off')
    axY = plt.subplot(gs[0], sharey=axIm)
    plt.plot(distribution.sum(axis=1), y)
    plt.plot(simFit[-1].sum(axis=1), y, '--')
    axX = plt.subplot(gs[3],sharex=axIm)
    plt.plot(x, distribution.sum(axis=0))
    plt.plot(x, simFit[-1].sum(axis=0), '--')
    plt.show()