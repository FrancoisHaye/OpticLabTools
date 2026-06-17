"""Coding UTF-8"""

#%% Introduction - imports and configuration

# classical imports
from dataclasses import dataclass
from functools import partial
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# rich terminal configuration
from rich.console import Console
from rich.traceback import install
from rich.table import Table
from rich.columns import Columns

console = Console(soft_wrap=True)
install(console=console, show_locals=True)

# Personnal library imports (from utils)
from .utils.pygauss import gaussianFit, gaussianCompute
from .utils.tl_camera import TLCamera, TLCameraSDK
try:
    from .utils import configure_path
    configure_path("./thorlabs_dlls")
except ImportError:
    configure_path = None
from .utils.mogdevice import MOGDevice


#%% Exception classes

class CameraException(Exception): ...
class ImageException(Exception): ...

#%% Parameter classes

@dataclass()
class CameraParameters:
    """
    ``dataclass`` representing the parameters to give to the TLCamera instance.

    Parameters
    ----------
    exposure_time_us : ``int``
        The exposure time of the camera in µs.
    
    poll_timeout_ms : ``int``
        The time an image is kept in the camera buffer. For more intel see utils.tlcamera.TLCamera
    
    frame_rate_target : ``int``
        The frame per second that we desire the camera to work at. Usefull only if ``frame_rate_control_enabled`` is ``True``.

    frame_rate_control_enabled : ``bool``
        Whether to control the camera imaging speed by a desired framerate.
    
    """

    exposure_time_us: int = 1
    poll_timeout_ms: int = 60
    frame_rate_target: int | None = 30
    frame_rate_control_enabled: bool = False

    def rich_print(self, console: Console):
        """
        Pretty display of the parameters in a rich ``Console`` object.
        
        Parameters
        ----------
        console : ``rich.console.Console``
            The console object on which to print the parameters
        
        """

        console.print("\n[bold]Camera parameters :camera:\n")
        console.print(f"Exposure time:                         {self.exposure_time_us:>10d} µs")
        console.print(f"Poll timeout:                          {self.poll_timeout_ms:>10d} ms")
        console.print(f"The camera is controlled by framerate: {self.frame_rate_control_enabled!s:>13}")
        if self.frame_rate_control_enabled:
            console.print(f"Framerate asked:                       {self.frame_rate_target:>9d} fps")
        console.print("\n\n")

@dataclass()
class VisualizationParameters:
    """
    ``dataclass`` representing the parameters for the matplotlib visualization of the images.
    
    Parameters
    ----------
    fontsize : ``int``, optional
        The fontsize for the plot.
        default = 12
        
    magnification : ``float``, optional
        The magnification of the imaging system used before the camera. This is used to compute a scalebar on the plot.
        default = 1.

    lengthscale_um : ``int`` or ``None``, optional
        The length of the scalebar plotted on the image in µm.
        default = None

    zoom_bool : ``bool``, optional
        Whether to zoom around the gaussian beam.
        default = False

    zoom_width : ``int``, optional
        The half-width of the zoom in pixels. The final image will be centered on the gaussian beam, with a width of 2*zoom_width.
        default = 100
    
    """
    
    fontsize: int = 12
    magnification: float = 1.
    lengthscale_um: int | None = None
    zoom_bool: bool = False
    zoom_width: int | None = 100


    def rich_print(self, console: Console):
        """
        Pretty display of the parameters in a rich ``Console`` object.
        
        Parameters
        ----------
        console : ``rich.console.Console``
            The console object on which to print the parameters
        
        """

        console.print("\n[bold]Visualization parameters :framed_picture:\n")
        console.print(f"Fontsize:                              {self.fontsize:>10d} pt")
        console.print(f"Magnification:                         {self.magnification:>11.2g} x")
        if self.lengthscale_um:
            console.print(f"Lengthscale:                           {self.lengthscale_um:>10d} µm")
        console.print(f"Do zoom?                               {self.zoom_bool!s:>13}")
        if self.zoom_bool:
            console.print(f"Zoom width:                            {self.zoom_width:>10d} px")
        console.print("\n\n")

@dataclass
class VisualizationGaussianParameters(VisualizationParameters):
    """
    subclass of ``VisualizationParameters`` taking into account gaussian fitting parameters.
    
    Parameters
    ----------
    fontsize : ``int``, optional
        The fontsize for the plot.
        default = 12
        
    magnification : ``float``, optional
        The magnification of the imaging system used before the camera. This is used to compute a scalebar on the plot.
        default = 1.

    lengthscale_um : ``int`` or ``None``, optional
        The length of the scalebar plotted on the image in µm.
        default = None

    zoom_bool : ``bool``, optional
        Whether to zoom around the gaussian beam.
        default = True

    zoom_width : ``int``, optional
        The half-width of the zoom in pixels. The final image will be centered on the gaussian beam, with a width of 2*zoom_width.
        default = 100

    downscale_bool : ``bool``, optional
        Whether to perform a downscaling step of the image to increase performances at the cost of fitting precision.
        default = False

    downscale_order : ``int``, optional
        The order of the downscaling, in pixels. This corresponds to the number of pixels _skipped_ for one final pixel (e.g. 1 pixel over 3 for an order of 3).
        default = 5

    gaussian_filter_sigma : ``int``, optional
        The stdev of the gaussian for the filter before downscaling, in pixels. Filtering is important for antialiasing when downscaling, but a value too large results in overestimation of the waist of the beam.
        default = 3

    gaussian_fitting : ``bool``, optional
        Whether to perform a leastsquare fit of the gaussian beam with scipy.optimize.curve_fit or only a statistic calculation on the distribution. The latter allows better performances at the cost of real bad fitting values. For more information see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
        default = False
    
    """

    downscale_bool: bool = False
    downscale_order: int = 5
    gaussian_filter_sigma: int = 3
    gaussian_fitting: bool = False

    def rich_print(self, console: Console):

        console.print("\n[bold]Visualization parameters :framed_picture:\n")
        console.print(f"Fontsize:                              {self.fontsize:>10d} pt")
        console.print(f"Magnification:                         {self.magnification:>11.2g} x")
        if self.lengthscale_um: 
            console.print(f"Lengthscale:                           {self.lengthscale_um:>10d} µm")
        console.print(f"Do zoom?                               {self.zoom_bool!s:>13}")
        if self.zoom_bool:
            console.print(f"Zoom width:                            {self.zoom_width:>10d} px")
        console.print(f"Do downscaling?                        {self.downscale_bool!s:>13}")
        if self.downscale_bool:
            console.print(f"Downscaling order:                     {self.downscale_order:>10d} px")
        console.print(f"Gaussian filter stdev:                 {self.gaussian_filter_sigma:>10d} px")
        console.print(f"Curve fitting with lstsquare?          {self.gaussian_fitting!s:>13}")
        console.print("\n\n")


#%% Imaging classes

class RealTimeImaging:
    """
    An imaging in real time abstract class for thorlabs camera, using tl_camera_SDK.
    
    Parameters
    ----------
    camParams : ``CameraParameters``, optional
        Parameters of the camera acquisition for the imaging.
        default = CameraParameters()

    visParams : ``VisualizationParameters``, optional
        Parameters of the visualization (matplotlib) for the imaging.
        default = VisualizationParameters()

    console : ``rich.console.Console``, optional
        Console on which to print all informations.
        default = Console()
    
    verbosity : ``int``, optional
        Verbosity of the program, between 1 and 4.
        default = 2

    """

    def __init__(
            self,
            camParams: CameraParameters = CameraParameters(),
            visParams: VisualizationParameters = VisualizationParameters(),
            console: Console = Console(),
            verbosity: int = 2
    ):
        
        self.camParams = camParams
        self.visParams = visParams
        self.cns: Console = console
        self.verbosity = verbosity

        self.fig, self.ax = plt.subplots()

        self.im: mpl.image.AxesImage | None = None
        self.scalebar: AnchoredSizeBar | None = None
        self.ani: FuncAnimation | None = None
        self.sdk: TLCameraSDK | None = None
        self.cam: TLCamera | None = None
        self.scale: float = 1.

        self.fontprops = mpl.font_manager.FontProperties(size=self.visParams.fontsize)
        self.max_frames: int = 0

    def rich_print_params(self):
        """Pretty prints the parameters of the imaging."""
        self.camParams.rich_print(self.cns)
        self.visParams.rich_print(self.cns)
    
    def _init_function(self): ...

    def _update_function(self, frame): ...

    def run(self, number_of_frames: int = 10000, interval_ms: int = 20):
        """
        Launches the animation in real time.
        
        Parameters
        ----------
        
        number_of_frames: ``int``, optional
            The number of frames to render. A low value will result in a very short imaging.
            default = 10000

        interval_ms : ``int``, optional
            The interval between each call of the animation function, in ms. Performances can be impacted by other processes, resulting in a fps drop with respect to the expected value.
            default = 20
        
        """

        self.max_frames = number_of_frames

        with TLCameraSDK() as self.sdk:
            available_cameras = self.sdk.discover_available_cameras()
            
            if not len(available_cameras):
                raise(CameraException("No cameras found."))
            
            with self.sdk.open_camera(available_cameras[0]) as self.cam:

                self.cam.exposure_time_us = self.camParams.exposure_time_us
                self.cam.frames_per_trigger_zero_for_unlimited = 0
                self.cam.image_poll_timeout_ms = self.camParams.poll_timeout_ms
                self.cam.is_frame_rate_control_enabled = self.camParams.frame_rate_control_enabled
                self.cam.frame_rate_control_value = self.camParams.frame_rate_target

                self.scale = self.scale * self.cam.sensor_pixel_height_um / self.visParams.magnification

                self.cam.arm(8)
                self.cam.issue_software_trigger()

                with self.cns.status(f"Imaging system running"):
                    self._init_function()           # drawing first frame of the animation
                    self.ani = FuncAnimation(
                        fig = self.fig,
                        func = self._update_function,
                        frames = number_of_frames,
                        interval = interval_ms,
                        repeat = False,
                        blit = False
                    )
                    plt.show()
                    plt.close()
            
                self.cam.disarm()

class SimpleImaging(RealTimeImaging):
    """
    ``RealTimeImaging`` subclass providing only the image and an eventual scalebar.
    
    Parameters
    ----------
    camParams : ``CameraParameters``, optional
        Parameters of the camera acquisition for the imaging.
        default = CameraParameters()

    visParams : ``VisualizationParameters``, optional
        Parameters of the visualization (matplotlib) for the imaging.
        default = VisualizationParameters()

    console : ``rich.console.Console``, optional
        Console on which to print all informations.
        default = Console()
    
    verbosity : ``int``, optional
        Verbosity of the program, between 1 and 4.
        default = 2

    """
    
    def _init_function(self):

        image_cam = self.cam.get_pending_frame_or_null()

        if image_cam is None: 
            raise(ImageException("Unable to acquire first image"))
        
        image_buffer = np.copy(image_cam.image_buffer)
        shaped_image = image_buffer.reshape(self.cam.image_height_pixels, self.cam.image_width_pixels)

        if self.visParams.zoom_bool:
            i, j = np.unravel_index(shaped_image.argmax(), shaped_image.shape)
            shaped_image = shaped_image[i-self.visParams.zoom_width : i+self.visParams.zoom_width, 
                                        j-self.visParams.zoom_width : j+self.visParams.zoom_width]

        self.im = self.ax.imshow(shaped_image)

        if self.visParams.lengthscale_um:

            lengthScale = self.visParams.lengthscale_um / self.scale

            self.scalebar = AnchoredSizeBar(
                self.ax.transData,
                lengthScale,
                f'{lengthScale*self.scale:.0f} µm',
                'lower right',
                pad=1, sep=6,
                color='white',frameon=False,
                size_vertical=lengthScale/100,
                fontproperties=self.fontprops
            )

            self.scalebar = self.ax.add_artist(self.scalebar)
            self.ax.axis('off')

    def _update_function(self, frame):

        if frame == self.max_frames - 1 :
            self.ani.event_source.stop()
            plt.close()
        
        t0 = time.time()
        image_cam = self.cam.get_pending_frame_or_null()
        poll_delay = time.time() - t0

        if image_cam is None:
            raise(ImageException(f"Frame {frame} not received"))
        
        t0 = time.time()
        image_buffer = np.copy(image_cam.image_buffer)
        shaped_image = image_buffer.reshape(self.cam.image_height_pixels, self.cam.image_width_pixels)
        if self.visParams.zoom_bool:
            i, j = np.unravel_index(shaped_image.argmax(), shaped_image.shape)
            shaped_image = shaped_image[i-self.visParams.zoom_width : i+self.visParams.zoom_width, 
                                        j-self.visParams.zoom_width : j+self.visParams.zoom_width]
        treat_delay = time.time() - t0

        t0 = time.time()
        self.im.set_data(shaped_image)
        plot_delay = time.time() - t0

        if self.verbosity > 2:
            col = Columns([f"polling: {poll_delay*1e3:>5.2f} ms", f"treating: {treat_delay*1e3:>5.2f} ms", f"plotting: {plot_delay*1e3:>5.2f} ms"])
            self.cns.print(col)

        if self.visParams.lengthscale_um:
            return self.im, self.scalebar,
        else:
            return self.im,

class GaussianFitImaging(RealTimeImaging):
    """
    ``RealTimeImaging`` subclass providing the image with a fitted 2D gaussian with the form of a textbox containing necessary informations, and a contour plot of the gaussian for visual check.
    
    Parameters
    ----------
    camParams : ``CameraParameters``, optional
        Parameters of the camera acquisition for the imaging.
        default = CameraParameters()

    visParams : ``VisualizationParameters``, optional
        Parameters of the visualization (matplotlib) for the imaging.
        default = VisualizationGaussianParameters()

    console : ``rich.console.Console``, optional
        Console on which to print all informations.
        default = Console()
    
    verbosity : ``int``, optional
        Verbosity of the program, between 1 and 4.
        default = 2
        
    """

    def __init__(
            self,
            camParams: CameraParameters = CameraParameters(),
            visParams: VisualizationParameters = VisualizationGaussianParameters(),
            console: Console = Console(),
            verbosity: int = 2
    ):

        super().__init__(camParams=camParams, visParams=visParams, console=console, verbosity=verbosity)
        
        if self.visParams.downscale_bool:
            self.scale = self.scale * self.visParams.downscale_order
        
        # Fitting parameters
        self.x0: float = 0.
        self.y0: float = 0.
        self.sigmax: float = 0.
        self.sigmay: float = 0.
        self.theta: float = 0.
        self.text: str = ""

        # Plot 
        self.contour: mpl.contour.QuadContourSet | None = None
        self.textbox: mpl.text.Text | None = None

    def _make_parameters_text(self):

        text = r"$w_x$" + f" = {2*self.sigmax*self.scale:>1.1f} µm\n"
        text+= r"$w_y$" + f" = {2*self.sigmay*self.scale:>1.1f} µm\n"
        text+= r"$w_0$" + f" = {(self.sigmax+self.sigmay)*self.scale:>1.1f} µm\n"
        text+= r"$\theta$  " + f" = {self.theta:>6.1f} °"

        self.text = text

    def _init_function(self):

        image_cam = self.cam.get_pending_frame_or_null()

        if image_cam is None: 
            raise ImageException("Unable to acquire first image")
        
        image_buffer = np.copy(image_cam.image_buffer)
        shaped_image = image_buffer.reshape(self.cam.image_height_pixels, self.cam.image_width_pixels)

        if self.visParams.zoom_bool:
            i, j = np.unravel_index(shaped_image.argmax(), shaped_image.shape)
            shaped_image = shaped_image[i-self.visParams.zoom_width : i+self.visParams.zoom_width, 
                                        j-self.visParams.zoom_width : j+self.visParams.zoom_width]
            
        self.im = self.ax.imshow(shaped_image)
            
        if self.visParams.downscale_bool:
            shaped_image = gaussian_filter(shaped_image, self.visParams.gaussian_filter_sigma)
            shaped_image = shaped_image[:: self.visParams.downscale_order, :: self.visParams.downscale_order]

        # First gaussian calculation
        popt, sim = gaussianCompute(shaped_image)
        self.x0, self.y0, self.sigmax, self.sigmay, self.theta = popt

        if self.visParams.gaussian_fitting:
            popt, sim = gaussianFit(shaped_image, self.x0, self.y0, self.sigmax, self.sigmay, self.theta)
            self.x0, self.y0, self.sigmax, self.sigmay, self.theta = popt

        self._make_parameters_text()

        if self.visParams.downscale_bool:
            self.contour = self.ax.contour(sim[0]*self.visParams.downscale_order, sim[1]*self.visParams.downscale_order, sim[2], levels=5, colors='white')
        else:
            self.contour = self.ax.contour(*sim, levels=5, colors='white')

        self.textbox = self.ax.text(.95,.95, self.text, transform=self.ax.transAxes, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='k', boxstyle='Round'), fontsize=self.visParams.fontsize)


        if self.visParams.lengthscale_um:
            
            lengthScale = self.visParams.lengthscale_um / self.scale

            if self.visParams.downscale_bool:
                lengthScale *= self.visParams.downscale_order

            self.scalebar = AnchoredSizeBar(
                self.ax.transData,
                lengthScale,
                f'{self.visParams.lengthscale_um:.0f} µm',
                'lower right',
                pad=1, sep=6,
                color='white',frameon=False,
                size_vertical=lengthScale/100,
                fontproperties=self.fontprops
            )

            self.scalebar = self.ax.add_artist(self.scalebar)
            self.ax.axis('off')

    def _update_function(self, frame):

        if frame == self.max_frames - 1 :
            self.ani.event_source.stop()
            plt.close()

        t0 = time.time()
        image_cam = self.cam.get_pending_frame_or_null()
        poll_delay = time.time() - t0

        if image_cam is None:
            raise ImageException(f"Frame {frame} not received")
        
        t0 = time.time()
        image_buffer = np.copy(image_cam.image_buffer)
        shaped_image = image_buffer.reshape(self.cam.image_height_pixels, self.cam.image_width_pixels)
        if self.visParams.zoom_bool:
            i, j = np.unravel_index(shaped_image.argmax(), shaped_image.shape)
            shaped_image = shaped_image[i-self.visParams.zoom_width : i+self.visParams.zoom_width, 
                                        j-self.visParams.zoom_width : j+self.visParams.zoom_width]
            
        self.im.set_data(shaped_image)
        image_delay = time.time() - t0

        t0 = time.time()
        if self.visParams.downscale_bool:
            shaped_image = gaussian_filter(shaped_image, self.visParams.gaussian_filter_sigma)
            shaped_image = shaped_image[::self.visParams.downscale_order, ::self.visParams.downscale_order]

        if self.visParams.gaussian_fitting:
            popt, sim = gaussianFit(shaped_image, self.x0, self.y0, self.sigmax, self.sigmay, self.theta)
            self.x0, self.y0, self.sigmax, self.sigmay, self.theta = popt

        else:
            popt, sim = gaussianCompute(shaped_image)
            self.x0, self.y0, self.sigmax, self.sigmay, self.theta = popt
        fit_delay = time.time() - t0

        t0 = time.time()
        for coll in self.ax.collections:
            coll.remove()
        if self.visParams.downscale_bool:
            self.contour = self.ax.contour(sim[0]*self.visParams.downscale_order, sim[1]*self.visParams.downscale_order, sim[2], levels=5, colors='white')
        else:
            self.contour = self.ax.contour(*sim, levels=5, colors='white')
        self._make_parameters_text()
        self.textbox.set_text(self.text)
        contour_delay = time.time() - t0

        if self.verbosity > 3:
            col = Columns([f"polling: {poll_delay*1e3:>5.2f} ms", f"image process: {image_delay*1e3:>5.2f} ms", f"fitting: {fit_delay*1e3:>5.2f} ms", f"plotting: {contour_delay*1e3:>5.2f} ms"], expand=True)
            self.cns.print(col)
            self.cns.print("\n")

        if self.visParams.lengthscale_um:
            return self.im, self.contour, self.textbox, self.scalebar,
        else:
            return self.im, self.contour, self.textbox,

class RFanim(GaussianFitImaging):
    """
    ``GaussianFitImaging`` subclass providing support for moving the gaussian defect by changing the frequencies of the AOMS.

    Parameters
    ----------

    mogPort : ``int``
        The port in which the Moglabs XRF is connected. Use mogrf app to determine the port.

    freqRF1 : ``np.ndarray`` or ``list``, optional
        The (ordered) frequencies you want the channel 1 of the RF synthetizer to take during the imaging, in MHz.
        default = ``np.linspace(70, 90, 20)``

    powRF1 : ``np.ndarray`` or ``list``, optional
        The (ordered) power you want the channel 1 of the RF synthetizer to take during the imaging, in dBm.
        default = ``30*np.ones(20)``

    freqRF2 : ``np.ndarray`` or ``list``, optional
        The (ordered) frequencies you want the channel 2 of the RF synthetizer to take during the imaging, in MHz.
        default = ``np.linspace(70, 90, 20)``

    powRF2 : ``np.ndarray`` or ``list``, optional
        The (ordered) power you want the channel 2 of the RF synthetizer to take during the imaging, in dBm.
        default = ``30*np.ones(20)``

    camParams : ``CameraParameters``, optional
        Parameters of the camera acquisition for the imaging.

    visParams : ``VisualizationParameters``, optional
        Parameters of the visualization (matplotlib) for the imaging.

    console : ``rich.console.Console``, optional
        Console on which to print all informations.

    verbosity : ``int``, optional
        verbosity of the program during the animation, between 1 and 4.
        default = 2

    """

    def __init__(
            self,
            mogPort: int,
            freqRF1: np.ndarray | list = np.linspace(70, 90, 20),
            powRF1: np.ndarray | list | None = None,
            freqRF2: np.ndarray | list = np.linspace(70, 90, 20),
            powRF2: np.ndarray | list | None = None,
            camParams: CameraParameters = CameraParameters(),
            visParams: VisualizationParameters = VisualizationGaussianParameters(),
            console: Console = Console(),
            verbosity: int = 2
            ):
        
        assert len(freqRF1) == len(freqRF2)
        if powRF1 is not None:
            assert len(powRF1) == len(powRF2)
            assert len(freqRF1) == len(powRF1)

        super().__init__(camParams=camParams, visParams=visParams, console=console, verbosity=verbosity)

        self.mogdevice: MOGDevice = MOGDevice("COM", port=mogPort)
        self.freqRF1: np.ndarray | list = freqRF1
        self.freqRF2: np.ndarray | list = freqRF2

        if powRF1 is not None:
            self.powRF1: np.ndarray | list = powRF1
            self.powRF2: np.ndarray | list = powRF2
        else:
            self.powRF1 = 30*np.ones_like(self.freqRF1)
            self.powRF2 = 30*np.ones_like(self.freqRF2)

        # Stocking results of experiment
        self.list_wx: list = []
        self.list_wy: list = []
        self.list_theta: list = []
        self.list_intensity: list = []
        self.list_positionx: list = []
        self.list_positiony: list = []
        
        # Track last processed frame to avoid duplicates from animation initialization
        self._last_frame_processed: int = -1

    def _init_mog_device(self):

        self.mogdevice.cmd('MODE,1,NSB')
        self.mogdevice.cmd('MODE,2,NSB')

        self.mogdevice.cmd('FREQ,1,80MHz')
        self.mogdevice.cmd('FREQ,2,80MHz')

        self.mogdevice.cmd('POWER,1,30dBm')
        self.mogdevice.cmd('POWER,2,30dBm')

        self.mogdevice.cmd('PHASE,1,0')
        self.mogdevice.cmd('PHASE,2,0')

        self.mogdevice.cmd('ON,1')
        self.mogdevice.cmd("ON,2")

        self.mogdevice.cmd('SYNC,on')

        self.mogdevice.cmd(f'FREQ,1,{self.freqRF1[0]}')
        self.mogdevice.cmd(f'POW,1,{self.powRF1[0]}')
        self.mogdevice.cmd(f'FREQ,2,{self.freqRF2[0]}')
        self.mogdevice.cmd(f'POW,2,{self.powRF2[0]}')

        if self.verbosity>1:

            tabMog = Table(caption="Mog Device Initialization")
            tabMog.add_column('Channel', justify='left')
            tabMog.add_column('Frequency', justify="right", highlight=True)
            tabMog.add_column('Amplitude', justify="right", highlight=True)

            tabMog.add_row("1", f"{float(self.mogdevice.ask("FREQ, 1").split("MHz")[0]):.1f} MHz", f"{float(self.mogdevice.ask("POW, 1").split("dBm")[0]):.1f} dBm")
            tabMog.add_row("2", f"{float(self.mogdevice.ask("FREQ, 2").split("MHz")[0]):.1f} MHz", f"{float(self.mogdevice.ask("POW, 2").split("dBm")[0]):.1f} dBm")

            self.cns.print(tabMog, justify='center')

    def _init_function(self):

        self._init_mog_device()
        super()._init_function()

    def _update_function(self, frame):

        # Skip duplicate frame calls from animation initialization
        if frame == self._last_frame_processed:
            if self.visParams.lengthscale_um:
                return self.im, self.contour, self.textbox, self.scalebar,
            else:
                return self.im, self.contour, self.textbox,
        
        self._last_frame_processed = frame

        self.mogdevice.cmd(f"FREQ,1,{self.freqRF1[frame]}")
        self.mogdevice.cmd(f"POW,1,{self.powRF1[frame]}")
        self.mogdevice.cmd(f"FREQ,2,{self.freqRF2[frame]}")
        self.mogdevice.cmd(f'POW,2,{self.powRF2[frame]}')

        if self.verbosity > 2:
            
            col1 = Columns([f"Channel {1:d}", f"freq {float(self.mogdevice.ask('FREQ, 1').split("MHz")[0]):>8.2f} MHz", f"pow {float(self.mogdevice.ask("POW,1").split("dBm")[0]):>8.2f} dBm"], expand=True)
            col2 = Columns([f"Channel {2:d}", f"freq {float(self.mogdevice.ask('FREQ, 2').split("MHz")[0]):>8.2f} MHz", f"pow {float(self.mogdevice.ask("POW, 2").split("dBm")[0]):>8.2f} dBm"], expand=True)
            self.cns.print(col1)
            self.cns.print(col2)
            self.cns.print('\n')

        super()._update_function(frame)

        self.list_intensity.append(self.im.get_array().max())
        self.list_positionx.append(self.x0 * self.scale)
        self.list_positiony.append(self.y0 * self.scale)
        self.list_wx.append(2 * self.sigmax * self.scale)
        self.list_wy.append(2 * self.sigmay * self.scale)
        self.list_theta.append(self.theta)

        """if frame == len(self.freqRF1) - 1: #last frame
            self.cns.print(":warning:  [bold red]Experiment over, Figure closing...[/]")
            self.ani.event_source.stop()
            plt.close()"""

        if self.visParams.lengthscale_um:
            return self.im, self.contour, self.textbox, self.scalebar,
        else:
            return self.im, self.contour, self.textbox,

    def run(self):
        """Launches the animation."""

        self.max_frames = len(self.freqRF1)

        super().run(number_of_frames=self.max_frames, interval_ms=1)
        self.mogdevice.cmd('FREQ,1,80')
        self.mogdevice.cmd("FREQ,2,80")
        self.mogdevice.cmd('POW,1,30 dBm')
        self.mogdevice.cmd('POW,2,30 dBm')

    def get_results(self):
        """
        Returns all values of gaussian defect parameters during imaging, and eventually prints the results in a table.
        
        Returns
        -------
        
        freqRF1, freqRF2 : ``np.ndarray`` or ``list``
            The frequencies of the AOMs at each animation step, in MHz.

        intensities : ``list``
            The maximum intensity of the gaussian defect, in a.u.

        x0, y0 : ``list``
            The positions of the center of the beam, in µm.

        wx, wy : ``list``
            The waists of the beam in two principal directions, in µm.

        theta : ``list``
            The angle that the elliptic gaussian beam makes with the x-axis of the camera, in degrees.

        """

        if self.verbosity > 1:

            tab = Table('frequency x [MHz]', 'frequency y [MHz]', 'Intensity', 'x0 [µm]', 'y0 [µm]', 'wx [µm]', 'wy [µm]', 'theta [°]', highlight=True, caption="Gaussian beam evolution")

            for i in range(0, len(self.list_intensity), 10):
                tab.add_row(f"{self.freqRF1[i]:.1f}", f"{self.freqRF2[i]:.1f}", f"{self.list_intensity[i]:.0f}", f"{self.list_positionx[i]:.1f}", f"{self.list_positiony[i]:.1f}", f"{self.list_wx[i]:.1f}", f"{self.list_wy[i]:.1f}", f"{self.list_theta[i]:.0f}")
            
            self.cns.print('\n')
            self.cns.print(tab)
            self.cns.print("\n")

        return self.freqRF1, self.freqRF2, self.list_intensity, self.list_positionx, self.list_positiony, self.list_wx, self.list_wy, self.list_theta


