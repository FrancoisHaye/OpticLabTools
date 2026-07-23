"""
---------------
mogrf_utilities
---------------

Classes to create motions of the point, either by specifying frequencies and amplitudes of rf signal or by specifying results from a calibration experiment (see MogrfTools.launch_calibration) and a desired path in position coordinates (x and y in µm).

"""

#from ThorlabsGaussianTools.utils.mogdevice import MOGDevice
from dataclasses import dataclass
import time
import numpy as np
import math
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ThorlabsGaussianTools.utils.mogdevice import MOGDevice
from rich.console import Console

# Constants
N = 8190    # Maximum number of entries for Simple Table Mode
DT = 1e-3   # Minimum duration of entries for Simple Table Mode, in ms

@dataclass
class Path:
    """
    Dataclass representing a path for the gaussian defect to follow. It consists of two arrays giving frequencies for both AOMs, two arrays giving the amplitudes of the AOMs and a time array giving the timestamps of each event.
    
    Parameters
    ----------
    time_ms : ``np.ndarray``
        The timestamps giving the moment of each event when drawing, in ms.
        For instance an array [0, 10] will consist of two events, one on the launch of the program and one 10ms later.

    x_frequencies : ``np.ndarray``
        The list of frequencies for the x-direction AOM, in MHz.

    y_frequencies : ``np.ndarray``
        The list of frequencies for the y-direction AOM, in MHz.

    x_amplitudes : ``np.ndarray``
        The list of amplitudes for the x-direction AOM, in dBm.

    y_amplitudes : ``np.ndarray``
        The list of amplitudes for the y-direction AOM, in dBm.

    central_freq : ``float``, optional
        The central frequency for AOM usage, in MHz.
        default = 80
    
    """

    time_ms: np.ndarray
    x_frequencies: np.ndarray
    y_frequencies: np.ndarray
    x_amplitudes: np.ndarray
    y_amplitudes: np.ndarray

    @classmethod
    def from_calibration(
        cls,
        amp_corr_x: function,
        amp_corr_y: function,
        x_pol: np.polynomial.Polynomial,
        y_pol: np.polynomial.Polynomial,
        alpha: float,
        beta: float,
        x_pos: np.ndarray,
        y_pos: np.ndarray,
        time_ms: np.ndarray
    ):
        """
        Create a path given a set of coordinates and timesteps. Results of an AOM calibration are needed to determine the right frequencies and amplitudes.

        Parameters
        ----------
        amp_corr_x, amp_corr_y: ``function``
            Vectorized function ``f`` giving ``A = f(freq)``, the amplitude corresponding to the frequency freq in order for the intensity to remain constant during the movement of the point.

        x_pol, y_pol : ``np.polynomial.Polynomial``
            1st order polynomial ``ax+b`` giving the relationship between position and frequency.

        alpha : ``float``
            Angle between the two axis of the AOMs, in rad.

        beta : ``float``
            Angle between the x-axis of the AOM coordinate system and the x-axis of the camera, in rad.

        x_pos, y_pos : ``np.ndarray``
            Ordered list of the point positions, in µm.

        time_ms : ``np.ndarray``
            Ordered list of time for each event (position), in ms.


        """

        b,a = x_pol.convert().coef
        x_pol_inv = np.polynomial.Polynomial((-b/a, 1/a))
        b,a = y_pol.convert().coef
        y_pol_inv = np.polynomial.Polynomial((-b/a, 1/a))

        Rot = np.array([
            [np.cos(beta), -np.sin(beta)],
            [np.sin(beta), np.cos(beta)]
        ])

        strain = np.array([
            [1, np.cos(alpha)],
            [0, np.sin(alpha)]
        ])

        M = strain @ Rot

        points = np.vstack((x_pos, y_pos))
        if alpha == 0 and beta == 0:
            new_points = np.copy(points)
        else:
            new_points = np.linalg.solve(M, points)

        frequencies_x = x_pol_inv(new_points[0, :])
        frequencies_y = y_pol_inv(new_points[1, :])
        amplitudes_x = amp_corr_x(frequencies_x)
        amplitudes_y = amp_corr_y(frequencies_y)

        path = Path(time_ms, frequencies_x, frequencies_y, amplitudes_x, amplitudes_y)
        return path

    def plot(self, x_pol = np.polynomial.Polynomial((-40,0.5)), y_pol = np.polynomial.Polynomial((-40,0.5))):
        """ Plot the set path. """
        fig, ax = plt.subplots()
        ax.plot(x_pol(self.x_frequencies), y_pol(self.y_frequencies))
        plt.show()

    def ani_plot(self, x_pol = np.polynomial.Polynomial((-40,0.5)), y_pol = np.polynomial.Polynomial((-40,0.5)), precision: int = 100):
        """Plot an animation of the path."""
        
        fig, ax = plt.subplots()
        x = x_pol(self.x_frequencies)
        y = y_pol(self.y_frequencies)
        line, = ax.plot([], [], linestyle='-', marker='.')
        j0: int = 0
        t0 = time.time()

        def init():
            ax.set(
            xlim = (x.min()-5, x.max()+5),
            ylim = (y.min()-5, y.max()+5)
            )
            return line,

        def update(frame):
            
            nonlocal j0
            if frame==0:
                j0 = 0

            if (j0<len(x)) and ((time.time() - t0)*1000 >= self.time_ms[j0]):
                line.set_data(x[:j0], y[:j0])
                j0+=1

            return line,

        interval = (self.time_ms[1:] - self.time_ms[:-1]).min() / precision
        frames = int((self.time_ms[-1]-self.time_ms[0])/interval) + precision//2

        ani = FuncAnimation(fig, update, frames = frames, init_func=init, blit=True, repeat=False, interval=interval)
        plt.show()

class Line(Path):
    """
    Subclass of ``Path`` with necessary constructors to make a straight line in an abritrary direction.
    
    Parameters
    ----------
    x_amplitudes, y_amplitudes : ``np.ndarray``
        The lists of amplitudes. ``len(amplitudes) == length_MHz / speed_MHz_ms / dt_ms``.
        
    theta_deg : ``float``, optional
        The angle between the line and the x-axis, in degrees.
        default = 45°

    length_MHz : ``float``, optional
        The length of the line. The point will move between ``-length_MHz/2`` and ``+length_MHz/2``, in MHz.
        default = 20

    speed_MHz_ms : ``float``, optional
        The speed of the moving point, in MHz/ms.
        default = 0.01

    n : ``int``, optional
        The number of steps in the movement of the point.
        default = N//2

    backwards : ``bool``, optional
        Whether to do a back and forth movement or a one way only.
        default = False


    Raises
    ------
    AssertionError
        if the total number of steps exceeds the maximum size of moglabs table mode (N).
        
    """

    def __init__(self, x_amplitudes, y_amplitudes, theta_deg: float = 45, length_MHz: float = 10., speed_MHz_ms: float = 0.01, n: int = N//2, backwards: bool = False):

        if backwards:
            assert 2 * n <= N
        else:
            assert n <= N
        assert n == len(x_amplitudes)

        dt_ms = length_MHz / n / speed_MHz_ms
        time_ms = np.arange(0, dt_ms*n, dt_ms)
        freq0 = np.linspace(-length_MHz/2, length_MHz/2, n)
        freqx = np.cos(-np.radians(theta_deg)) * freq0
        freqy = np.sin(-np.radians(theta_deg)) * freq0

        if backwards:
            time_ms = np.concatenate((time_ms, time_ms[-1] + time_ms))
            freqx = np.concatenate((freqx, freqx[::-1]))
            freqy = np.concatenate((freqy, freqy[::-1]))
            x_amplitudes = np.concatenate((x_amplitudes, x_amplitudes[::-1]))
            y_amplitudes = np.concatenate((y_amplitudes, y_amplitudes[::-1]))

        super().__init__(time_ms=time_ms, x_frequencies=freqx, y_frequencies=freqy, x_amplitudes=x_amplitudes, y_amplitudes=y_amplitudes)

    @classmethod
    def from_calibration(cls, amp_corr_x: function, amp_corr_y: function, x_pol: np.polynomial.Polynomial, y_pol: np.polynomial.Polynomial, alpha: float, beta: float, theta_deg: float = 45, length_um: float = 10., speed_um_ms: float = 0.01, n: int = N//2, backwards: bool = False):
        """
        Make a Line Path from AOM calibration results.

        Parameters
        ----------
        amp_corr_x, amp_corr_y : ``function``
            The function giving the amplitude correction to intensity variation with frequency. Call signature: amplitudes_i = amp_corr_i(frequencies_i).

        x_pol, y_pol : ``np.polynomial.Polynomial``
            Polynomial of degree 1 giving the variation of position with frequency. Call signature: position_i = i_pol(frequency_i).

        alpha : ``float``
            Angle between the two AOMs directions, in rad.

        beta : ``float``
            Angle between the AOMs x-axis and the camera x-axis, in rad.

        theta_deg : ``float``, optional
            The angle between the line and the x-axis, in degrees.
            default = 45°

        length_um : ``float``, optional
            The length of the line. The point will move between ``-length_um/2`` and ``+length_um/2``, in µm.
            default = 10

        speed_um_ms : ``float``, optional
            The speed of the moving point, in µm/ms.
            default = 0.01

        n : ``int``, optional
            The number of steps in the movement of the point.
            default = N//2

        backwards : ``bool``, optional
            Whether to do a back and forth or only a one way.
            default = False
        
        """
        if backwards:
            assert 2 * n <= N
        else:
            assert n <= N
        
        dt_ms = length_um / n / speed_um_ms
        time_ms = np.arange(0, dt_ms*n, dt_ms)
        r_pos = np.linspace(-length_um/2, length_um/2, n)
        x_pos = r_pos * np.cos(-np.radians(theta_deg))
        y_pos = r_pos * np.sin(-np.radians(theta_deg))

        if backwards:
            x_pos = np.concatenate((x_pos, x_pos[::-1]))
            y_pos = np.concatenate((y_pos, y_pos[::-1]))
            time_ms = np.concatenate((time_ms, time_ms[-1] + time_ms))

        return super().from_calibration(amp_corr_x=amp_corr_x, amp_corr_y=amp_corr_y, x_pol=x_pol, y_pol=y_pol, alpha=alpha, beta=beta, x_pos=x_pos, y_pos=y_pos, time_ms=time_ms)

class Circle(Path):
    """
    Subclass of ``Path`` with necessary constructors to make a Circle.
    
    Parameters
    ----------
    x_amplitudes, y_amplitudes : ``np.ndarray``
        The lists of amplitudes. ``len(amplitudes) == length_MHz / speed_MHz_ms / dt_ms``.
        
    R_MHz : ``float``, optional
        The radius of the circle, in MHz.
        default = 20

    x0_MHz, y0_MHz : ``float``, optional
        The position of the center of the circle, in MHz.
        default = 0

    speed_MHz_ms : ``float``, optional
        The speed of the moving point, in MHz/ms.
        default = 0.01

    n : ``int``, optional
        The number of steps in the movement of the point.
        default = N//2

    Raises
    ------
    AssertionError
        if the number of steps is larger than the mogrf table mode limit.
        
    """

    def __init__(self, x_amplitudes, y_amplitudes, R_MHz: float = 20, x0_MHz: float = 0, y0_MHz: float = 0, speed_MHz_ms: float = 0.01, n: int = N//2):
        
        assert n <= N

        dt_ms = 2*np.pi*R_MHz / (n * abs(speed_MHz_ms))
        if speed_MHz_ms >= 0:
            theta = np.linspace(0, 2*np.pi, n)
        else:
            theta = np.linspace(0, -2*np.pi, n)
        time_ms = np.linspace(0, n*dt_ms, n)
        freqx = x0_MHz + R_MHz * np.cos(theta)
        freqy = y0_MHz + R_MHz * np.sin(theta)

        return super().__init__(time_ms, freqx, freqy, x_amplitudes, y_amplitudes)

    @classmethod
    def from_calibration(cls, amp_corr_x: function, amp_corr_y: function, x_pol: np.polynomial.Polynomial, y_pol: np.polynomial.Polynomial, alpha: float, beta: float, R_um: float = 10., x0_um: float = 0, y0_um: float = 0, speed_um_ms: float = 0.01, n: int = N//2):
        """
        Make a Circle Path from AOM calibration results.

        Parameters
        ----------
        amp_corr_x, amp_corr_y : ``function``
            The function giving the amplitude correction to intensity variation with frequency. Call signature: amplitudes_i = amp_corr_i(frequencies_i).

        x_pol, y_pol : ``np.polynomial.Polynomial``
            Polynomial of degree 1 giving the variation of position with frequency. Call signature: position_i = i_pol(frequency_i).

        alpha : ``float``
            Angle between the two AOMs directions, in rad.

        beta : ``float``
            Angle between the AOMs x-axis and the camera x-axis, in rad.

        R_um : ``float``, optional
            The radius of the circle, in µm.
            default = 10

        x0_um, y0_um : ``float``, optional
            The position of the center of the circle, in µm.
            default = 0

        speed_um_ms : ``float``, optional
            The speed of the moving point, in µm/ms. The sign (+/-) determines the the direction of rotation.
            default = 0.01

        n : ``int``, optional
            The number of steps in the movement of the point.
            default = N//2
        
        """

        assert n <= N
        dt_ms = 2*np.pi*R_um / (n * abs(speed_um_ms))
        assert dt_ms > DT
        if speed_um_ms >= 0:
            theta = np.linspace(0, 2*np.pi, n)
        else:
            theta = np.linspace(0, -2*np.pi, n)
        x_pos = x0_um + R_um * np.cos(theta)
        y_pos = y0_um + R_um * np.sin(theta)
        time_ms = np.linspace(0, dt_ms*n, n)

        return super().from_calibration(amp_corr_x=amp_corr_x, amp_corr_y=amp_corr_y, x_pol=x_pol, y_pol=y_pol, alpha=alpha, beta=beta, x_pos=x_pos, y_pos=y_pos, time_ms=time_ms)

class Lissajous(Path):
    """
    Subclass of ``Path`` with necessary constructors to make a Lissajous curve.
    
    Parameters
    ----------
    x_amplitudes, y_amplitudes : ``np.ndarray``
        The lists of amplitudes. ``len(amplitudes) == length_MHz / speed_MHz_ms / dt_ms``.
        
    Lx_MHz, Ly_MHz : ``float``, optional
        The length of displacement in x (resp. y) direction.
        default = 20

    speed_ratio : ``float``, optional
        The ratio of speeds b/a in the lissajous curve def. Must be a rational number.
        default = 0

    delta_rad : ``float``, optional
        The dephasage between x and y, in rad.
        default = 0

    x0_MHz, y0_MHz : ``float``, optional
        The position of the center of the lissajous curve.
        default = 0

    duration_ms : ``float``, optional
        The overall duration of one entire pass of the Lissajous curve, in ms.
        default = 1000

    n : ``int``, optional
        The number of points.
        default = N/2
        
    """

    def __init__(self, x_amplitudes, y_amplitudes, Lx_MHz: float = 20, Ly_MHz: float = 10, speed_ratio: float = 1, delta_rad: float = 0, x0_MHz: float = 0, y0_MHz: float = 0, duration_ms: float = 1000, n: int = N//2):
        
        assert n <= N
        time_ms = np.linspace(0, duration_ms, n, endpoint=False)
        frac = Fraction(speed_ratio).limit_denominator()
        q = frac.denominator
        a = q * 2*np.pi / duration_ms
        b = speed_ratio*a
        freqx = x0_MHz + Lx_MHz/2 * np.sin(a * time_ms + delta_rad)
        freqy = y0_MHz + Ly_MHz/2 * np.sin(b * time_ms)

        return super().__init__(time_ms, freqx, freqy, x_amplitudes, y_amplitudes)

    @classmethod
    def from_calibration(cls, amp_corr_x: function, amp_corr_y: function, x_pol: np.polynomial.Polynomial, y_pol: np.polynomial.Polynomial, alpha: float, beta: float, Lx_um: float = 10., Ly_um: float = 10, delta_rad: float = 0, speed_ratio: float = 1, x0_um: float = 0, y0_um: float = 0, duration_ms: int = 1000, n: int = N//2):
        """
        Make a ``Lissajous`` path from calibration results.
        
        Parameters
        ----------
        amp_corr_x, amp_corr_y : ``function``
            The function giving the amplitude correction to intensity variation with frequency. Call signature: amplitudes_i = amp_corr_i(frequencies_i).

        x_pol, y_pol : ``np.polynomial.Polynomial``
            Polynomial of degree 1 giving the variation of position with frequency. Call signature: position_i = i_pol(frequency_i).

        alpha : ``float``
            Angle between the two AOMs directions, in rad.

        beta : ``float``
            Angle between the AOMs x-axis and the camera x-axis, in rad.
            
        Lx_um, Ly_um : ``float``, optional
            The length of displacement in x (resp. y) direction.
            default = 10

        speed_ratio : ``float``, optional
            The ratio of speeds b/a in the lissajous curve def. Must be a rational number.
            default = 0

        delta_rad : ``float``, optional
            The dephasage between x and y, in rad.
            default = 0

        x0_um, y0_um : ``float``, optional
            The position of the center of the lissajous curve.
            default = 0

        duration_ms : ``float``, optional
            The overall duration of one entire pass of the Lissajous curve, in ms.
            default = 1000

        n : ``int``, optional
            The number of points.
            default = N/2
            
        """

        assert n <= N
        time_ms = np.linspace(0, duration_ms, n, endpoint=False)
        frac = Fraction(speed_ratio).limit_denominator()
        q = frac.denominator
        a = q * 2*np.pi / duration_ms
        b = speed_ratio*a
        x_pos = x0_um + Lx_um/2 * np.sin(a * time_ms + delta_rad)
        y_pos = y0_um + Ly_um/2 * np.sin(b * time_ms)

        return super().from_calibration(amp_corr_x=amp_corr_x, amp_corr_y=amp_corr_y, x_pol=x_pol, y_pol=y_pol, alpha=alpha, beta=beta, x_pos=x_pos, y_pos=y_pos, time_ms=time_ms)


class MovePoint:
    """
    Interface class with mogRF to move the points following the specified ``Path``.

    Parameters
    ----------
    path : ``Path``
        The path to follow, containing all necessary information to program the mogdevice.

    mog_port : ``int``, optional
        The USB port of the mogdevice.
        default = 7

    cns : ``rich.console.Console``, optional
        The console on which to print results.
        default = Console()
    """

    def __init__(self, path: Path, mog_port: int = 7, cns = Console()):

        assert path.time_ms.shape == path.x_frequencies.shape
        assert path.x_frequencies.shape == path.y_frequencies.shape
        assert path.y_frequencies.shape == path.y_amplitudes.shape
        assert path.x_frequencies.shape == path.x_amplitudes.shape

        self.mogdevice = MOGDevice("COM", mog_port, timeout=100)
        self.path = path
        self.cns = cns


    def run(self, repeat=False, pause_ms: float = 10):
        """
        Launch the loop for moving the point following the ``path``, without camera.
        
        Parameters
        ----------
        repeat : ``bool``, optional
            Whether to run the animation only once or to repeat it until stopped by user with keyboard (CTRL+C)
            default = True

        pause_ms : ``float``, optional
            If the animation repeat, the duration between the end of a run and the start of the next one, in ms.
            default = 10
            
        """


        with self.cns.status('Table mode Creation') :

            duration_ms = self.path.time_ms[1:] - self.path.time_ms[:-1]
            duration_ms = np.concatenate((duration_ms, np.array([pause_ms])))
            self.cns.print(f"times\n{self.path.time_ms}\n\ndurations\n{duration_ms}\n\nfrequencies_x\n{self.path.x_frequencies}")
            still_going = True

            try:

                self.mogdevice.cmd('MODE, 1, TSB')
                self.mogdevice.cmd('MODE, 2, TSB')
                self.mogdevice.cmd('TABLE, ENTRIES, 1, 0')
                self.mogdevice.cmd('TABLE, ENTRIES, 2, 0')

                self.mogdevice.cmd('TABLE, SYNC, ON')

                for i in range(len(self.path.time_ms)):
                    # TABLE, APPEND, ch, freq, ampl, phase, duration
                    self.mogdevice.cmd(f" TABLE, ENTRY, 1, {i+1}, {self.path.x_frequencies[i]} MHz, {self.path.x_amplitudes[i]} dBm, 0, {duration_ms[i]} ms")
                    time.sleep(0.001)
                    self.mogdevice.cmd(f" TABLE, ENTRY, 2, {i+1}, {self.path.y_frequencies[i]} MHz, {self.path.y_amplitudes[i]} dBm, 0, {duration_ms[i]} ms")
                    time.sleep(0.001)
                
                self.mogdevice.cmd(f'TABLE, ENTRIES, 1, {len(self.path.x_amplitudes)}')
                time.sleep(0.01)
                self.mogdevice.cmd(f'TABLE, ENTRIES, 2, {len(self.path.y_amplitudes)}')
                time.sleep(0.01)
                self.mogdevice.cmd('TABLE, SAVE, 1, 1') # Save on slot 1
                time.sleep(0.01)
                self.mogdevice.cmd('TABLE, SAVE, 2, 1') # Save on slot 1
                time.sleep(0.01)

            except KeyboardInterrupt as e:
                self.cns.print("Animation stopped, exiting...")
                self.mogdevice.cmd('TABLE, STOP, 1')
                self.mogdevice.cmd('MODE, 1, NSB')
                self.mogdevice.cmd('MODE, 2, NSB')
                self.mogdevice.cmd('ON, 1')
                self.mogdevice.cmd('ON, 2')
                self.mogdevice.close()

        with self.cns.status('Animation launched'):

            try:
                if repeat:
                    self.mogdevice.cmd('TABLE, REARM, 1, ON')
                    self.mogdevice.cmd('TABLE, REARM, 2, ON')
                    self.mogdevice.cmd('TABLE, RESTART, 1, ON')
                    #self.mogdevice.cmd('TABLE, RESTART, 2, ON')

                self.mogdevice.cmd('TABLE, ARM, 1')
                self.mogdevice.cmd('TABLE, ARM, 2')
                self.mogdevice.cmd('TABLE, START, 1')
                t0 = time.time()

                while still_going:
                    
                    still_going = repeat or (time.time() - t0) <= self.path.time_ms[-1]/1000

            except KeyboardInterrupt as e:
                self.cns.print("Animation stopped, exiting...")

            finally :
                self.mogdevice.cmd('TABLE, STOP, 1')
                self.mogdevice.cmd('MODE, 1, NSB')
                self.mogdevice.cmd('MODE, 2, NSB')
                self.mogdevice.cmd('ON, 1')
                self.mogdevice.cmd('ON, 2')
                self.mogdevice.close()