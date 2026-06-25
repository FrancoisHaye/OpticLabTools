#from ThorlabsGaussianTools.utils.mogdevice import MOGDevice
from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ThorlabsGaussianTools.utils.mogdevice import MOGDevice
from rich.console import Console

#TODO: Create a dataclass representing the path we want the point to follow
# Tested: it works !!!

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
        Creating a path given a set of coordinates and timesteps. Results of an AOM calibration is needed to determine the right frequencies and amplitudes.

        Parameters
        ----------
        amp_corr_x, amp_corr_y: ``function``
            Vectorized function ``f`` giving ``A = f(freq)``, the amplitude corresponding to the frequency freq in order for the intensity to remain constant during the movement of the point.

        x_pol, y_pol : ``np.polynomial.Polynomial``
            1st order polynomial ``ax+b`` giving the relatioship between position and frequency.

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
        new_points = np.linalg.solve(M, points)

        frequencies_x = x_pol_inv(new_points[0, :])
        frequencies_y = y_pol_inv(new_points[1, :])
        amplitudes_x = amp_corr_x(frequencies_x)
        amplitudes_y = amp_corr_y(frequencies_y)

        path = cls(time_ms, frequencies_x, frequencies_y, amplitudes_x, amplitudes_y)
        return path

    def plot(self, x_pol = np.polynomial.Polynomial((0,1)), y_pol = np.polynomial.Polynomial((0,1))):
        fig, ax = plt.subplots()
        ax.plot(x_pol(self.x_frequencies), y_pol(self.y_frequencies))
        plt.show()

    def ani_plot(self, x_pol = np.polynomial.Polynomial((0,1)), y_pol = np.polynomial.Polynomial((0,1)), precision: int = 100):
        fig, ax = plt.subplots()
        x = x_pol(self.x_frequencies)
        y = y_pol(self.y_frequencies)
        line, = ax.plot([], [], linestyle='-', marker='o')
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


#TODO: Create simple class(es) to control frequencies and power of a mogdevice through time (using Path class)

class MovePoint:
    """
    Interfacing with mogRF to move the points following the ``Path``

    Parameters
    ----------
    path : ``path``
        The path to follow, containing all necessary information to program the mogdevice. The smaller the timesteps the more precise.

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

        self.mogdevice = MOGDevice("COM", mog_port)
        self.path = path
        self.cns = cns

    def _mogdevice_zero(self):
        
        self.mogdevice.cmd('FREQ,1,80')
        self.mogdevice.cmd('FREQ,2,80')
        self.mogdevice.cmd('POW,1,30')
        self.mogdevice.cmd('POW,2,30')
        self.cns.print(f'Mog device status after zero: {self.mogdevice.ask('STATUS'):>8}')


    def run(self, repeat=True):
        """
        Launch the loop for moving the point following the ``path``, without camera.
        
        Parameters
        ----------
        repeat : ``bool``, optional
            Whether to run the animation only once or to repeat it until stopped by user.
            default = True
            
        """


        with self.cns.status('MovePoint running') :
                
            print(f"MogDevice Status: {self.mogdevice.ask("STATUS"):>8}")
            self._mogdevice_zero()

            i = 0
            while True:

                if i>=len(self.path.time_ms):
                    if repeat:
                        i = 0
                    else:
                        break
                
                if i == 0:
                    time.sleep(self.path.time_ms[i]/1000)
                else:
                    time.sleep((self.path.time_ms[i] - self.path.time_ms[i-1])/1000)

                self.mogdevice.cmd(f'FREQ, 1, {self.path.x_frequencies[i]}')
                self.mogdevice.cmd(f'FREQ, 2, {self.path.y_frequencies[i]}')
                self.mogdevice.cmd(f'POW, 1, {self.path.x_amplitudes[i]}')
                self.mogdevice.cmd(f'POW, 2, {self.path.y_amplitudes[i]}')

                i+=1

            self._mogdevice_zero()
            self.mogdevice.close()
            self.cns.print('Animation over')

                