#from ThorlabsGaussianTools.utils.mogdevice import MOGDevice
from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#TODO: Create a dataclass representing the path we want the point to follow

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

    x_pol : optional
        The function giving the x position change (in µm) in function of the x frequency (in MHz).
        default = np.polynomial.Polynomial((0,1))
    
    y_pol : optional
        The function giving the y position change (in µm) in function of the y frequency (in MHz).
        default = np.polynomial.Polynomial((0,1))
    
    """

    time_ms: np.ndarray
    x_frequencies: np.ndarray
    y_frequencies: np.ndarray
    x_amplitudes: np.ndarray
    y_amplitudes: np.ndarray
    x_pol = np.polynomial.Polynomial((0,1))
    y_pol = np.polynomial.Polynomial((0,1))

    @classmethod
    def from_calibration(
        cls,
        amp_corr_x: function,
        amp_corr_y: function,
        x_pol: np.polynomial.Polynomial,
        y_pol: np.polynomial.Polynomial,
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

        x_pos, y_pos : ``np.ndarray``
            Ordered list of the point positions, in µm.

        time_ms : ``np.ndarray``
            Ordered list of time for each event (position), in ms.


        """
        #TODO: From a calibration result, get the functions giving amplitudes for each frequency and frequency for a given position

        b,a = x_pol.convert().coef
        x_pol_inv = np.polynomial.Polynomial((-b/a, 1/a))
        b,a = y_pol.convert().coef
        y_pol_inv = np.polynomial.Polynomial((-b/a, 1/a))

        frequencies_x = x_pol_inv(x_pos)
        frequencies_y = y_pol_inv(y_pos)
        amplitudes_x = amp_corr_x(frequencies_x)
        amplitudes_y = amp_corr_y(frequencies_y)

        cls(time_ms, frequencies_x, frequencies_y, amplitudes_x, amplitudes_y, x_pol, y_pol)


    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.x_pol(self.x_frequencies), self.y_pol(self.y_frequencies))
        plt.show()

    def ani_plot(self, precision: int = 100):
        fig, ax = plt.subplots()
        x = self.x_pol(self.x_frequencies)
        y = self.y_pol(self.y_frequencies)
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





#TODO: Create simple class(es) to control frequencies and power of a mogdevice through time.

if __name__ == "__main__":

    myPath = Path(
        1000*np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        np.array([70, 90, 90, 70, 70, 80, 70, 70]),
        np.array([70, 70, 90, 90, 70, 80, 90, 70]),
        np.array([30, 30, 30, 30, 30, 30, 30, 30]),
        np.array([30, 30, 30, 30, 30, 30, 30, 30])
    )

    myPath.ani_plot()