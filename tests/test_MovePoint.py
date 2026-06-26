from MogrfTools import Path, MovePoint
import numpy as np

N=100
x_frequencies = np.linspace(60, 100, N)
y_frequencies = np.ones_like(x_frequencies) * 80
time_ms = np.linspace(0, 1000, N)
x_amplitudes = np.ones_like(x_frequencies) * 30
y_amplitudes = np.ones_like(y_frequencies) * 30

myPath = Path(time_ms, x_frequencies, y_frequencies, x_amplitudes, y_amplitudes)

#myPath.ani_plot()

myAni = MovePoint(myPath, mog_port=7)

myAni.run(repeat = True)