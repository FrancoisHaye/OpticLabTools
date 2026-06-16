import numpy as np
import matplotlib.pyplot as plt

res=np.load('AOMy.npz')
freq = res['frequency']
intensity = res['intensity']
x0 = res['positionx']
y0 = res['positiony']
wx = res['wx']
wy = res['wy']
theta = res['theta']

plt.plot(freq, intensity, '.')
plt.show()