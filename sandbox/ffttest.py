import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft
from scipy import signal

# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0, N*T, N+1)

y = signal.sawtooth(2 * np.pi * 5 * x, width=.5)
yf = fft(y)
xf = np.linspace(0.0, 1.0 / (2.0*T), N//2)

plt.plot(x, y)
plt.show()


plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.show()

