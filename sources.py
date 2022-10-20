import numpy as np
import matplotlib.pyplot as plt

# manufactured solution 1
# u(x,t) = cos(2*pi*x/L) * 1/3*sin(2*pi*t)**3
# u'(x,t) = -2*pi*/L*sin(2*pi*x/L) * 1/3*sin(2*pi*t)**3
# u''(x,t) = -4*pi**2/L**2*cos(2*pi*x/L)  * 1/3*sin(2*pi*t)**3
# dudt(x,t) = cos(2*pi*x/L) * sin(2*pi*t)**2 * 2*pi*cos(2*pi*t)
# ddudt^2(x,t) = cos(2*pi*x/L) * ( 2*sin(2*pi*t) * 4*pi**2*cos(2*pi*t)**2 - sin(2*pi*t)**2 * 4*pi**2*sin(2*pi*t) )


class Manufactured1:
    def __init__(self, wx, wt):
        self.wx = wx
        self.wt = wt

    def uxt(self, x, t):
        return np.cos(self.wx * x) * 1 / 3 * np.sin(self.wt * t) ** 3

    def fx(self, x):
        return np.cos(self.wx * x)

    def ft(self, t):
        sin = np.sin(self.wt * t)
        cos = np.cos(self.wt * t)
        return 2 * sin * self.wt ** 2 * cos ** 2 - sin ** 2 * self.wt ** 2 * sin + self.wx ** 2 * 1 / 3 * sin ** 3

    def fxt(self, x, t):
        return self.fx(x) * self.ft(t)


class RicklersWavelet:
    def __init__(self, freq, alpha):
        self.alpha = alpha
        self.freq = freq
        self.t0 = 1.0 / freq
        self.sigmaT = 1.0 / (2.0 * np.pi * freq)
        self.sigmaS = 0.03

    def ft(self, t):
        return -(t - self.t0) / (np.sqrt(2 * np.pi) * self.sigmaT ** 3) * np.exp(-(t - self.t0) ** 2 / (2 * self.sigmaT ** 2))

    def fx(self, x):
        return 0.25 * self.alpha(x) / np.sqrt(2 * np.pi * self.sigmaS ** 2) * np.exp(-x ** 2 / (2 * self.sigmaS ** 2))

    def fxt(self, x, t):
        return self.fx(x) * self.ft(t)

    def uxt(self, x, t):
        return 0.0


class NoSource:
    def __init__(self):
        pass

    def ft(self, t):
        return 0

    def fx(self, x):
        return x * 0

    def fxt(self, x, t):
        return self.fx(x) * self.ft(t)

    def uxt(self, x, t):
        return 0.0



def plotSource(source, left, right, n):
    xx = np.linspace(left, right, n)
    yy = xx * 0
    for i in range(len(xx)):
        yy[i] = source.fx(xx[i])
    figure, ax = plt.subplots()
    ax.plot(xx, yy)
    plt.show()

