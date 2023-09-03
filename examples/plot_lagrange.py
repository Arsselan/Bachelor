import numpy as np
import matplotlib.pyplot as plt

from context import fem1d

k = 3
n = 2

left = 0
right = 1.2

# create mesh
print("Meshing...", flush=True)

t = np.linspace(left, right, n + 1)
points = np.linspace(-1, 1, k + 1);

# plot
print("Plotting...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlim(left - 0.1, right + 0.1)
ax1.set_ylim(-0.5, 1.5)

x = np.linspace(left, right, 500)

nPlot = 51
for i in range(n):
    x1 = t[i]
    x2 = t[i + 1]
    xx = np.linspace(x1, x2, nPlot)
    yy = np.zeros((nPlot, k + 1))
    dy = np.zeros((nPlot, k + 1))
    for j in range(len(xx)):
        ders = fem1d.lagrange.evaluateLagrangeBases(i, xx[j], points, 1, t)
        yy[j] = ders[0]
        dy[j] = ders[1]
    for j in range(k + 1):
        ax1.plot(xx, yy[:, j], '-x')
        ax2.plot(xx, dy[:, j], '-x')
    ax1.plot(xx, np.sum(yy, 1), '--')
    ax2.plot(xx, np.sum(dy, 1), '-')

ax1.plot(t, np.zeros(t.size), '-o')
ax2.plot(t, np.zeros(t.size), '--o')

plt.show()
