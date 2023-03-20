import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from scipy.interpolate import BSpline
import bspline
from waves1d import *


p = 3
n = 5

left = 0
right = 1.0

d = (right - left) / n

# create mesh
print("Meshing...", flush=True)

# Create grid and ansatz
grid = UniformGrid(left, right, n)
ansatz = createAnsatz('Spline', 'p-1', p, grid)
t = ansatz.knots

# plot
print("Plotting...", flush=True)
fig, ax = plt.subplots(2, 3)
fig.tight_layout(pad=0.5)

ax1 = ax[0][0]
ax2 = ax[0][1]
ax3 = ax[0][2]
ax21 = ax[1][0]
ax22 = ax[1][1]
ax23 = ax[1][2]

ax1.set_title("B-spline basis")
ax1.set_ylim(-0.5, 1.5)

ax2.set_title("Dual basis")
#ax2.set_ylim(-0.5, 1.5)

ax3.set_title("Transformed basis squared + 1e-15")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

nPoints = 50
for i in range(len(t) - 2 * p - 1):
    x1 = t[p + i]
    x2 = t[p + i + 1]
    xx = np.linspace(x1, x2, nPoints)
    yy = np.zeros((nPoints, p + 1))
    dy = np.zeros((nPoints, p + 1))
    tyy = np.zeros((nPoints, p + 1))
    tdy = np.zeros((nPoints, p + 1))
    Me = np.zeros((p+1, p + 1))
    diagMe = np.zeros((p + 1, p + 1))

    # compute mass matrix
    gaussPoints = np.polynomial.legendre.leggauss(p + 1)
    points = gaussPoints[0]
    weights = gaussPoints[1]
    for j in range(len(points)):
        # print("K p: " + str(j) + " " + str(pointsK[j]))
        x1 = i * d
        points[j] = x1 + d * 0.5 * (points[j] + 1)
        weights[j] = weights[j] * d / 2

        shapes = ansatz.evaluate(points[j], 1, i)
        N = np.asarray(shapes[0])
        Me += np.outer(N, N) * weights[j]
    for iEntry in range(Me.shape[0]):
        diagMe[iEntry, iEntry] = sum(Me[iEntry, :])

    # transformation matrix
    tMat = np.linalg.inv(Me).dot(diagMe)

    for j in range(len(xx)):
        ders = bspline.evaluateBSplineBases(p + i, xx[j], p, 1, t)
        yy[j] = ders[0]
        dy[j] = ders[1]
        tyy[j] = yy[j].dot(tMat)
        tdy[j] = dy[j].dot(tMat)
    for j in range(p + 1):
        ax1.plot(xx, yy[:, j], '-', color=colors[(i + j) % len(colors)])
        ax21.plot(xx, dy[:, j], '-', color=colors[(i + j) % len(colors)])
        ax2.plot(xx, tyy[:, j], '-', color=colors[(i + j) % len(colors)])
        ax22.plot(xx, tdy[:, j], '-', color=colors[(i + j) % len(colors)])
    for j in range(p+1):
        ax3.semilogy(xx, np.abs(tyy[:, j] * tyy[:, j])+1e-15, '-', color=colors[j % len(colors)])
        ax23.semilogy(xx, np.abs(tdy[:, j] * tdy[:, j])+1e-15, '-', color=colors[j % len(colors)])

ax1.plot(t, np.zeros(t.size), '-+', label='knots')
ax2.plot(t, np.zeros(t.size), '-+', label='knots')

plt.show()
