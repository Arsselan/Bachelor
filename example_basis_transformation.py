import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from scipy.interpolate import BSpline
import bspline
from waves1d import *


p = 2
n = 3

left = 0
right = 1.0

# create mesh
print("Meshing...", flush=True)

# Create grid and ansatz
grid = UniformGrid(left, right, n)
ansatz = createAnsatz('Spline', 'p-1', p, grid)
t = ansatz.knots

# Greville points
ng = len(t) - p - 1
g = np.zeros(ng)
for i in range(ng):
    g[i] = np.sum(t[i+1:i+p+1]) / p

#t[1] = 0.1
#t[-2] = 1.1

# Transformation matrix
nC = len(t)-p-1
T = ansatz.interpolationMatrix(g, 0).toarray().T
#T[T < 1e-30] = 0
invT = np.linalg.inv(T)


# plot
print("Plotting...", flush=True)
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_xlim(left - 0.1, right + 0.1)
ax1.set_ylim(-0.5, 1.5)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

nPoints = 50
for i in range(len(t) - 2 * p - 1):
    x1 = t[p + i]
    x2 = t[p + i + 1]
    xx = np.linspace(x1, x2, nPoints)
    yy = np.zeros((nPoints, p + 1))
    dy = np.zeros((nPoints, p + 1))
    tyy = np.zeros((nPoints, nC))
    for j in range(len(xx)):
        ders = bspline.evaluateBSplineBases(p + i, xx[j], p, 1, t)
        yy[j] = ders[0]
        dy[j] = ders[1]
        #tyy[j] = invT.T[i:i+p+1, i:i+p+1].dot(yy[j])
        #tyy[j] = np.zeros(p+1)
        #TAB = T[i:i + p + 1, i:i + p + 1]
        invTAB = invT[:, ansatz.locationMap(i)]
        tyy[j] = invTAB.dot(yy[j])
    for j in range(p + 1):
        ax1.plot(xx, yy[:, j], '-')
        #ax2.plot(xx, dy[:, j], '-')
    for j in range(nC):
        ax2.plot(xx, tyy[:, j], '-', color=colors[j])

ax1.plot(g, g*0, 'o', label='Greville')
ax1.plot(t, np.zeros(t.size), '-+', label='knots')

ax2.plot(g, g*0, 'o', label='Greville')
ax2.plot(t, np.zeros(t.size), '-+', label='knots')
ax2.plot(g, g*0+1, 'o', label='Greville')

plt.show()
