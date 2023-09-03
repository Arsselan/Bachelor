import numpy as np
import matplotlib.pyplot as plt

from context import fem1d


p = 3
n = 2

left = 0
right = 1.0

# create mesh
print("Meshing...", flush=True)

# Create grid and ansatz
grid = fem1d.UniformGrid(left, right, n)
ansatz = fem1d.createAnsatz('Spline', 'p-1', p, grid)
ansatzLagrange = fem1d.createAnsatz('Lagrange', '0', p, grid)
t = ansatz.knots

# Greville points
ng = len(t) - p - 1
g = np.zeros(ng)
for i in range(ng):
    g[i] = np.sum(t[i+1:i+p+1]) / p

#g = np.linspace(left, right, ng)

#t[1] = 0.1
#t[-2] = 1.1

# Transformation matrix
nC = len(t)-p-1
T = ansatz.interpolationMatrix(g, 0).toarray().T
#T[T < 1e-30] = 0
invT = np.linalg.inv(T)


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

ax2.set_title("Transformed basis")
ax2.set_ylim(-0.5, 1.5)

ax3.set_title("Transformed basis squared + 1e-15")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

nPoints = 50
nElements = len(t) - 2 * p - 1
dataSplines = np.ndarray((nPoints * nElements, 1 + nC))
dataLagrange = np.ndarray((nPoints * nElements, 1 + (p+1) * nElements))

dataFunc = np.ndarray((nPoints * nElements, 1 + nC))
dataDeri = np.ndarray((nPoints * nElements, 1 + nC))
for i in range(nElements):
    x1 = t[p + i]
    x2 = t[p + i + 1]
    xx = np.linspace(x1+1e-10, x2-1e-10, nPoints)
    yy = np.zeros((nPoints, p + 1))
    yyLagrange = np.zeros((nPoints, p + 1))
    dy = np.zeros((nPoints, p + 1))
    tyy = np.zeros((nPoints, nC))
    tdy = np.zeros((nPoints, nC))
    for j in range(len(xx)):
        ders = fem1d.bspline.evaluateBSplineBases(p + i, xx[j], p, 1, t)
        yy[j] = ders[0]
        dy[j] = ders[1]
        invTAB = invT[:, ansatz.locationMap(i)]
        tyy[j] = invTAB.dot(yy[j])
        tdy[j] = invTAB.dot(dy[j])
        dersLagrange = ansatzLagrange.evaluate(xx[j], p, i)
        yyLagrange[j] = dersLagrange[0]

    dataLagrange[i * nPoints:(i + 1) * nPoints, 0] = xx
    dataSplines[i*nPoints:(i+1)*nPoints, 0] = xx
    for j in range(p + 1):
        ax1.plot(xx, yy[:, j], '-', color=colors[(i + j) % len(colors)])
        ax21.plot(xx, dy[:, j], '-', color=colors[(i + j) % len(colors)])
        dataSplines[i * nPoints:(i + 1) * nPoints, 1 + j + i] = yy[:, j]
        dataLagrange[i * nPoints:(i + 1) * nPoints, 1 + j + i*(p+1)] = yyLagrange[:, j]

    dataFunc[i*nPoints:(i+1)*nPoints, 0] = xx
    dataDeri[i * nPoints:(i + 1) * nPoints, 0] = xx
    for j in range(nC):
        dataFunc[i*nPoints:(i+1)*nPoints, 1+j] = tyy[:, j]
        dataDeri[i * nPoints:(i + 1) * nPoints, 1 + j] = tdy[:, j]
        ax2.plot(xx, tyy[:, j], '-', color=colors[j % len(colors)])
        ax22.plot(xx, tdy[:, j], '-', color=colors[j % len(colors)])
    for j in range(3):
        ax3.semilogy(xx, np.abs(tyy[:, j] * tyy[:, j])+1e-15, '-', color=colors[j % len(colors)])
        ax23.semilogy(xx, np.abs(tdy[:, j] * tdy[:, j])+1e-15, '-', color=colors[j % len(colors)])

ax1.plot(g, g*0, 'o', label='Greville')
ax1.plot(t, np.zeros(t.size), '-+', label='knots')

ax2.plot(g, g*0, 'o', label='Greville')
ax2.plot(t, np.zeros(t.size), '-+', label='knots')
ax2.plot(g, g*0+1, 'o', label='Greville')

plt.show()
fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/basis_transformation/", "basis")

np.savetxt(fileBaseName + "_splines.dat", dataSplines)
np.savetxt(fileBaseName + "_lagrange.dat", dataLagrange)
np.savetxt(fileBaseName + "_functions.dat", dataFunc)
np.savetxt(fileBaseName + "_derivatives.dat", dataDeri)
