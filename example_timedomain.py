import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline

import sources

from waves1d import *
from progress import *

# problem
left = 0
right = 2.0
extra = 0.0

# method
ansatzType = 'Lagrange'
# ansatzType = 'Spline'
continuity = '0'

depth = 40
p = 4
n = 25

tMax = 10
nt = 1000
dt = tMax / nt

# create grid and domain
grid = UniformGrid(left, right, n)
rightBoundary = right - extra - grid.elementSize * 15
L = rightBoundary
pi = np.pi


def alpha(x):
    if left + extra <= x <= rightBoundary:
        return 1.0
    return 1e-20


domain = Domain(alpha)

source = sources.RicklersWavelet(1.0, alpha)

# create ansatz and quadrature
ansatz = createAnsatz(ansatzType, continuity, p, grid)
gaussPoints = np.polynomial.legendre.leggauss(p + 1)
quadrature = SpaceTreeQuadrature(grid, gaussPoints, domain, depth)

# create system
system = TripletSystem.fromOneQuadrature(ansatz, quadrature, source.fx)
system.findZeroDof(-1e60)
M, K, MHRZ = system.createSparseMatrices(returnHRZ=True)
F = system.getReducedVector(system.F)
M = MHRZ

# compute critical time step size
w = scipy.sparse.linalg.eigs(K, 1, M.toarray(), which='LM', return_eigenvectors=False)
w = np.sqrt(w[0] + 0j)

critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)

dt = critDeltaT * 0.1
nt = int(tMax / dt + 0.5)
dt = tMax / nt
print("Corrected time step size is %e" % dt)

# solve sparse
factorized = scipy.sparse.linalg.splu(M)

print("Time integration ... ", flush=True)

u = np.zeros((nt + 1, M.shape[0]))
fullU = np.zeros((nt + 1, ansatz.nDof()))
evalU = 0 * fullU

nodes = np.linspace(grid.left, grid.right, ansatz.nDof())
I = ansatz.interpolationMatrix(nodes)

nodes2 = np.linspace(grid.left, rightBoundary, 1000)
I2 = ansatz.interpolationMatrix(nodes2)

# printProgressBar(0, nt+1, prefix = 'Progress:', suffix = 'Complete', length = 50)

errorSum = 0
for i in range(2, nt + 1):
    u[i] = factorized.solve(M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (F * source.ft(i * dt) - K * u[i - 1]))
    fullU[i] = system.getFullVector(u[i])
    evalU[i] = I * fullU[i]
    evalU2 = I2 * fullU[i]
    errorSum += dt * np.linalg.norm((evalU2 - source.uxt(nodes2, (i + 1) * dt)) / system.nDof())

    # if i % int(nt / 100) == 0:
    #    print( np.linalg.norm(evalU[i] - uxt(nodes, i*dt )) )
    # printProgressBar(i, nt + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

# evalU2 = nodes2 * 0
# for j in range(ansatz.nDof()):
#    evalU2[j] = ansatz.interpolate( nodes2[j], fullU[i] )
print("Error: %e " % errorSum)


# Plot animation
def postProcess():
    figure, ax = plt.subplots()
    ax.set_xlim(grid.left, grid.right)
    ax.set_ylim(-2, 2)

    ax.plot([rightBoundary, rightBoundary], [-0.1, 0.1], '--', label='domain boundary')

    line, = ax.plot(0, 0, label='conrrol points')
    line.set_xdata(np.linspace(grid.left, grid.right, ansatz.nDof()))

    line2, = ax.plot(0, 0, label='numerical')
    line2.set_xdata(nodes)

    line3, = ax.plot(0, 0, '--', label='analytical')
    line3.set_xdata(nodes)

    # ax.plot([0, xmax],[1, 1], '--b')
    # ax.plot([interface, interface],[-0.5, 2.1], '--r')

    ax.legend()

    plt.rcParams['axes.titleweight'] = 'bold'
    title = 'Solution'
    plt.title(title)
    plt.xlabel('solution')
    plt.ylabel('x')

    animationSpeed = 1

    def prepareFrame(i):
        plt.title(title + " time %3.2e" % i)
        line.set_ydata(fullU[int(round(i / tMax * nt))])
        line2.set_ydata(evalU[int(round(i / tMax * nt))])
        line3.set_ydata(source.uxt(nodes, i + dt))
        return line,

    frames = np.linspace(0, tMax, round(tMax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000 / 60, repeat=False)

    plt.show()


postProcess()
