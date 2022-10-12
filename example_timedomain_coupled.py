import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline

from waves1d import *
from sources import *
from progress import *

# problem
left = 0
right = 2.0
boundary = 1.0


def alphaF(x):
    if left <= x <= boundary:
        return 1.0
    return 1e-8


def alphaS(x):
    if boundary <= x <= right:
        return 1.0
    return 1e-8


source = RicklersWavelet(1.0, alphaF)

# method
# ansatzType = 'Spline'
ansatzType = 'Lagrange'
continuity = '0'
spaceTreeDepth = 40
n = 10
p = 1
tMax = 7
nt = 10000
dt = tMax / nt

# create grid and domains
grid = UniformGrid(left, right, n)
domainS = Domain(alphaS)
domainF = Domain(alphaF)

# create ansatz and quadratures
ansatz = createAnsatz(ansatzType, continuity, p, grid)

gaussPoints = np.polynomial.legendre.leggauss(p + 8)
quadratureF = SpaceTreeQuadrature(grid, gaussPoints, domainF, spaceTreeDepth)
quadratureS = SpaceTreeQuadrature(grid, gaussPoints, domainS, spaceTreeDepth)

# create system
systemF = TripletSystem.fromOneQuadrature(ansatz, quadratureF, source.fx)
systemS = TripletSystem.fromOneQuadrature(ansatz, quadratureS)

systemF.findZeroDof()
systemS.findZeroDof()

print("Zero dof F: " + str(systemF.zeroDof))
print("Zero dof S: " + str(systemS.zeroDof))

nDofF = systemF.nDof()
nDofS = systemS.nDof()

M, K, C = createSparseMatrices(systemF, systemS, boundary)
F = getReducedVector(systemF, systemS)

# compute critical time step size
w = scipy.sparse.linalg.eigs(K, 1, M + dt / 2 * C, which='LM', return_eigenvectors=False)
w = np.sqrt(w[0] + 0j)
critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)

# solve sparse
lu = scipy.sparse.linalg.splu(M + dt / 2 * C)

print("Time integration ... ", flush=True)

nDof = M.shape[0]
u = np.zeros((nt + 1, nDof))
fullU = np.zeros((nt + 1, ansatz.nDof()*2))
fullUF = np.zeros((nt + 1, ansatz.nDof()))
fullUS = np.zeros((nt + 1, ansatz.nDof()))
evalUF = np.zeros((nt + 1, 1000))
evalUS = np.zeros((nt + 1, 1000))

nodesF = np.linspace(grid.left, boundary, 1000)
IF = ansatz.interpolationMatrix(nodesF)
nodesS = np.linspace(boundary, grid.right, 1000)
IS = ansatz.interpolationMatrix(nodesS)

for i in range(2, nt + 1):
    rhs = M * (2 * u[i - 1] - u[i - 2]) + dt/2 * C * u[i-2] + dt ** 2 * (F * source.ft(i * dt) - K * u[i - 1])
    u[i] = lu.solve(rhs)
    fullUF[i] = systemF.getFullVector(u[i][0:nDofF])
    fullUS[i] = systemS.getFullVector(u[i][nDofF:nDofF+nDofS])
    evalUF[i] = IF * fullUF[i]
    evalUS[i] = IS * fullUS[i]


# Plot animation
def postProcess():
    figure, ax = plt.subplots()
    ax.set_xlim(grid.left, grid.right)
    ax.set_ylim(-2, 2)

    ax.plot([boundary, boundary], [-0.1, 0.1], '--', label='domain boundary')

    #line, = ax.plot(0, 0, label='conrrol points')
    #line.set_xdata(np.linspace(grid.left, grid.right, ansatz.nDof()))

    line2, = ax.plot(0, 0, label='F')
    line2.set_xdata(nodesF)

    line3, = ax.plot(0, 0, '--', label='S')
    line3.set_xdata(nodesS)

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
        #line.set_ydata(fullU[int(round(i / tMax * nt))])
        line2.set_ydata(evalUF[int(round(i / tMax * nt))])
        line3.set_ydata(evalUS[int(round(i / tMax * nt))])

    frames = np.linspace(0, tMax, round(tMax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000 / 60, repeat=False)

    plt.show()


postProcess()

# import cProfile
# cProfile.run('runStudy(20, 3)')

