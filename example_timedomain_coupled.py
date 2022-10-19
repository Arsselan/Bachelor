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
boundary = 1.0 - 0.2 * 2.0/50


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
n = 50
p = 3
tMax = 2
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

systemF.findZeroDof(-1e60)
systemS.findZeroDof(-1e60)

print("Zero dof F: " + str(systemF.zeroDof))
print("Zero dof S: " + str(systemS.zeroDof))

nDofF = systemF.nDof()
nDofS = systemS.nDof()

M, K = createSparseMatrices(systemF, systemS)
C = createCouplingMatrix(systemF, systemS, [boundary])
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
fullU = np.zeros((nt + 1, ansatz.nDof() * 2))
fullUF = np.zeros((nt + 1, ansatz.nDof()))
fullUS = np.zeros((nt + 1, ansatz.nDof()))
evalUF = np.zeros((nt + 1, 1000))
evalUS = np.zeros((nt + 1, 1000))
evalUFG = np.zeros((nt + 1, 1000))
evalUSG = np.zeros((nt + 1, 1000))
evalUFV = np.zeros((nt + 1, 1000))
evalUSV = np.zeros((nt + 1, 1000))

nodesF = np.linspace(grid.left, boundary * 1.3, 1000)
IF = ansatz.interpolationMatrix(nodesF)
IFG = ansatz.interpolationMatrix(nodesF, 1)

nodesS = np.linspace(boundary * 0.7, grid.right, 1000)
IS = ansatz.interpolationMatrix(nodesS)
ISG = ansatz.interpolationMatrix(nodesS, 1)

for i in range(2, nt + 1):
    rhs = M * (2 * u[i - 1] - u[i - 2]) + dt / 2 * C * u[i - 2] + dt ** 2 * (F * source.ft(i * dt) - K * u[i - 1])
    u[i] = lu.solve(rhs)
    fullUF[i] = systemF.getFullVector(u[i][0:nDofF])
    fullUS[i] = systemS.getFullVector(u[i][nDofF:nDofF + nDofS])
    evalUF[i] = IF * fullUF[i]
    evalUS[i] = IS * fullUS[i]
    evalUFV[i - 1] = IF * (fullUF[i] - fullUF[i - 2]) / (2 * dt)
    evalUSV[i - 1] = IS * (fullUS[i] - fullUS[i - 2]) / (2 * dt)
    evalUFG[i] = IFG * fullUF[i]
    evalUSG[i] = ISG * fullUS[i]


# Plot animation
def postProcess():
    # plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams["figure.figsize"] = (12, 6)

    figure, ax = plt.subplots()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.set_xlim(grid.left, grid.right)
    ax.set_ylim(-2, 2)

    ax.plot([boundary, boundary], [-2, 2], '--', color='#000000', label='domain boundary')

    # line, = ax.plot(0, 0, label='conrrol points')
    # line.set_xdata(np.linspace(grid.left, grid.right, ansatz.nDof()))

    lineUF, = ax.plot(0, 0, '--', label='potential u (F)')
    lineUF.set_xdata(nodesF)

    lineUS, = ax.plot(0, 0, '-', label='displacement d (S)')
    lineUS.set_xdata(nodesS)

    lineUFG, = ax.plot(0, 0, '--', label='velocity -grad(u) (F)')
    lineUFG.set_xdata(nodesF)

    lineUSG, = ax.plot(0, 0, '-', label='pressure -grad(d) (S)')
    lineUSG.set_xdata(nodesS)

    lineUFV, = ax.plot(0, 0, '--', label='pressure du/dt (F)')
    lineUFV.set_xdata(nodesF)

    lineUSV, = ax.plot(0, 0, '-', label='velocity dd/dt (S)')
    lineUSV.set_xdata(nodesS)

    # ax.plot([0, xmax],[1, 1], '--b')
    # ax.plot([interface, interface],[-0.5, 2.1], '--r')

    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    title = 'Solution'
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('pressure / solution / velocity')

    animationSpeed = 1

    def prepareFrame(i):
        plt.title(title + " time %3.2e" % i)

        lineUF.set_ydata(evalUF[int(round(i / tMax * nt))])
        lineUS.set_ydata(evalUS[int(round(i / tMax * nt))])

        lineUFG.set_ydata(-0.5 * evalUFG[int(round(i / tMax * nt))] + 1)
        lineUSG.set_ydata(-0.5 * evalUSG[int(round(i / tMax * nt))] - 1)

        lineUFV.set_ydata(0.5 * evalUFV[int(round(i / tMax * nt))] - 1)
        lineUSV.set_ydata(0.5 * evalUSV[int(round(i / tMax * nt))] + 1)

    frames = np.linspace(0, tMax, round(tMax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000 / 60, repeat=False)

    prepareFrame(2)
    plt.savefig('results/time_domain_coupled.pdf')

    plt.show()


postProcess()

# import cProfile
# cProfile.run('runStudy(20, 3)')
