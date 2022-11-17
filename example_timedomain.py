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
right = 1.2
#extra = 0.099 * 0.1
#extra = 0.005 * 0.8

# method
ansatzType = 'Lagrange'
spectral = True

#ansatzType = 'InterpolatorySpline'
#ansatzType = 'Spline'
continuity = 'p-1'

mass = 'HRZ'

depth = 35
p = 2
n = 240


# corrections
if ansatzType == 'Lagrange':
    continuity = '0'

if ansatzType == 'Spline':
    spectral = False

k = eval(continuity)
n = int(n / (p - k))

# create grid and domain
grid = UniformGrid(left, right, n)
#extra = 0.8*grid.elementSize
L = grid.right - 2*extra - grid.left
pi = np.pi

tMax = L
nt = 120000
dt = tMax / nt


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 1e-8


domain = Domain(alpha)

#source = sources.RicklersWavelet(10.0, alpha)
source = sources.NoSource()

# create ansatz and quadrature
ansatz = createAnsatz(ansatzType, continuity, p, grid)
gaussPoints = np.polynomial.legendre.leggauss(p + 1)
quadrature = SpaceTreeQuadrature(grid, gaussPoints, domain, depth)

# create system
gaussPointsM = GLL(p + 1)
quadratureM = SpaceTreeQuadrature(grid, gaussPointsM, domain, depth)

gaussPointsK = np.polynomial.legendre.leggauss(p + 1)
quadratureK = SpaceTreeQuadrature(grid, gaussPointsK, domain, depth)

# create system
if spectral:
    system = TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK, source.fx)
else:
    system = TripletSystem.fromOneQuadrature(ansatz, quadratureK, source.fx)

system.findZeroDof(0)
if len(system.zeroDof) > 0:
    print("Found zero dof: " + str(system.zeroDof))

M, K, MHRZ, MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)
F = system.getReducedVector(system.F)

if mass == 'CON':
    Muse = M
elif mass == 'HRZ':
    Muse = MHRZ
elif mass == 'RS':
    Muse = MRS
else:
    print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

# compute critical time step size
w = scipy.sparse.linalg.eigs(K, 1, Muse.toarray(), which='LM', return_eigenvectors=False)
w = np.sqrt(w[0] + 0j)

critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)

if dt > critDeltaT * 0.9:
    dt = critDeltaT * 0.9
    nt = int(tMax / dt + 0.5)
    dt = tMax / nt

print("Corrected time step size is %e" % dt)

# solve sparse
factorized = scipy.sparse.linalg.splu(Muse)

# prepare result arrays
u = np.zeros((nt + 1, M.shape[0]))
fullU = np.zeros((nt + 1, ansatz.nDof()))
times = np.zeros(nt+1)

nodes = np.linspace(grid.left+extra, grid.right-extra, ansatz.nDof())
I = ansatz.interpolationMatrix(nodes)
evalU = 0 * fullU

# set initial conditions
times[0] = -dt
times[1] = 0.0
u0, u1 = sources.applyGaussianInitialConditions(ansatz, dt, -0.6, alpha)
u[0] = system.getReducedVector(u0)
u[1] = system.getReducedVector(u1)
for i in range(2):
    fullU[i] = system.getFullVector(u[i])
    evalU[i] = I * fullU[i]


# time integration
print("Time integration ... ", flush=True)

errorSum = 0
for i in range(2, nt + 1):
    times[i] = i*dt
    u[i] = factorized.solve(Muse * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (F * source.ft((i-1) * dt) - K * u[i - 1]))
    fullU[i] = system.getFullVector(u[i])
    evalU[i] = I * fullU[i]


# Plot animation
def postProcess(animationSpeed=4):
    figure, ax = plt.subplots()
    ax.set_xlim(grid.left, grid.right)
    ax.set_ylim(-2, 2)

    ax.plot([left+extra, left+extra], [-0.1, 0.1], '--', label='left boundary')
    ax.plot([right-extra, right-extra], [-0.1, 0.1], '--', label='right boundary')

    line, = ax.plot(0, 0, label='conrtol points')
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

    def prepareFrame(i):
        step = int(round(i / tMax * nt))
        plt.title(title + " time %3.2e step %d" % (i, step))
        line.set_ydata(fullU[step])
        line2.set_ydata(evalU[step])
        line3.set_ydata(source.uxt(nodes, i + dt))
        return line,

    frames = np.linspace(0, tMax, round(tMax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000 / 60, repeat=False)
    plt.show()


error = np.linalg.norm(evalU[1] - evalU[-1])
print("Error: %e" % error)


#postProcess(1)
