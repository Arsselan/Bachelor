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
extra = 0.0

rightBoundary = right - extra - 0 * 0.12345  # grid.elementSize * 5.5 * 1
L = rightBoundary
pi = np.pi

wx = 2 * pi / L * 10
wt = 2 * pi * 3
source = Manufactured1(wx, wt)

# method
#ansatzType = 'Spline'
#continuity = 'p-1'

ansatzType = 'Lagrange'
continuity = '0'

depth = 40


# plotSource(source, left, right, 1000)


def alpha(x):
    if left + extra <= x <= rightBoundary:
        return 1.0
    return 1e-20


domain = Domain(alpha)


def runStudy(n, p, spectral):
    tMax = 0.7
    nt = 1000
    dt = tMax / nt

    # create grid and domain
    grid = UniformGrid(left, right, n)

    # create ansatz and quadratures
    ansatz = createAnsatz(ansatzType, continuity, p, grid)

    gaussPointsK = np.polynomial.legendre.leggauss(p + 8)
    quadratureK = SpaceTreeQuadrature(grid, gaussPointsK, domain, depth)

    gaussPointsM = GLL(p + 1)
    quadratureM = SpaceTreeQuadrature(grid, gaussPointsM, domain, depth)

    # create system
    if spectral:
        system = TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK, source.fx)
    else:
        system = TripletSystem.fromOneQuadrature(ansatz, quadratureK, source.fx)

    system.findZeroDof(-1e60)
    print("Zero dof: " + str(system.zeroDof))

    # create matrices
    fullM, K, M = system.createSparseMatrices(returnRS=True)

    lumpError = np.linalg.norm(fullM.toarray() - M.toarray())
    print("Lump error: " + str(lumpError))

    F = system.getReducedVector(system.F)

    # compute critical time step size
    w = scipy.sparse.linalg.eigs(K, 1, fullM, which='LM', return_eigenvectors=False)
    w = np.sqrt(w[0] + 0j)

    critDeltaT = 2 / abs(w)
    print("Critical time step size is %e" % critDeltaT)
    print("Chosen time step size is %e" % dt)

    dt = 1e-5  # critDeltaT * 0.1
    nt = int(tMax / dt + 0.5)
    dt = tMax / nt
    print("Corrected time step size is %e" % dt)

    u = np.zeros((nt + 1, M.shape[0]))
    fullU = np.zeros((nt + 1, ansatz.nDof()))
    evalU = 0 * fullU

    nodes = np.linspace(grid.left, grid.right, ansatz.nDof())
    I = ansatz.interpolationMatrix(nodes)

    nodes2 = np.linspace(grid.left, rightBoundary, 1000)
    I2 = ansatz.interpolationMatrix(nodes2)

    # spectral radius
    invM = np.diag(1 / np.diag(M.toarray()))
    ident = np.eye(fullM.shape[0])
    iterMat = ident - np.matmul(invM, fullM.toarray())
    spectralRadius = np.max(np.linalg.eigvals(iterMat))
    print("Spectral radius is %e" % spectralRadius)

    # solve sparse
    print("Factorization ... ", flush=True)
    lu = scipy.sparse.linalg.splu(M)
    luFull = scipy.sparse.linalg.splu(fullM)

    print("Time integration ... ", flush=True)
    errorSum = 0
    for i in range(2, nt + 1):
        #print("t = %e" % (i*dt))
        rhs = fullM * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (F * source.ft(i * dt) - K * u[i - 1])
        #u[i] = lu.solve(rhs)
        u[i] = luFull.solve(rhs)

        if False:
            nCorr = 500
            omega = 1.0
            deltaU = u[i]
            for iCorr in range(nCorr):
                #print(np.linalg.norm(u[i] - uCompare))
                deltaUP = deltaU
                deltaU = lu.solve(rhs - fullM * u[i])
                deltaR = deltaU - deltaUP
                if iCorr > 0:
                    omega = - omega * deltaR.dot(deltaUP) / deltaR.dot(deltaR)
                if np.isnan(omega).any() or np.linalg.norm(deltaU)<1e-19:
                    break
                u[i] += omega*deltaU

        fullU[i] = system.getFullVector(u[i])
        evalU[i] = I * fullU[i]
        if i == nt:
            evalU2 = I2 * fullU[i]
            errorSum += dt*np.linalg.norm((evalU2 - source.uxt(nodes2, (i + 1) * dt)) / system.nDof())

    nDof = system.nDof()

    print("Dof: %d Error: %e " % (nDof, errorSum))

    return errorSum, nDof, dt


figure, ax = plt.subplots()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['axes.titleweight'] = 'bold'

nRef = 6
errors = [0] * nRef
dofs = [0] * nRef
dts = [0] * nRef
for p in [1, 2, 3, 4]:
    k = eval(continuity)
    k = max(0, min(k, p - 1))
    if ansatzType == 'Lagrange':
        k = 0
    print("p=%d" % p)
    for i in range(nRef):
        errors[i], dofs[i], dts[i] = runStudy(int((20 / (p - k)) * 1.5 ** (i + 2)), p, False)
    ax.loglog(dofs, errors, '-o', label='p=' + str(p), color=colors[p - 1])
    if ansatzType == 'Lagrange':
        print("spectral:")
        for i in range(nRef):
            errors[i], dofs[i], dts[i] = runStudy(int((20 / (p - k)) * 1.5 ** (i + 2)), p, True)
        ax.loglog(dofs, errors, '--x', label='p=' + str(p) + ' spectral', color=colors[p - 1])

ax.legend()

title = ansatzType + ' C' + str(continuity)
title += ' L=' + str(rightBoundary)
plt.title(title)

plt.xlabel('degrees of freedom')
plt.ylabel('time domain error')

plt.savefig('results/' + title.replace(' ', '_') + '.pdf')
plt.show()
