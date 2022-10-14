import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from waves1d import *

# problem
left = 0
right = 1.2

# method
#ansatzType = 'Lagrange'
ansatzType = 'Spline'
continuity = 'p-1'
mass = 'HRZ'
depth = 40
spectral = False

# analysis
n = 12
axLimitY = 500

if ansatzType == 'Lagrange':
    continuity = '0'

if ansatzType == 'Spline':
    spectral = False
    if mass != 'CON' and continuity=='p-1':
        axLimitY = 50


def runStudy(p, extra):
    # create grid and domain
    grid = UniformGrid(left, right, n)

    def alpha(x):
        if left + extra <= x <= right - extra:
            return 1
        return 0

    domain = Domain(alpha)

    # create ansatz and quadrature points
    ansatz = createAnsatz(ansatzType, continuity, p, grid)

    # gaussPointsM = gll.computeGllPoints(p + 1)
    gaussPointsM = GLL(p + 1)
    quadratureM = SpaceTreeQuadrature(grid, gaussPointsM, domain, depth)

    gaussPointsK = np.polynomial.legendre.leggauss(p + 1)
    quadratureK = SpaceTreeQuadrature(grid, gaussPointsK, domain, depth)

    # create system
    if spectral:
        system = TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK)
    else:
        system = TripletSystem.fromOneQuadrature(ansatz, quadratureK)

    system.findZeroDof(0.0, [])
    if len(system.zeroDof) > 0:
        print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

    # solve sparse
    # M, K = system.createSparseMatrices()
    # w = scipy.sparse.linalg.eigs(K, K.shape[0]-2, M.toarray(), which='LM', return_eigenvectors=False)

    # solve dense
    # M, K = system.createSparseMatrices()
    M, K, MHRZ, MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)

    if mass == 'CON':
        # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, M, which='SM', return_eigenvectors=False)
        w = scipy.linalg.eigvals(K.toarray(), M.toarray())
    elif mass == 'HRZ':
        # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MHRZ, which='SM', return_eigenvectors=False)
        w = scipy.linalg.eigvals(K.toarray(), MHRZ.toarray())
    elif mass == 'RS':
        # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MRS, which='SM', return_eigenvectors=False)
        w = scipy.linalg.eigvals(K.toarray(), MRS.toarray())
    else:
        print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

    w = np.sqrt(np.abs(w))
    w = np.sort(w)

    return max(w)


figure, ax = plt.subplots()
ax.set_ylim(5, axLimitY)
for p in range(4):
    ne = 11
    extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
        0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]
    ne = len(extras)
    maxw = [0] * ne
    for i in range(ne):
        maxw[i] = runStudy(p + 1, extras[i])
        print("e = %e, wmax = %e" % (extras[i], maxw[i]))
    ax.plot(extras, maxw, '-o', label='p=' + str(p + 1))

ax.legend()

plt.rcParams['axes.titleweight'] = 'bold'

title = ansatzType + ' C' + str(continuity)
title += ' ' + mass
plt.title(title)

plt.xlabel('ficticious domain size')
plt.ylabel('largest eigenvalue')

plt.savefig(title.replace(' ', '_') + '.pdf')
plt.show()
