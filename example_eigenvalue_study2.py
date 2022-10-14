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
mass = 'RS'
depth = 40
spectral = False
eigenvalueSearch = 'nearest'
eigenvalue = 3

# analysis
nBase = 24
axLimitY = 25

if ansatzType == 'Lagrange':
    continuity = '0'

if ansatzType == 'Spline':
    spectral = False
    if mass != 'CON' and continuity == 'p-1':
        axLimitY = 50


def runStudy(p, extra):
    # create grid and domain
    grid = UniformGrid(left, right, n)

    def alpha(x):
        if left + extra <= x <= right - extra:
            return 1
        return 1e-10

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

    # compute frequencies
    w = np.real(w)
    w = np.abs(w)
    w = np.sqrt(w + 0j)
    w = np.sort(w)

    dof = system.nDof()

    if eigenvalueSearch == 'nearest':
        wExact = (eigenvalue * np.pi) / (1.2 - 2 * extra)
        wNum = find_nearest(w, wExact)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    return system.nDof(), np.real(wNum)


n = nBase
ne = 51
extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
    0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]
ne = len(extras)

# extras = list(np.array(extras) * 12 / n)
extras = list(np.array(extras) * nBase / n)

wExact = (eigenvalue * np.pi) / (1.2 - 2 * np.array(extras))

figure, ax = plt.subplots()
# ax.set_ylim(5, axLimitY)

ax.plot(extras, wExact, '-', label='reference', color='#000000')

for p in [1, 2, 3, 4]:
    maxw = [0] * ne
    for i in range(ne):
        k = eval(continuity)
        n = int(nBase / (p - k))
        nDof, maxw[i] = runStudy(p, extras[i])
        print("e = %e, wmax = %e" % (extras[i], maxw[i]))
    ax.plot(extras, maxw, '-o', label='n=' + str(n) + 'p=' + str(p) + ' dof=' + str(nDof))

ax.legend()

plt.rcParams['axes.titleweight'] = 'bold'

title = ansatzType + ' C' + str(continuity)
title += ' ' + mass
plt.title(title)

plt.xlabel('ficticious domain size')
plt.ylabel('largest eigenvalue')

plt.savefig(title.replace(' ', '_') + '.pdf')
plt.show()
