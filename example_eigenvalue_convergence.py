import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from waves1d import *

from sandbox.gllTemp import *

# problem
left = 0
right = 1.2
extra = 0.2
eigenvalue = 6

# analysis
# nh = 8  # boundary fitted
nh = 240  # immersed

# method
ansatzType = 'Lagrange'
#ansatzType = 'Spline'
#ansatzType = 'InterpolatorySpline'
continuity = 'p-1'


if ansatzType == 'Lagrange':
    continuity = '0'

axLimitLowY = 1e-13
axLimitHighY = 1e-0

#mass = 'CON'
#mass = 'HRZ'
mass = 'RS'
dual = False


depth = 40
eigenvalueSearch = 'nearest'
#eigenvalueSearch = 'number'

wExact = (eigenvalue * np.pi) / (1.2 - 2 * extra)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 1e-16


domain = Domain(alpha)


def runStudy(n, p, spectral):
    # create grid and domain
    grid = UniformGrid(left, right, n)

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
        if dual:
            system = TripletSystem.fromOneQuadratureWithDualBasis(ansatz, quadratureK)
        else:
            system = TripletSystem.fromOneQuadrature(ansatz, quadratureK)

    system.findZeroDof()
    if len(system.zeroDof) > 0:
        print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

    # solve sparse
    M, K, MHRZ, MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)
    #print("asym: %e" % np.linalg.norm(K.toarray()-K.toarray().T))

    if mass == 'CON':
        #w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, M, which='SM', return_eigenvectors=False)
        w = scipy.linalg.eigvals(K.toarray(), M.toarray())
        #w1 = scipy.linalg.eigvals(K.toarray(), M.toarray())
        #w2 = scipy.linalg.eigvals(K.T.toarray(), M.toarray())
        #print("diff: %e" % np.linalg.norm(np.sort(w1)-np.sort(w2)))
        #print(np.sort(w1)-np.sort(w2))
        #w = w2
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
    #print(w[1:10]/wExact)

    dof = system.nDof()

    if eigenvalueSearch == 'nearest':
        wNum = find_nearest(w, wExact)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    return dof, np.real(wNum)


figure, ax = plt.subplots()
plt.rcParams['axes.titleweight'] = 'bold'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax.set_ylim(axLimitLowY, axLimitHighY)

title = ansatzType + ' C' + str(continuity)
title += ' ' + mass
title += ' d=' + str(extra)
title += ' ' + eigenvalueSearch
fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvalue_convergence_stabilized_1e-8/", title.replace(' ', '_'))

for p in [1, 2, 3, 4]:

    k = eval(continuity)
    k = max(0, min(k, p - 1))

    #nhh = nh  # boundary fitted
    nhh = int(nh/(p-k))  # immersed

    print("p = %d" % p)
    minws = [0] * nhh
    errors = [0] * nhh
    dofs = [0] * nhh

    for i in range(nhh):
        # n = int((12 / (p - k)) * 2 ** (0.2*i))
        n = int(12/(p-k))+i  # immersed
        #n = int(12/(p-k) * 1.5 ** (i))  # boundary fitted
        print("n = %d" % n)
        dofs[i], minws[i] = runStudy(n, p, False)
        errors[i] = np.abs(minws[i] - wExact) / wExact
        print("dof = %e, w = %e, e = %e" % (dofs[i], minws[i], errors[i]))

    ax.loglog(dofs, errors, '-o', label='p=' + str(p), color=colors[p - 1])
    writeColumnFile(fileBaseName + '_p=' + str(p) + '.dat', (dofs, errors))

ax.legend()
plt.title(title)

plt.xlabel('degrees of freedom')
plt.ylabel('relative error in sixth eigenvalue ')

fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvalue_convergence_stabilized_1e-8/", title.replace(' ', '_'))
#np.savetxt(fileBaseName + '.dat', res)

plt.savefig(fileBaseName + '.pdf')
plt.show()
