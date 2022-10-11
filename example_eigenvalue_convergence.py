import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from waves1d import *

# problem
left = 0
right = 1.2
extra = 0.20
eigenvalue = 6

# analysis
nh = 6

# method
#ansatzType = 'Lagrange'
ansatzType = 'Spline'
continuity = 'p-1'
mass = 'CON'
# mass = 'HRZ'
#mass = 'RS'
depth = 40
eigenvalueSearch = 'nearest'
#eigenvalueSearch = 'number'

if ansatzType == 'Lagrange':
    continuity = '0'

wExact = (eigenvalue * np.pi) / (1.2 - 2 * extra)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 1e-20


domain = Domain(alpha)



def runStudy(n, p, spectral):
    # create grid and domain
    grid = UniformGrid(left, right, n)

    # create ansatz
    if ansatzType == 'Spline':
        k = eval(continuity)
        k = max(0, min(k, p - 1))
        ansatz = SplineAnsatz(grid, p, k)
    elif ansatzType == 'Lagrange':
        gllPoints = GLL(p + 1)
        # gllPoints[0][0] += 1e-16
        # gllPoints[0][-1] -=1e-16
        ansatz = LagrangeAnsatz(grid, gllPoints[0])
    else:
        print("Error! Choose ansatzType 'Spline' or 'Lagrange'")

    # print(ansatz.knots)

    # create quadrature points
    gaussPointsM = GLL(p + 1)
    quadratureM = SpaceTreeQuadrature(grid, gaussPointsM, domain, depth)

    gaussPointsK = np.polynomial.legendre.leggauss(p + 1)
    quadratureK = SpaceTreeQuadrature(grid, gaussPointsK, domain, depth)

    # create system
    if spectral:
        system = TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK)
    else:
        system = TripletSystem.fromOneQuadrature(ansatz, quadratureK)

    system.findZeroDof()
    if len(system.zeroDof) > 0:
        print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

    # solve sparse
    M, K, MHRZ, MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)

    if mass == 'CON':
        w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, M, which='SM', return_eigenvectors=False)
    elif mass == 'HRZ':
        w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MHRZ, which='SM', return_eigenvectors=False)
    elif mass == 'RS':
        w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MRS, which='SM', return_eigenvectors=False)
    else:
        print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

    # compute frequencies
    w = np.real(w)
    w = np.abs(w)
    w = np.sqrt(w + 0j)
    w = np.sort(w)

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

for p in [1, 2, 3, 4]:
    print("p = %d" % p)
    minws = [0] * nh
    errors = [0] * nh
    dofs = [0] * nh

    k = eval(continuity)
    k = max(0, min(k, p - 1))

    for i in range(nh):
        n = int((24 / (p - k)) * 1.5 ** i)
        print("n = %d" % n)
        dofs[i], minws[i] = runStudy(n, p, False)
        errors[i] = np.abs(minws[i] - wExact) / wExact
        print("dof = %e, w = %e, e = %e" % (dofs[i], minws[i], errors[i]))
    ax.loglog(dofs, errors, '-o', label='p=' + str(p), color=colors[p - 1])
    if ansatzType == 'Lagrange':
        for i in range(nh):
            n = int((24 / (p - k)) * 1.5 ** i)
            print("n = %d" % n)
            dofs[i], minws[i] = runStudy(n, p, True)
            errors[i] = np.abs(minws[i] - wExact) / wExact
            print("dof = %e, w = %e, e = %e" % (dofs[i], minws[i], errors[i]))
        ax.loglog(dofs, errors, '--x', label='p=' + str(p) + ' spectral', color=colors[p - 1])

ax.legend()

title = ansatzType + ' C' + str(continuity)
title += ' ' + mass
title += ' d=' + str(extra)
title += ' ' + eigenvalueSearch
plt.title(title)

plt.xlabel('degrees of freedom')
plt.ylabel('relative error in sixth eigenvalue ')

plt.savefig('results/' + title.replace(' ', '_') + '2.pdf')
plt.show()
