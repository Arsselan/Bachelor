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
extra = 0.19
eigenvalue = 6

# analysis
nh = 8

# method
ansatzType = 'Lagrange'
# ansatzType = 'Spline'

mass = 'CON'
# mass = 'HRZ'
# mass = 'RS'

depth = 40

eigenvalueSearch = 'nearest'
# eigenvalueSearch = 'number'

if ansatzType == 'Lagrange':
    continuity = '0'

L = 1.2 - 2 * extra
wExact = eigenvalue * np.pi / L

nodesEval = np.linspace(left + extra, right - extra, 1000)
vExact = np.cos(eigenvalue * np.pi / L * (nodesEval - extra))


# plot(nodesEval, vExact)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 1e-12


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

    nEigen = min(K.shape[0] - 2, 2 * eigenvalue)
    if mass == 'CON':
        # w, v = scipy.sparse.linalg.eigs(K, nEigen, M, which='SM', return_eigenvectors=True)
        w, v = scipy.linalg.eig(K.toarray(), M.toarray(), right=True)
    elif mass == 'HRZ':
        # w, v = scipy.sparse.linalg.eigs(K, nEigen, MHRZ, which='SM', return_eigenvectors=True)
        w, v = scipy.linalg.eig(K.toarray(), MHRZ.toarray(), right=True)
    elif mass == 'RS':
        # w, v = scipy.sparse.linalg.eigs(K, nEigen, MRS, which='SM', return_eigenvectors=True)
        w, v = scipy.linalg.eig(K.toarray(), MRS.toarray(), right=True)
    else:
        print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

    # compute frequencies
    w = np.real(w)
    w = np.abs(w)
    w = np.sqrt(w + 0j)
    # w = np.sort(w)

    dof = system.nDof()

    idx = eigenvalue
    if eigenvalueSearch == 'nearest':
        wNum = find_nearest(w, wExact)
        idx = find_nearest_index(w, wExact)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    iMatrix = ansatz.interpolationMatrix(nodesEval)
    eVector = iMatrix * v[:, idx]
    eVector = eVector / eVector[0]
    # plot(nodesEval, eVector)

    return dof, np.real(wNum), eVector


plt.rcParams["figure.figsize"] = (13, 6)

figure, ax = plt.subplots(1, 2)
plt.rcParams['axes.titleweight'] = 'bold'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

ax[0].set_xlim(20, 450)
ax[0].set_ylim(1e-12, 0.1)
if extra != 0:
    ax[1].set_ylim(1e-8, 0.1)
else:
    ax[1].set_ylim(1e-11, 0.1)

for p in [1, 2, 3, 4]:
    print("p = %d" % p)
    minws = [0] * nh
    valErrors = [0] * nh
    vecErrors = [0] * nh
    dofs = [0] * nh

    continuity = 'p-1'
    if ansatzType == 'Lagrange':
        continuity = '0'
    k = eval(continuity)

    for i in range(nh):
        n = int((24 / (p - k)) * 1.5 ** i)
        print("n = %d" % n)
        dofs[i], minws[i], eVector = runStudy(n, p, False)
        valErrors[i] = np.abs(minws[i] - wExact) / wExact
        vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
        print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], minws[i], vecErrors[i], valErrors[i]))
    ax[0].loglog(dofs, valErrors, '-o', label='p=' + str(p), color=colors[p - 1])
    ax[1].loglog(dofs, vecErrors, '-o', label='p=' + str(p), color=colors[p - 1])
    if ansatzType == 'Lagrange':
        for i in range(nh):
            n = int((24 / (p - k)) * 1.5 ** i)
            print("n = %d" % n)
            dofs[i], minws[i], eVector = runStudy(n, p, True)
            valErrors[i] = np.abs(minws[i] - wExact) / wExact
            vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
            print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], minws[i], vecErrors[i], valErrors[i]))
        ax[0].loglog(dofs, valErrors, '--x', label='p=' + str(p) + ' spectral', color=colors[p - 1])
        ax[1].loglog(dofs, vecErrors, '--x', label='p=' + str(p) + ' spectral', color=colors[p - 1])
    if ansatzType == 'Spline':
        continuity = '0'
        k = eval(continuity)
        for i in range(nh):
            n = int((24 / (p - k)) * 1.5 ** i)
            print("n = %d" % n)
            dofs[i], minws[i], eVector = runStudy(n, p, True)
            valErrors[i] = np.abs(minws[i] - wExact) / wExact
            vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
            print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], minws[i], vecErrors[i], valErrors[i]))
        ax[0].loglog(dofs, valErrors, '--x', label='p=' + str(p) + ' C0', color=colors[p - 1])
        ax[1].loglog(dofs, vecErrors, '--x', label='p=' + str(p) + ' C0', color=colors[p - 1])

ax[0].legend()
ax[1].legend()

title = ansatzType
title += ' ' + mass
title += ' d=' + str(extra)
title += ' ' + eigenvalueSearch
figure.suptitle(title)

ax[0].set_title('Eigenvalues')
ax[1].set_title('Eigenvectors')

ax[0].set_xlabel('degrees of freedom')
ax[1].set_xlabel('degrees of freedom')
ax[0].set_ylabel('relative error in sixth eigenvalue ')
ax[1].set_ylabel('relative error in sixth eigenvector ')

plt.savefig('results/eigen_' + title.replace(' ', '_') + '2.pdf')
plt.show()
