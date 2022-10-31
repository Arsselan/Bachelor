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
extra = 0.2
eigenvalue = 6

# analysis
#nh = 10
nh = 450

# method
#ansatzType = 'Lagrange'
ansatzType = 'Spline'
#ansatzType = 'InterpolatorySpline'

continuity = 'p-1'

mass = 'RS'
#mass = 'HRZ'
#mass = 'RS'

depth = 30

eigenvalueSearch = 'nearest'
# eigenvalueSearch = 'number'

if ansatzType == 'Lagrange':
    continuity = '0'

L = 1.2 - 2 * extra
wExact = eigenvalue * np.pi / L

nodesEval = np.linspace(left + extra, right - extra, 5000)
vExact = np.cos(eigenvalue * np.pi / L * (nodesEval - extra))


# plot(nodesEval, vExact)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 1e-16


domain = Domain(alpha)


def runStudy(n, p, spectral):
    # create grid and domain
    grid = UniformGrid(left, right, n)

    # create ansatz
    ansatz = createAnsatz(ansatzType, continuity, p, grid)

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

    nEigen = K.shape[0] - 2 #min(K.shape[0] - 2, 2 * eigenvalue)
    if mass == 'CON':
        w, v = scipy.sparse.linalg.eigs(K, nEigen, M, which='SM', return_eigenvectors=True)
        #w, v = scipy.linalg.eig(K.toarray(), M.toarray(), right=True)
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

    #print(np.sort(w))

    dof = system.nDof()

    idx = eigenvalue
    if eigenvalueSearch == 'nearest':
        wNum = find_nearest(w, wExact)
        idx = find_nearest_index(w, wExact)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")
    wIndex = idx

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    iMatrix = ansatz.interpolationMatrix(nodesEval)

    if True:
        minIndex = 0
        minError = 1e10
        for idx in range(nEigen):
            if np.linalg.norm(np.imag(v[:, idx])) == 0:
                eVector = iMatrix * system.getFullVector(np.real(v[:, idx]))
                eVector = eVector / eVector[0]
                eVector *= np.linalg.norm(vExact) / np.linalg.norm(eVector)
                error = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
                if error < minError:
                    minError = error
                    minIndex = idx

        print("wIndex=%d, vIndex=%d, minError=%e" % (wIndex, minIndex, minError))
        idx = minIndex

        eVector = iMatrix * system.getFullVector(np.real(v[:, idx]))
    eVector = eVector / eVector[0]
    eVector *= np.linalg.norm(vExact) / np.linalg.norm(eVector)

    #plot(nodesEval, [eVector, vExact])

    return dof, np.real(wNum), eVector


title = ansatzType
title += ' ' + mass
title += ' d=' + str(extra)
title += ' ' + eigenvalueSearch
fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvector_convergence_matched_0.0/", title.replace(' ', '_'))


def plotVectorsAndValues():
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

        k = eval(continuity)
        k = max(0, min(k, p - 1))
        nhh = int(nh / (p - k))  # immersed

        print("p = %d" % p)
        minws = [0] * nhh
        valErrors = [0] * nhh
        vecErrors = [0] * nhh
        dofs = [0] * nhh

        continuity = 'p-1'
        if ansatzType == 'Lagrange':
            continuity = '0'
        k = eval(continuity)

        for i in range(nhh):
            n = int(12 / (p - k)) + i  # immersed
            # n = int(12/(p-k) * 1.5 ** (i))  # boundary fitted
            print("n = %d" % n)
            dofs[i], minws[i], eVector = runStudy(n, p, False)
            valErrors[i] = np.abs(minws[i] - wExact) / wExact
            vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
            print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], minws[i], vecErrors[i], valErrors[i]))
        ax[0].loglog(dofs, valErrors, '-o', label='p=' + str(p), color=colors[p - 1])
        ax[1].loglog(dofs, vecErrors, '-o', label='p=' + str(p), color=colors[p - 1])

        if False:
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
        if False:
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


#plotVectorsAndValues()


def plotVectors():
    figure, ax = plt.subplots(1, 1)
    plt.rcParams['axes.titleweight'] = 'bold'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if extra != 0:
        ax.set_ylim(1e-8, 10)
    else:
        ax.set_ylim(1e-11, 10)

    for p in [1,2, 3, 4]:
        continuity = 'p-1'
        if ansatzType == 'Lagrange':
            continuity = '0'
        k = eval(continuity)

        k = eval(continuity)
        k = max(0, min(k, p - 1))
        nhh = int(nh / (p - k))  # immersed

        print("p = %d" % p)
        minws = [0] * nhh
        valErrors = [0] * nhh
        vecErrors = [0] * nhh
        dofs = [0] * nhh

        for i in range(nhh):
            n = int(12 / (p - k)) + i  # immersed
            # n = int(12/(p-k) * 1.5 ** (i))  # boundary fitted
            print("n = %d" % n)
            dofs[i], minws[i], eVector = runStudy(n, p, False)
            valErrors[i] = np.abs(minws[i] - wExact) / wExact
            vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
            print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], minws[i], vecErrors[i], valErrors[i]))
        ax.loglog(dofs, vecErrors, '-o', label='p=' + str(p), color=colors[p - 1])
        writeColumnFile(fileBaseName + '_p=' + str(p) + '.dat', (dofs, vecErrors))

        if False:
            if ansatzType == 'Lagrange':
                for i in range(nh):
                    n = int((24 / (p - k)) * 1.5 ** i)
                    print("n = %d" % n)
                    dofs[i], minws[i], eVector = runStudy(n, p, True)
                    valErrors[i] = np.abs(minws[i] - wExact) / wExact
                    vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
                    print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], minws[i], vecErrors[i], valErrors[i]))
                ax.loglog(dofs, vecErrors, '--x', label='p=' + str(p) + ' spectral', color=colors[p - 1])
        if False:
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
                ax.loglog(dofs, vecErrors, '--x', label='p=' + str(p) + ' C0', color=colors[p - 1])

    ax.legend()

    ax.set_title(title)
    ax.set_xlabel('degrees of freedom')
    ax.set_ylabel('relative error in sixth eigenvector ')

    plt.savefig(fileBaseName + '.pdf')

    plt.show()


plotVectors()
