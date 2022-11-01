import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from waves1d import *
from fem1d.studies import *

config = StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0.2,

    # method
    # ansatzType='Lagrange',
    # ansatzType = 'InterpolatorySpline',
    ansatzType='Lagrange',
    n=12,
    p=3,

    continuity='p-1',
    #mass='RS',
    mass='HRZ',
    #mass='RS',

    depth=35,
    stabilize=1e-8,
    spectral=False,
    dual=False
)

# study
eigenvalue = 6
# nh = 10
nRefinements = 450

eigenvalueSearch = 'nearest'
#eigenvalueSearch = 'number'


# analytical
L = 1.2 - 2 * config.extra
wExact = eigenvalue * np.pi / L

nodesEval = np.linspace(config.left + config.extra, config.right - config.extra, 5000)
vExact = np.cos(eigenvalue * np.pi / L * (nodesEval - config.extra))

# plot(nodesEval, vExact)


# title
title = config.ansatzType
title += ' ' + config.continuity
title += ' ' + config.mass
title += ' a=%2.1e' % config.stabilize
title += ' d=' + str(config.extra)
title += ' ' + eigenvalueSearch
fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvector_convergence/", title.replace(' ', '_'))

# run
allValues = []
allValErrors = []
allVecErrors = []
allDofs = []
allPs = [1, 2, 3, 4]
for p in allPs:
    config.p = p

    k = eval(config.continuity)
    k = max(0, min(k, p - 1))
    nStudies = int(nRefinements / (p - k))

    values = [0] * nStudies
    valErrors = [0] * nStudies
    vecErrors = [0] * nStudies
    dofs = [0] * nStudies
    for i in range(nStudies):
        n = int(12 / (p - k)) + i  # immersed
        # n = int(12/(p-k) * 1.5 ** (i))  # boundary fitted
        print("p = %d, n = %d" % (p, n))

        config.n = n
        study = EigenvalueStudy(config)

        if config.ansatzType == 'Lagrange' and config.mass == 'RS':
            study.runDense(computeEigenvectors=True, sort=True)
        else:
            study.runSparse(computeEigenvectors=True, sort=True)

        dofs[i] = study.system.nDof()

        values[i] = findEigenvalue(study.w, eigenvalueSearch, eigenvalue, wExact)
        valErrors[i] = np.abs(values[i] - wExact) / wExact

        iMatrix = study.ansatz.interpolationMatrix(nodesEval)
        eVector = findEigenvector(study.v, eigenvalueSearch, eigenvalue, iMatrix, study.system, vExact)
        vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)

        print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], values[i], vecErrors[i], valErrors[i]))

    writeColumnFile(fileBaseName + '_p=' + str(p) + '.dat', (dofs, vecErrors))
    allValues.append(values)
    allValErrors.append(valErrors)
    allVecErrors.append(vecErrors)
    allDofs.append(dofs)


def postProcess():
    figure, ax = plt.subplots(1, 1)
    plt.rcParams['axes.titleweight'] = 'bold'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if config.extra != 0:
        ax.set_ylim(1e-8, 10)
    else:
        ax.set_ylim(1e-11, 10)

    iStudy = 0
    for p in allPs:
        ax.loglog(allDofs[iStudy], allVecErrors[iStudy], '-o', label='p=' + str(p), color=colors[p - 1])
        iStudy += 1

    ax.legend()

    ax.set_title(title)
    ax.set_xlabel('degrees of freedom')
    ax.set_ylabel('relative error in sixth eigenvector ')

    plt.savefig(fileBaseName + '.pdf')
    plt.show()


postProcess()

