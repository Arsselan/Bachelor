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
    # ansatzType = 'Lagrange'
    # ansatzType = 'InterpolatorySpline'
    n=12,
    p=3,
    ansatzType='Spline',
    continuity='p-1',
    mass='HRZ',

    depth=35,
    stabilize=1e-8,
    spectral=False,
    dual=False,
  )

# study
eigenvalue = 6
nh = 240

eigenvalueSearch = 'nearest'
wExact = (eigenvalue * np.pi) / (1.2 - 2 * config.extra)


# title
title = config.ansatzType
title += ' ' + config.continuity
title += ' ' + config.mass
title += ' a=%2.1e' % config.stabilize
title += ' d=' + str(config.extra)
title += ' ' + eigenvalueSearch

fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvalue_convergence/", title.replace(' ', '_'))

# run
allValues = []
allErrors = []
allDofs = []
allPs = [1, 2, 3, 4]
for p in allPs:
    config.p = p

    k = eval(config.continuity)

    #nhh = nh  # boundary fitted
    nStudies = int(nh / (config.p - k))  # immersed

    values = [0] * nStudies
    errors = [0] * nStudies
    dofs = [0] * nStudies
    for i in range(nStudies):
        n = int(12/(p-k))+i  # immersed
        #n = int(12/(p-k) * 1.5 ** (i))  # boundary fitted
        print("p = %d, n = %d" % (p, n))

        config.n = n
        config.p = p
        study = EigenvalueStudy(config)
        if config.ansatzType == 'Lagrange' and config.mass == 'RS':
            study.runDense(sort=True)
        else:
            study.runSparse(sort=True)

        # dofs[i] = study.M.shape[0]
        dofs[i] = study.system.nDof()

        values[i] = findEigenvalue(study.w, eigenvalueSearch, eigenvalue, wExact)
        errors[i] = np.abs(values[i] - wExact) / wExact
        print("dof = %e, w = %e, e = %e" % (dofs[i], values[i], errors[i]))

    writeColumnFile(fileBaseName + '_p=' + str(p) + '.dat', (dofs, errors))
    allValues.append(values)
    allErrors.append(errors)
    allDofs.append(dofs)


def postProcess():
    # plot
    axLimitLowY = 1e-13
    axLimitHighY = 1e-0

    figure, ax = plt.subplots()
    plt.rcParams['axes.titleweight'] = 'bold'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax.set_ylim(axLimitLowY, axLimitHighY)

    iStudy = 0
    for p in allPs:
        ax.loglog(allDofs[iStudy], allErrors[iStudy], '-o', label='p=' + str(p), color=colors[p - 1])
        iStudy += 1

    ax.legend()
    plt.title(title)

    plt.xlabel('degrees of freedom')
    plt.ylabel('relative error in sixth eigenvalue ')

    plt.savefig(fileBaseName + '.pdf')
    plt.show()


postProcess()

