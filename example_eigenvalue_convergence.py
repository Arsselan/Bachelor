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
    mass='CON',

    depth=40,
    stabilize=0.0,
    spectral=False,
    dual=False,
  )

# study
eigenvalue = 6
nh = 40

eigenvalueSearch = 'nearest'
wExact = (eigenvalue * np.pi) / (1.2 - 2 * config.extra)


# title
title = config.ansatzType
title += ' ' + config.mass
title += ' a=%2.1e' % config.stabilize
title += ' d=' + str(config.extra)
title += ' ' + eigenvalueSearch
fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvalue_convergence_stabilized_1e-8/", title.replace(' ', '_'))

# run
allMinW = []
allErrors = []
allDofs = []
for p in [1, 2, 3, 4]:
    config.p = p

    k = eval(config.continuity)

    #nhh = nh  # boundary fitted
    nStudies = int(nh / (config.p - k))  # immersed

    print("p = %d" % p)
    minws = [0] * nStudies
    errors = [0] * nStudies
    dofs = [0] * nStudies
    for i in range(nStudies):
        n = int(12/(p-k))+i  # immersed
        #n = int(12/(p-k) * 1.5 ** (i))  # boundary fitted
        print("n = %d" % n)

        config.n = n
        config.p = p
        study = EigenvalueStudy(config)
        study.runDense()

        # dofs[i] = study.M.shape[0]
        dofs[i] = study.system.nDof()

        minws[i] = findEigenvalue(study.w, eigenvalueSearch, eigenvalue, wExact)
        errors[i] = np.abs(minws[i] - wExact) / wExact
        print("dof = %e, w = %e, e = %e" % (dofs[i], minws[i], errors[i]))

    writeColumnFile(fileBaseName + '_p=' + str(p) + '.dat', (dofs, errors))
    allMinW.append(minws)
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

    for p in [1, 2, 3, 4]:
        ax.loglog(allDofs[p-1], allErrors[p-1], '-o', label='p=' + str(p), color=colors[p - 1])

    ax.legend()
    plt.title(title)

    plt.xlabel('degrees of freedom')
    plt.ylabel('relative error in sixth eigenvalue ')

    fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvalue_convergence/", title.replace(' ', '_'))
    #np.savetxt(fileBaseName + '.dat', res)

    plt.savefig(fileBaseName + '.pdf')
    plt.show()

postProcess()