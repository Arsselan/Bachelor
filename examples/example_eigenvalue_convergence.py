import numpy as np
import matplotlib.pyplot as plt

from context import fem1d

config = fem1d.StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0.2,

    # method
    ansatzType='Lagrange',
    # ansatzType='InterpolatorySpline',
    n=12,
    p=3,
    # ansatzType='Spline',
    continuity='p-1',
    mass='HRZ',

    depth=35,
    stabilize=0,
    spectral=False,
    dual=False,
    smartQuadrature=False,

    source=fem1d.sources.NoSource()
  )

# study
eigenvalue = 6

if config.extra == 0.0:
    nh = 8  # boundary fitted
else:
    nh = 240  # immersed


# eigenvalueSearch = 'nearest'
eigenvalueSearch = 'number'

wExact = (eigenvalue * np.pi) / (1.2 - 2 * config.extra)


# title
title = config.ansatzType
title += ' ' + config.continuity
title += ' ' + config.mass
title += ' a=%2.1e' % config.stabilize
title += ' d=' + str(config.extra)
title += ' ' + eigenvalueSearch

fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/example_eigenvalue_convergence/", title.replace(' ', '_'))

# run
allValues = []
allErrors = []
allDofs = []
allPs = [1, 2, 3, 4]
for p in allPs:
    config.p = p

    k = eval(config.continuity)

    if config.extra == 0.0:
        nStudies = nh  # boundary fitted
    else:
        nStudies = int(nh / (config.p - k))  # immersed

    values = [0] * nStudies
    errors = [0] * nStudies
    dofs = [0] * nStudies
    for i in range(nStudies):

        if config.extra == 0.0:
            n = int(12/(p-k) * 1.5 ** i)  # boundary fitted
        else:
            n = int(12/(p-k))+i  # immersed

        print("p = %d, n = %d" % (p, n))

        config.n = n
        config.p = p
        study = fem1d.EigenvalueStudy(config)
        if config.ansatzType == 'Lagrange' and config.mass == 'RS':
            study.runDense(sort=True)
        else:
            study.runSparse(sort=True)

        # dofs[i] = study.M.shape[0]
        dofs[i] = study.system.nDof()

        values[i], idx = fem1d.findEigenvalue(study.w, eigenvalueSearch, eigenvalue, wExact)
        errors[i] = np.abs(values[i] - wExact) / wExact
        print("dof = %e, w = %e, e = %e" % (dofs[i], values[i], errors[i]))

    fem1d.writeColumnFile(fileBaseName + '_p=' + str(p) + '.dat', (dofs, errors))
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

