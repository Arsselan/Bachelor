import numpy as np
import matplotlib.pyplot as plt

from context import fem1d


config = fem1d.StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0.2,

    # method
    ansatzType='Spline',
    #ansatzType='InterpolatorySpline',
    #ansatzType='Lagrange',
    n=12,
    p=3,

    continuity='p-1',
    #mass='CON',
    #mass='HRZ',
    mass='RS',

    depth=35,
    stabilize=0,
    spectral=False,
    dual=False,
    smartQuadrature=True,

    source=fem1d.sources.NoSource()
)

# study
eigenvalue = 6

if config.extra == 0.0:
    nRefinements = 9  # boundary fitted
else:
    nRefinements = 450  # immersed

eigenvalueSearch = 'nearest'
# eigenvalueSearch = 'number'


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
fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/asymptotic_accuracy/", title.replace(' ', '_'))

# run
allValues = []
allValErrors = []
allVecErrors = []
allVecValChecks = []
allValNegative = []
allValComplex = []
allMinMass = []
allDofs = []
allFirstElementM = []
allCondM = []
allCondK = []

allPs = [4]
for p in allPs:
    config.p = p

    k = eval(config.continuity)
    k = max(0, min(k, p - 1))

    if config.extra == 0.0:
        nStudies = nRefinements
    else:
        nStudies = int(nRefinements / (p - k))

    #nStudies = 3

    values = [0] * nStudies
    valErrors = [0] * nStudies
    vecErrors = [0] * nStudies
    dofs = [0] * nStudies
    vecValChecks = [0] * nStudies
    valNegative = [0] * nStudies
    valComplex = [0] * nStudies
    minMass = [0] * nStudies
    firstElementM = [0] * nStudies
    condM = [0] * nStudies
    condK = [0] * nStudies

    for i in range(nStudies):

        if config.extra == 0.0:
            n = int(12/(p-k) * 1.5 ** i)  # boundary fitted
        else:
            n = int(12 / (p - k)) + i  # immersed

        n = 13 + i

        print("p = %d, n = %d" % (p, n))

        #if n < 90 or n > 100:
        #    continue

        config.n = n
        study = fem1d.EigenvalueStudy(config)

        #if config.ansatzType == 'Lagrange' and config.mass == 'RS':
        #    study.runDense(computeEigenvectors=True, sort=True)
        #else:
        #    study.runSparse(computeEigenvectors=True, sort=True)

        study.runDense(computeEigenvectors=True, sort=True)

        dofs[i] = study.system.nDof()

        values[i], wIdx = fem1d.findEigenvalue(study.w, eigenvalueSearch, eigenvalue, wExact)
        valErrors[i] = np.abs(values[i] - wExact) / wExact

        iMatrix = study.ansatz.interpolationMatrix(nodesEval)
        eVector, vIdx = fem1d.findEigenvector(study.v, eigenvalueSearch, eigenvalue, iMatrix, study.system, vExact)
        vecErrors[i] = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)

        # eVector2, vIdx2 = fem1d.findEigenvector(study.v, "number", eigenvalue, iMatrix, study.system, vExact)
        # eVector3, vIdx3 = fem1d.findEigenvector(study.v, "number", wIdx, iMatrix, study.system, vExact)
        # fem1d.plot(nodesEval, [vExact, eVector, eVector2, eVector3], ["exact", "nearest", "number", "nearest w"])

        MAT = (study.K.toarray() - study.w[wIdx]**2 * study.getMassMatrix().toarray())
        # MAT = (np.float32(study.K.toarray()) - study.w[wIdx]**2 * np.float32(study.getMassMatrix().toarray()))

        VEC = study.v[:, vIdx]

        vecValChecks[i] = np.linalg.norm(np.matmul(MAT, VEC))
        valNegative[i] = study.nNegative
        valComplex[i] = study.nComplex
        minMass[i] = study.system.minNonZeroMass
        firstElementM[i] = study.getMassMatrix().diagonal()[0]
        condM[i] = np.linalg.cond(study.getMassMatrix().toarray())
        condK[i] = np.linalg.cond(study.K.toarray())

        print("dof = %e, w = %e, eVec = %e, eVal = %e" % (dofs[i], values[i], vecErrors[i], valErrors[i]))

    # fileBaseName = fileBaseName + "_float32"
    fem1d.writeColumnFile(fileBaseName + '_p=' + str(p) + '_vec_errors.dat', (dofs, vecErrors))
    fem1d.writeColumnFile(fileBaseName + '_p=' + str(p) + '_val_errors.dat', (dofs, valErrors))
    fem1d.writeColumnFile(fileBaseName + '_p=' + str(p) + '_negative_complex.dat', (valNegative, valComplex))
    fem1d.writeColumnFile(fileBaseName + '_p=' + str(p) + '_checks.dat', (dofs, vecValChecks, minMass, firstElementM, condM, condK))

    # fill lists for all in one plot
    allValues.append(values)
    allValErrors.append(valErrors)
    allVecErrors.append(vecErrors)
    allDofs.append(dofs)
    allVecValChecks.append(vecValChecks)
    allValNegative.append(valNegative)
    allValComplex.append(valComplex)
    allMinMass.append(minMass)
    allFirstElementM.append(firstElementM)
    allCondM.append(condM)
    allCondK.append(condK)


def postProcess():
    figure, ax = plt.subplots(1, 1)
    plt.rcParams['axes.titleweight'] = 'bold'
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    #if config.extra != 0:
    #    ax.set_ylim(1e-8, 10)
    #else:
    #    ax.set_ylim(1e-11, 10)

    iStudy = 0
    for p in allPs:
        ax.loglog(allDofs[iStudy], allVecErrors[iStudy], '-o', label='p=' + str(p), color=colors[p - 1])
        ax.loglog(allDofs[iStudy], allVecValChecks[iStudy], '--x', label='check acc p=' + str(p), color=colors[p - 1 + 1])
        #ax.loglog(allDofs[iStudy], allValNegative[iStudy], '-.x', label='check neg p=' + str(p), color=colors[p - 1])
        #ax.loglog(allDofs[iStudy], allValComplex[iStudy], ':x', label='check com p=' + str(p), color=colors[p - 1])
        ax.loglog(allDofs[iStudy], allMinMass[iStudy], '-*', label='min mass p=' + str(p), color=colors[p - 1 + 2])
        ax.loglog(allDofs[iStudy], allFirstElementM[iStudy], '-*', label='first element p=' + str(p), color=colors[p - 1 + 3])
        #ax.loglog(allDofs[iStudy], float32data, '-*', label='float32 p=' + str(p), color=colors[p - 1 + 4])
        ax.loglog(allDofs[iStudy], allCondM[iStudy], '-*', label='cond M p=' + str(p), color=colors[p - 1 + 5])
        ax.loglog(allDofs[iStudy], allCondK[iStudy], '-*', label='cond K p=' + str(p), color=colors[p - 1 + 6])

        iStudy += 1

    ax.legend()

    ax.set_title(title)
    ax.set_xlabel('degrees of freedom')
    ax.set_ylabel('relative error in sixth eigenvector ')

    plt.savefig(fileBaseName + '.pdf')
    plt.show()


postProcess()

