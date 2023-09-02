import numpy as np
import matplotlib.pyplot as plt

from context import fem1d

if 'config' not in locals():
    config = fem1d.StudyConfig(
        # problem
        left=0,
        right=1.2,
        extra=0.1,

        # method
        ansatzType='Lagrange',
        # ansatzType = 'InterpolatorySpline',
        # ansatzType='Spline',
        n=80,
        p=3,
        continuity='p-1',
        mass='CON',

        depth=15,
        spectral=False,
        dual=False,
        stabilize=0,
        smartQuadrature=True,

        eigenvalueStabilizationM=0.0,
        eigenvalueStabilizationK=0.0,

        source=fem1d.sources.NoSource()
    )

config.extra = 1.2/config.n*0.99
L = config.right - 2*config.extra
tMax = L
nt = 10000
dt = tMax / nt


def getResults():
    error = np.linalg.norm(evalU[1] - evalU[-1]) / np.linalg.norm(evalU[1])
    return config.eigenvalueStabilizationM, config.stabilize, w, error, tMax, dt, nt, critDeltaT


nStudies = 1
results = np.ndarray((nStudies, 8))
for i in range(nStudies):
    config.eigenvalueStabilizationM = 10 ** (-7 + 0.5 * i)

    #config.eigenvalueStabilization = False
    #config.stabilize = 10 ** (-7 + 0.5 * i)

    # create study
    study = fem1d.EigenvalueStudy(config)

    # compute critical time step size
    w = study.computeLargestEigenvalueSparse()
    critDeltaT = 2 / abs(w)
    print("Critical time step size is %e" % critDeltaT)
    print("Chosen time step size is %e" % dt)
    #dt, nt = fem1d.correctTimeStepSize(dt, tMax, critDeltaT)
    print("Corrected time step size is %e" % dt)

    # solve sparse
    u0, u1 = fem1d.sources.applyGaussianInitialConditions(study.ansatz, dt, -0.6, config.stabilize)
    evalNodes = np.linspace(study.grid.left + config.extra, study.grid.right - config.extra, study.ansatz.nDof())
    u, fullU, evalU, iMat = fem1d.runCentralDifferenceMethod(study, dt, nt, u0, u1, evalNodes)

    res = np.array(getResults())
    print(res, "\n")
    results[i][:] = res


def postProcess(animationSpeed=4):
    fem1d.postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed)


def saveSnapshots():
    data = np.zeros((evalNodes.size, 5))
    data[:, 0] = evalNodes
    data[:, 1] = evalU[1]
    data[:, 2] = evalU[int(nt/2)]
    data[:, 3] = evalU[int(3*nt/4)]
    data[:, 4] = evalU[-1]

    if dt == 8e-6:
        np.savetxt("wave_reflection_lagrange_p3_snapshots_dt8e-6.dat", data)
        dataClipped = np.clip(data, -0.1, 2.1)
        np.savetxt("wave_reflection_lagrange_p3_snapshots_clipped_dt8e-6.dat", dataClipped)
