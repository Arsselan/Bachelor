import numpy as np
from context import fem1d

if 'config' not in locals():
    config = fem1d.StudyConfig(
        # problem
        left=0,
        right=1.2,
        extra=0.8*1.2/120,
        #extra=0.01495,

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

        source=fem1d.sources.NoSource(),

        eigenvalueStabilizationM=1e-4
    )

if 0:
    config.ansatzType = 'Lagrange'
    config.p = 4
    config.mass = 'HRZ'
    config.n = 51
    config.extra = 0.2

if 0:
    config.ansatzType = 'Spline'
    config.p = 4
    config.mass = 'RS'
    config.n = 238
    config.extra = 0.2


L = config.right - 2*config.extra
tMax = L
nt = 1200*20
if config.p == 30:
    nt = 12000*10*2
# nt = int(tMax / 8e-6)
# nt = 1
dt = tMax / nt

# create study
study = fem1d.EigenvalueStudy(config)

# compute critical time step size
w = study.computeLargestEigenvalueSparse()
critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)
dt, nt = fem1d.correctTimeStepSize(dt, tMax, critDeltaT)
print("Corrected time step size is %e" % dt)

# solve sparse
u0, u1 = fem1d.sources.applyGaussianInitialConditions(study.ansatz, dt, -0.6, config.stabilize)
evalNodes = np.linspace(study.grid.left + config.extra, study.grid.right - config.extra, study.ansatz.nDof())
u, fullU, evalU, iMat = fem1d.runCentralDifferenceMethod(study, dt, nt, u0, u1, evalNodes)
#u, fullU, evalU, iMat = study.runCentralDifferenceMethodLowMemory(dt, nt, u0, u1, evalNodes, [-dt, 0.0, tMax])


def postProcess(animationSpeed=4):
    fem1d.postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed)


def getResults():
    error = np.linalg.norm(evalU[1] - evalU[-1])
    return w, error, tMax, dt, nt


def saveSnapshots():
    data = np.zeros((evalNodes.size, 5))
    data[:, 0] = evalNodes
    data[:, 1] = evalU[1]
    data[:, 2] = evalU[int(nt/2)]
    data[:, 3] = evalU[int(3*nt/4)]
    data[:, 4] = evalU[-1]

    np.savetxt("wave_reflection_lagrange_p5_n51_dof205.dat", data)
    #dataClipped = np.clip(data, -0.1, 2.1)
    #np.savetxt("wave_reflection_lagrange_p3_snapshots_clipped_dt8e-6.dat", dataClipped)

    if dt == 8e-6:
        np.savetxt("wave_reflection_lagrange_p3_snapshots_dt8e-6.dat", data)
        dataClipped = np.clip(data, -0.1, 2.1)
        np.savetxt("wave_reflection_lagrange_p3_snapshots_clipped_dt8e-6.dat", dataClipped)
