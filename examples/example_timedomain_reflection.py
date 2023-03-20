import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

import sources
from fem1d.studies import *

if 'config' not in locals():
    config = StudyConfig(
        # problem
        left=0,
        right=1.2,
        extra=0.8*1.2/120,

        # method
        ansatzType='Lagrange',
        # ansatzType = 'InterpolatorySpline',
        n=120,
        p=2,
        # ansatzType='Spline',
        continuity='p-1',
        mass='CON',

        depth=35,
        spectral=False,
        dual=False,
        stabilize=0,
        smartQuadrature=False,
        source=sources.NoSource()
    )

L = config.right - 2*config.extra
tMax = L
#nt = 1200*20
nt = 120000
dt = tMax / nt

# create study
temp = config.extra
for i in range(0):
    config.extra = i / 10 * 1.2/120
    print("extra = %e" % config.extra)
    study = EigenvalueStudy(config)
config.extra = temp
study = EigenvalueStudy(config)

# compute critical time step size
w = study.computeLargestEigenvalueSparse()
critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)
dt = correctTimeStepSize(dt, tMax, critDeltaT)
print("Corrected time step size is %e" % dt)

# solve sparse
u0, u1 = sources.applyGaussianInitialConditions(study.ansatz, dt, -0.6, config.stabilize)
evalNodes = np.linspace(study.grid.left + config.extra, study.grid.right - config.extra, study.ansatz.nDof())
u, fullU, evalU, iMat = study.runCentralDifferenceMethod(dt, nt, u0, u1, evalNodes)


def postProcess(animationSpeed=4):
    postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed)


def getResults():
    error = np.linalg.norm(evalU[1] - evalU[-1])
    return w, error
