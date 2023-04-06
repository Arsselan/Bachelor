import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
from scipy.fftpack import fft

from context import fem1d

# create config if not already present (may be constructed from calling script)
if 'config' not in locals():
    config = fem1d.StudyConfig(
        # problem
        left=0,
        right=1.2,
        extra=0,

        # method
        ansatzType='Lagrange',
        #ansatzType='Spline',
        # ansatzType = 'InterpolatorySpline',
        n=100,
        p=2,

        continuity='p-1',
        mass='HRZ',

        depth=25,
        spectral=False,
        dual=False,
        stabilize=0,
        smartQuadrature=True,
        source=fem1d.sources.NoSource()
    )

L = config.right - 2*config.extra
tMax = L*10*2
nt = 120000
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

# apply initial conditions
u0, u1 = fem1d.sources.applyConstantVelocityInitialConditions(study.ansatz, dt, 0.1)

# define evaluation positions
# evalNodes = np.linspace(study.grid.left + config.extra, study.grid.right - config.extra, study.ansatz.nDof())

left = study.grid.left + config.extra
right = study.grid.right - config.extra
evalNodes = np.array([left, left+1e-6, 0.5*(right-left), right-1e-6, right])

# solve
title = config.ansatzType + " n%d" % config.n + " p%d" % config.p + " " + config.mass + " dt%e" % dt
penaltyFactor = 1e4
if penaltyFactor > 0:
    times, u, fullU, evalU, iMat = fem1d.runCentralDifferenceMethodWeakContactBoundaryFittedLowMemory(
        study, dt, nt, u0, u1, evalNodes, penaltyFactor)
    title = title + " pen%e" % penaltyFactor
else:
    times, u, fullU, evalU, iMat = fem1d.runCentralDifferenceMethodStrongContactBoundaryFittedLowMemory(
        study, dt, nt, u0, u1, evalNodes)

# save
fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/timedomain_impact/", title.replace(' ', '_'))
fem1d.writeColumnFile(fileBaseName + '.dat', (times, evalU[:, 0], evalU[:, -1]))

# evaluate maxima
max1 = np.max(evalU[0:int(nt/3), 0])
max2 = np.max(evalU[int(nt/3):int(2*nt/3), 0])
max3 = np.max(evalU[int(2*nt/3):int(nt), 0])
print("Maxima: %e, %e, %e" % (max1, max2, max3))

fem1d.plot(times, [evalU[:, -1], evalU[:, 0]])


def plotBar():
    positions = []
    n = evalNodes.size
    for i in range(n):
        idx = int((evalNodes.size-1) * i / (n-1))
        print(idx)
        positions.append(evalNodes[idx] + evalU[:, idx])
    fem1d.plot(times, positions)


def postProcess(animationSpeed=4):
    fem1d.postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed)


def computeSpectrum():
    ps = np.abs(np.fft.fft(evalU[:, 0]))
    freqs = np.fft.fftfreq(evalU[:, 0].size, dt)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], 2.0 / nt * ps[idx], "-*")
    plt.show()


def computeScipySpectrum():
    uf = fft(evalU[:, 0])
    ff = np.linspace(0.0, 1.0 / (2.0 * dt), nt // 2)
    plt.semilogy(ff, 2.0 / nt * np.abs(uf[0:nt // 2]))

