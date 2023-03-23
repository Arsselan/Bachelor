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
        #ansatzType = 'InterpolatorySpline',
        n=10000,
        p=1,

        continuity='p-1',
        mass='RS',

        depth=35,
        spectral=False,
        dual=False,
        stabilize=0,
        smartQuadrature=False,
        source=fem1d.sources.NoSource()
    )

L = config.right - 2*config.extra
tMax = L*10*2
nt = 1200000
#nt = 120000
dt = tMax / nt

# create study
study = fem1d.EigenvalueStudy(config)

# compute critical time step size
w = study.computeLargestEigenvalueSparse()
critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)
dt = fem1d.correctTimeStepSize(dt, tMax, critDeltaT)
print("Corrected time step size is %e" % dt)

# apply initial conditions
u0, u1 = fem1d.sources.applyConstantVelocityInitialConditions(study.ansatz, dt, 0.1)

#evalNodes = np.linspace(study.grid.left + config.extra, study.grid.right - config.extra, study.ansatz.nDof())

left = study.grid.left + config.extra
right = study.grid.right - config.extra
evalNodes = np.array([left, left+1e-6, 0.5*(right-left), right-1e-6, right])

# solve
times, u, fullU, evalU, iMat = study.runCentralDifferenceMethod4(dt, nt, u0, u1, evalNodes)

title = config.ansatzType + " n=%d" % config.n + " p=%d" % config.p + " " + config.mass
fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/example_timedomain_impact_reference/", title.replace(' ', '_'))
fem1d.writeColumnFile(fileBaseName + '.dat', (times, evalU[:, 0], evalU[:, -1]))

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


def getResults():
    error = np.linalg.norm(evalU[1] - evalU[-1])
    return w, error


def computeSpectrum():
    ps = np.abs(np.fft.fft(u[:, 1]))
    freqs = np.fft.fftfreq(u[:, 1].size, dt)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], 2.0 / nt * ps[idx])


def computeScipySpectrum():
    uf = fft(u[:, 1])
    ff = np.linspace(0.0, 1.0 / (2.0 * dt), nt // 2)
    plt.semilogy(ff, 2.0 / nt * np.abs(uf[0:nt // 2]))

