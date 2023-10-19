import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
from scipy.fftpack import fft

from context import fem1d

outputDir = "results/crash_test_plastic_new_elastic/"

# create config if not already present (may be constructed from calling script)
if 'config' not in locals():
    config = fem1d.StudyConfig(
        # problem
        left=0,
        right=1.0,
        extra=0.05,

        # method
        #ansatzType='Lagrange',
        ansatzType='Spline',
        #ansatzType='InterpolatorySpline',
        n=12,
        p=2,

        continuity='p-1',
        mass='CON',

        depth=50,
        spectral=False,
        dual=False,
        stabilize=0,
        smartQuadrature=True,
        source=fem1d.sources.NoSource(),
        fixedDof=[]
    )

p = config.p
config.n = config.n * (eval(config.continuity) + 1)
print("Elements: %d" % config.n)

# create study
study = fem1d.EigenvalueStudy(config)
epsYield = 0.00004
hardening = 0.00005
E = 5e8
rho = 5e2
c = np.sqrt(E / rho)
print("Wave speed: %e" % c)
study.K *= E / rho

# time stuff
L = config.right - 2*config.extra
tMax = 0.1
nt = int(20000)
dt = tMax / nt

# compute critical time step size
w = study.computeLargestEigenvalueSparse()
critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)
dt, nt = fem1d.correctTimeStepSize(dt, tMax, critDeltaT)
print("Corrected time step size is %e" % dt)

#exit(1)

# apply initial conditions
vInitial = 5
print("Initial velocity: %e" % vInitial)
u0, u1 = fem1d.sources.applyConstantVelocityInitialConditions(study.ansatz, dt, -vInitial)

# define evaluation positions
left = study.grid.left + config.extra
right = study.grid.right - config.extra
#evalNodes = np.linspace(left, right, study.ansatz.nDof())
evalNodes = np.linspace(left, right, 96*4)
#evalNodes = np.array([left, left+1e-6, 0.5*(right-left), right-1e-6, right])

# external load
rightF = study.system.getReducedVector(fem1d.createNeumannVector(study.system, [evalNodes[-1]], [1], [1]))
leftF = study.system.getReducedVector(fem1d.createNeumannVector(study.system, [evalNodes[0]], [1], [1]))


def computeExternalLoad(time, currentU, previousU):
    # check penetration
    penaltyFactor = 1e6
    if currentU[0] < 0:  # and currentU[0] - previousU[0] < 0:
        leftFactor = penaltyFactor * (-currentU[0])
    else:
        leftFactor = 0
    return leftF * leftFactor


# solve
title = config.ansatzType + " n%d" % config.n + " p%d" % config.p + " " + config.mass + " dt%e" % dt
times, u, fullU, evalU, iMat, epsPla = fem1d.runCentralDifferenceMethodWeakContactImmersedLowMemoryPlastic(
        study, c, epsYield, hardening, dt, nt, u0, u1, evalNodes, computeExternalLoad, earliestLumping=0, dampingM=0)

# save
fileBaseName = fem1d.getFileBaseNameAndCreateDir(outputDir, title.replace(' ', '_'))
fem1d.writeColumnFile(fileBaseName + '.dat', (times, evalU[:, 0], evalU[:, -1]))

title2 = config.ansatzType + " n%d" % config.n + " p%d" % config.p + " " + config.mass + " dt%e" % dt + " final_disp"
fileBaseName2 = fem1d.getFileBaseNameAndCreateDir(outputDir, title2.replace(' ', '_'))
fem1d.writeColumnFile(fileBaseName2 + '.dat', (evalNodes, evalU[-1, :]))

qpIds = [0]
for iElement in range(study.config.n):
    qpIds.append(qpIds[-1] + len(epsPla[iElement]))

nQp = qpIds[-1]
qpData = np.ndarray((nQp, 2))
for iElement in range(study.config.n):
    qpRange = range(qpIds[iElement], qpIds[iElement+1])
    qpData[qpRange, 0] = study.quadratureK.points[iElement]
    qpData[qpRange, 1] = epsPla[iElement]
    for iPoint in qpRange:
        if qpData[iPoint, 0] < left or qpData[iPoint, 0] > right:
            print("Hi!")
            qpData[iPoint, 1] = 0.0
            epsPla[iElement][iPoint-qpRange.start] = 0.0
title2 = config.ansatzType + " n%d" % config.n + " p%d" % config.p + " " + config.mass + " dt%e" % dt + " eps"
fileBaseName2 = fem1d.getFileBaseNameAndCreateDir(outputDir, title2.replace(' ', '_'))
fem1d.writeColumnFile(fileBaseName2 + '.dat', (qpData[:, 0], qpData[:, 1]))


def plotBar():
    positions = []
    n = evalNodes.size
    for i in range(n):
        idx = int((evalNodes.size-1) * i / (n-1))
        #print(idx)
        positions.append(evalNodes[idx] + evalU[:, idx])
    fem1d.plot(times, positions)


def postProcess(animationSpeed=4, factor=1):
    fem1d.plot(times, [evalU[:, -1], evalU[:, 0]])
    fem1d.postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed, factor)
    figure, ax = plt.subplots()
    for iElement in range(study.config.n):
        ax.plot(study.quadratureK.points[iElement], epsPla[iElement], "-*", label=str(iElement))
    plt.show()


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

