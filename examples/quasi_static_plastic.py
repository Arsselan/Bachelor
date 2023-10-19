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
        right=1.0,
        extra=0,

        # method
        #ansatzType='Lagrange',
        ansatzType='Spline',
        #ansatzType='InterpolatorySpline',
        n=5,
        p=1,

        continuity='p-1',
        mass='RS',

        depth=25,
        spectral=False,
        dual=False,
        stabilize=0,
        smartQuadrature=True,
        source=fem1d.sources.NoSource(),
        fixedDof=[0]
    )

# create study
study = fem1d.EigenvalueStudy(config)
epsYield = 0.0002
hardening = 0.1
E = 5e8
rho = 5e2
c = np.sqrt(E / rho)
print("Wave speed: %e" % c)
study.K *= E / rho

# time stuff
L = config.right - 2*config.extra
tMax = 10.0 * 2.0
nt = 100000 * 1.0
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
u0, u1 = fem1d.sources.applyConstantVelocityInitialConditions(study.ansatz, dt, 0.0)

# define evaluation positions
left = study.grid.left + config.extra
right = study.grid.right - config.extra
evalNodes = np.linspace(left, right, study.ansatz.nDof())
#evalNodes = np.array([left, left+1e-6, 0.5*(right-left), right-1e-6, right])

# compute Neumann vectors
leftF = study.system.getReducedVector(fem1d.createNeumannVector(study.system, [evalNodes[0]], [1], [1]))
rightF = study.system.getReducedVector(fem1d.createNeumannVector(study.system, [evalNodes[-1]], [1], [1]))

leftFactor = 0


def computeRightFactor(time):
    loadTime = time * tMax / 20.0
    # rightFactor = 0.5 * (1 - np.cos(2 * np.pi * time / 5)) * 1e3 * 0.5
    rightFactor = - 0.5 * np.cos(2 * np.pi * loadTime / 5) * 1e3 * loadTime*loadTime * 0.01
    #rightFactor = 0.5 * np.sin(2 * np.pi * time / 5) * 1e3
    return rightFactor


def computeExternalLoad(time, currentU, previousU):
    return leftFactor * leftF + computeRightFactor(time) * rightF


# solve
title = config.ansatzType + " n%d" % config.n + " p%d" % config.p + " " + config.mass + " dt%e" % dt
times, u, fullU, evalU, iMat, epsPla = fem1d.runCentralDifferenceMethodWeakContactImmersedLowMemoryPlastic(
        study, c, epsYield, hardening, dt, nt, u0, u1, evalNodes, computeExternalLoad, dampingM=1e3)

# save
fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/quasi_static_plastic/", title.replace(' ', '_'))
fem1d.writeColumnFile(fileBaseName + '.dat', (times, evalU[:, 0], evalU[:, -1]))


def plotBar():
    positions = []
    n = evalNodes.size
    for i in range(n):
        idx = int((evalNodes.size-1) * i / (n-1))
        #print(idx)
        positions.append(evalNodes[idx] + evalU[:, idx])
    fem1d.plot(times, positions)


def postProcess(animationSpeed=4, factor=1):
    forces = times.copy()
    for i in range(forces.size):
        forces[i] = computeRightFactor(times[i])
        
    fem1d.plot(times, [evalU[:, -1], evalU[:, 0]], ["Disp. right", "Disp. left"], ["time", "displacement"])
    fem1d.plot(times, [forces], ["Force"], ["time", "force"])
    
    fem1d.plot(evalU[:, -1], [forces], ["Force"], ["displacement", "force"])
    fem1d.postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed, factor)
    figure, ax = plt.subplots()
    nqp = study.config.p + 1
    qpData = np.ndarray((study.config.n*nqp, 2))
    for iElement in range(study.config.n):
        qpData[iElement * nqp:(iElement+1) * nqp, 0] = study.quadratureK.points[iElement]
        qpData[iElement * nqp:(iElement + 1) * nqp, 1] = epsPla[iElement]
        ax.plot(study.quadratureK.points[iElement], epsPla[iElement], "-", label=str(iElement))
    title2 = config.ansatzType + " n%d" % config.n + " p%d" % config.p + " " + config.mass + " dt%e" % dt + " eps"
    fileBaseName2 = fem1d.getFileBaseNameAndCreateDir("results/quasi_static_plastic/", title2.replace(' ', '_'))
    fem1d.writeColumnFile(fileBaseName2 + '.dat', (qpData[:, 0], qpData[:, 1]))
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

