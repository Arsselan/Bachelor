import numpy as np
import os
import matplotlib.pyplot as plt
from context import fem1d
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def run(extraDelta = 0.0, params = [ 0.5e5, 0.5, 0.5e-5, 1e-10 ], frequency = 200, disablePlots = False):
    if 'config' not in locals():
        config = fem1d.StudyConfig(
            # problem
            left=0,
            right=0.02,
            extra=0.0,
            #extra=0.01495,

            # method
            ansatzType='Lagrange',
            # ansatzType = 'InterpolatorySpline',
            # ansatzType='Spline',
            n=8,
            p=2,
            continuity='p-1',
            mass='RS',

            depth=15,
            spectral=False,
            dual=False,
            stabilize=0,
            smartQuadrature=True,

            source=fem1d.sources.NoSource(),

            eigenvalueStabilizationM=0
        )

    # parameters
    config.density = 30.0

    config.elasticity = params[0]
    damping = list(params)
    damping.pop(0)

    print("Elasticity set to %e" % config.elasticity)
    print("Damping parameters: ",  damping)
    print("Frequency set to %e" % frequency)

    L = config.right - 2*config.extra
    tMax = L*10
    nt = 1
    dt = tMax / nt

    # element size
    maxFreq = 1000
    elementsPerWaveLength = 20
    c = np.sqrt(config.elasticity / config.density)
    waveLength = c / maxFreq
    print("Wavelength / wavespeed: %e / %e" % (waveLength, c))
    if waveLength < elementsPerWaveLength * L / config.n:
        config.n = int(elementsPerWaveLength * L / waveLength + 0.5)
    print("Number of elements (per wavelength): %d (%d)" % (config.n, waveLength / L * config.n))

    # create study
    study = fem1d.EigenvalueStudy(config)

    outputDir = "shaker_elas%e_damp" % config.elasticity
    for d in damping:
        outputDir += "_%e" % d

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    amplitude = 25e-6
    preDisp = -0.6e-3

    # compute critical time step size
    w = study.computeLargestEigenvalueSparse()
    critDeltaT = 2 / abs(w)
    print("Critical time step size is %e" % critDeltaT)
    print("Chosen time step size is %e" % dt)
    dt, nt = fem1d.correctTimeStepSize(dt, tMax, critDeltaT, safety=0.5)
    print("Corrected time step size is %e" % dt)

    # solve sparse
    u0 = np.zeros(study.ansatz.nDof())
    u1 = np.zeros(study.ansatz.nDof())
    evalNodes = np.linspace(study.grid.left + config.extra, study.grid.right - config.extra, study.ansatz.nDof())
    u, fullU, evalU, iMat, times, reactionLeft, reactionRight = fem1d.runCentralDifferenceMethodWithDamping(study, dt, nt, u0, u1, evalNodes, damping, frequency, amplitude, preDisp)

    def shift(vector):
        return vector - np.mean(vector)

    def normalize(vector):
        return vector / max(abs(vector))

    def computeStorageAndLoss(dispPeriod, forcePeriod):
        radius = 0.015
        area = np.pi * radius**2

        maxStress = max(forcePeriod)
        maxStressIndex = np.argmax(forcePeriod)

        maxStrain = max(dispPeriod) / L
        maxStrainIndex = np.argmax(dispPeriod)

        timeShift = dt*(maxStressIndex - maxStrainIndex)

        delta = timeShift * (2*np.pi*frequency) + extraDelta

        while delta > 2*np.pi:
            delta -= 2*np.pi

        while delta < 0:
            delta += 2*np.pi

        deltaStorage = delta

        while deltaStorage > 0.5*np.pi:
            deltaStorage -= np.pi

        while deltaStorage < -0.5*np.pi:
            deltaStorage += np.pi

        deltaLoss = deltaStorage
        if deltaLoss < 0:
            deltaLoss = -deltaLoss

        storage = maxStress / maxStrain * np.cos(deltaStorage)
        loss = maxStress / maxStrain * np.sin(deltaLoss)

        return storage, loss, delta, deltaStorage, deltaLoss

    def evaluate():
        start = int(nt - 1.0 / frequency / dt) - 1
        end = -1
        print("Start: %d" % start)
        disp = shift(u[start:end, -1])
        reacLeft = shift(-reactionLeft[start:end])
        reacRight = shift(reactionRight[start:end])

        if not disablePlots:
            plt.figure()
            plt.plot(disp, reacLeft, label='Kraft links: $F(0,t)$ in $N$')
            plt.plot(disp, reacRight, label='Kraft rechts: $F(L,t)$ in $N$')
            plt.plot(disp, config.elasticity * disp / L, label="statische Kraft in $N$")
            plt.xlabel("Verschiebung in $m$")
            plt.ylabel("Kraft in $N$")
            plt.legend()
            plt.savefig("plot_disp_force_static.pdf") 
            plt.close()  

        storageLeft, lossLeft, deltaLeft, deltaStorageLeft, deltaLossLeft = computeStorageAndLoss(disp, reacLeft)
        storageRight, lossRight, deltaRight, deltaStorageRight, deltaLossRight = computeStorageAndLoss(disp, reacRight)

        with open(outputDir + "/shaker_freq_storage_loss_delta_left.dat", "a") as file:
            file.write("%e %e %e %e %e %e\n" % (frequency, storageLeft, lossLeft, deltaLeft, deltaStorageLeft, deltaLossLeft))

        with open(outputDir + "/shaker_freq_storage_loss_delta_right.dat", "a") as file:
            file.write("%e %e %e %e %e %e\n" % (frequency,  storageRight, lossRight, deltaRight, deltaStorageRight, deltaLossRight))

        data = np.ndarray((nt+1, 4))
        data[:, 0] = times
        data[:, 1] = u[:, -1]
        data[:, 2] = reactionLeft
        data[:, 3] = reactionRight
        np.savetxt(outputDir + "/shaker_time_dispRight_forceLeft_forceRight_freq%d.dat" % int(frequency), data)

        print("storage loss delta left: %e, %e, %e" % (storageLeft, lossLeft, deltaLeft))
        print("storage loss delta right: %e, %e, %e" % (storageRight, lossRight, deltaRight))

        if not disablePlots:
            plt.figure()
            plt.plot(times[start:end], normalize(disp), label='Verschiebung rechts: $y(L,t)$ in $\\mu$m')
            plt.plot(times[start:end], normalize(reacLeft), label='Kraft links: $F(0,t)$ in $N$')
            plt.plot(times[start:end], normalize(reacRight), label='Kraft rechts: $F(L,t)$ in $N$')
            plt.xlabel("Zeit in $s$")
            plt.ylabel("Kraft in $N$")
            plt.legend()
            plt.savefig("plot_normalized_disp_force.pdf") 
            plt.close()  

        return storageLeft, lossLeft, deltaLeft, deltaStorageLeft, deltaLossLeft

    def postProcess(animationSpeed=4, factor=1):
        fem1d.postProcessTimeDomainSolution(study, evalNodes, evalU*factor, tMax, nt, animationSpeed)

    def getResults():
        error = np.linalg.norm(evalU[1] - evalU[-1])
        return w, error, tMax, dt, nt

    if not disablePlots:
        plt.figure()
        plt.plot(times, 1e6 * u[:, -1], label='Verschiebung rechts: $y(L,t)$ in $\\mu$m')
        plt.plot(times, reactionLeft, label='Kraft links: $F(0,t)$ in $N$')
        plt.plot(times, reactionRight, label='Kraft rechts: $F(L,t)$ in $N$')
        plt.xlabel("Zeit in $s$")
        plt.ylabel("Kraft in $N$ und Verschiebung in $\\mu$m")
        plt.legend()
        plt.savefig("plot_disp_and_force_over_time.pdf")  
        plt.close()  

    storageLeft, lossLeft, deltaLeft, deltaStorageLeft, deltaLossLeft = evaluate()

    # if not disablePlots:
    #     postProcess(0.02, 1000)
