import numpy as np
from context import fem1d

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
        n=80,
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
config.elasticity = 4e4#1.5e5
damping = 2000#2e5#100


L = config.right - 2*config.extra
tMax = L*10
nt = 1
dt = tMax / nt

# create study
study = fem1d.EigenvalueStudy(config)

disablePlots = True
if 'frequency' not in locals():
    frequency = 20
    disablePlots = False

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

    delta = timeShift * (2*np.pi*frequency)

    storage = maxStress / maxStrain * np.cos(delta)
    loss = maxStress / maxStrain * np.sin(delta)

    #fem1d.plot(times, [dispPeriod, forcePeriod], ["disp", "force"])
    #print("Max disp at: %d, %e" % (maxStrainIndex, times[maxStrainIndex]))
    #print("Max force at: %d, %e" % (maxStressIndex, times[maxStressIndex]))
    #print("Time shift: %e" % timeShift)

    while delta > np.pi:
        delta -= 2*np.pi

    while delta < -np.pi:
        delta += 2*np.pi

    return storage, loss, delta


def evaluate():
    start = int(nt - 1.0 / frequency / dt) - 1
    end = -1
    print("Start: %d" % start)
    disp = shift(u[start:end, -1])
    reacLeft = shift(-reactionLeft[start:end])
    reacRight = shift(reactionRight[start:end])

    storageLeft, lossLeft, deltaLeft = computeStorageAndLoss(disp, reacLeft)
    storageRight, lossRight, deltaRight = computeStorageAndLoss(disp, reacRight)

    with open("shaker_damp_freq_storage_loss_delta_left.dat", "a") as file:
        file.write("%e %e %e %e %e\n" % (damping, frequency, storageLeft, lossLeft, deltaLeft))

    with open("shaker_damp_freq_storage_loss_delta_right.dat", "a") as file:
        file.write("%e %e %e %e %e\n" % (damping, frequency,  storageRight, lossRight, deltaRight))

    data = np.ndarray((nt+1, 4))
    data[:, 0] = times
    data[:, 1] = u[:, -1]
    data[:, 2] = reactionLeft
    data[:, 3] = reactionRight
    np.savetxt("shaker_time_dispRight_forceLeft_forceRight_freq%d_damp_%e.dat" % (int(frequency), damping), data)

    #print("Max disp: %e, max reaction: %e" % (maxDisp, maxReac))
    #print("Max disp: %d, max reaction: %d" % (maxDispIndex, maxReacIndex))
    #print("Time shift: %e, delta: %e" % (timeShift, delta))
    print("storage loss delta left: %e, %e, %e" % (storageLeft, lossLeft, deltaLeft))
    print("storage loss delta right: %e, %e, %e" % (storageRight, lossRight, deltaRight))

    if not disablePlots:
        fem1d.plot(times[start:end], [normalize(disp), normalize(reacLeft), normalize(reacRight)], ["disp", "left", "right"])


def postProcess(animationSpeed=4, factor=1):
    fem1d.postProcessTimeDomainSolution(study, evalNodes, evalU*factor, tMax, nt, animationSpeed)


def getResults():
    error = np.linalg.norm(evalU[1] - evalU[-1])
    return w, error, tMax, dt, nt


if not disablePlots:
    fem1d.plot(1e6 * u[-100000:, -1], [reactionRight[-100000:]], ["force right"])
    fem1d.plot(1e6 * u[-100000:, -1], [reactionLeft[-100000:]], ["force left"])
    fem1d.plot(times, [1e6 * u[:, -1], reactionLeft, reactionRight], ["disp. right", "force left", "force right"])

evaluate()
#postProcess(0.1)

