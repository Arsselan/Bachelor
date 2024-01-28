import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import fem1d


# 1
def runCentralDifferenceMethod(study, dt, nt, u0, u1, evalPos):
    M = study.getMassMatrix()

    # prepare result arrays
    u = np.zeros((nt + 1, M.shape[0]))
    fullU = np.zeros((nt + 1, study.ansatz.nDof()))
    evalU = np.zeros((nt + 1, len(evalPos)))

    times = np.zeros(nt + 1)

    iMat = study.ansatz.interpolationMatrix(evalPos)

    # set initial conditions
    times[0] = -dt
    times[1] = 0.0
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    for i in range(2):
        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    print("Factorization ... ", flush=True)
    factorized = scipy.sparse.linalg.splu(M)

    print("Time integration ... ", flush=True)
    for i in range(2, nt + 1):
        times[i] = i * dt
        u[i] = factorized.solve(
            M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (
                        study.F * study.config.source.ft((i - 1) * dt) - study.K * u[i - 1]))

        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    return u, fullU, evalU, iMat


# 1
def runCentralDifferenceMethodLowMemory(study, dt, nt, u0, u1, evalPos, evalTimes):
    M = study.getMassMatrix()

    nSavedTimeSteps = len(evalTimes)
    print("Saving times ", evalTimes)
    savedTimeStepIndex = 0

    # prepare result arrays
    u = np.zeros((3, M.shape[0]))
    fullU = np.zeros((3, study.ansatz.nDof()))
    evalU = np.zeros((nSavedTimeSteps, len(evalPos)))
    times = np.zeros(nSavedTimeSteps)

    iMat = study.ansatz.interpolationMatrix(evalPos)

    # set initial conditions
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    fullU[0] = study.system.getFullVector(u[0])
    fullU[1] = study.system.getFullVector(u[1])

    time = -dt
    if time >= evalTimes[savedTimeStepIndex] - 0.5 * dt:
        evalU[savedTimeStepIndex] = iMat * fullU[0]
        times[savedTimeStepIndex] = time
        savedTimeStepIndex += 1
        print("Stored time step %d, time %e" % (0, time))
    time = 0.0
    if time >= evalTimes[savedTimeStepIndex] - 0.5 * dt:
        evalU[savedTimeStepIndex] = iMat * fullU[1]
        times[savedTimeStepIndex] = time
        savedTimeStepIndex += 1
        print("Stored time step %d, time %e" % (1, time))

    print("Factorization ... ", flush=True)
    factorized = scipy.sparse.linalg.splu(M)

    print("Time integration ... ", flush=True)
    study.F = study.F * 0
    for i in range(2, nt + 1):
        time = i * dt
        u[2] = factorized.solve(
            M * (2 * u[1] - u[0]) + dt ** 2 * (study.F * study.config.source.ft((i - 1) * dt) - study.K * u[1]))
        if savedTimeStepIndex < nSavedTimeSteps and time >= evalTimes[savedTimeStepIndex] - 0.5 * dt:
            fullU[2] = study.system.getFullVector(u[2])
            evalU[savedTimeStepIndex] = iMat * fullU[2]
            times[savedTimeStepIndex] = time
            savedTimeStepIndex += 1
            print("Stored time step %d, time %e" % (i, time))

        u[0] = u[1]
        u[1] = u[2]

    return times, fullU, evalU, iMat


# 2
def runCentralDifferenceMethodStrongContactBoundaryFitted(study, dt, nt, u0, u1, evalPos):
    if study.config.extra != 0.0:
        print("Error! This function is designed for boundary fitted cases only.")
        return

    # get mass matrix
    M = study.getMassMatrix()

    # prepare result arrays
    u = np.zeros((nt + 2, M.shape[0]))
    fullU = np.zeros((nt + 2, study.ansatz.nDof()))
    evalU = np.zeros((nt + 2, len(evalPos)))
    times = np.zeros(nt + 2)

    # get interpolation matrix
    iMat = study.ansatz.interpolationMatrix(evalPos)

    # set initial conditions
    times[0] = -dt
    times[1] = 0.0
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    for i in range(2):
        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    print("Factorization ... ", flush=True)
    factorized = scipy.sparse.linalg.splu(M)

    print("Time integration ... ", flush=True)
    for i in range(2, nt + 2):
        times[i] = (i-1) * dt
        u[i] = factorized.solve(
            M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (
                        study.F * study.config.source.ft((i - 1) * dt) - study.K * u[i - 1]))

        # check penetration
        if u[i][-1] > 0.1:
            u[i][-1] = 0.1
            u[i - 1][-1] = 0.1
            u[i - 2][-1] = 0.1

        if u[i][0] < -0.1:
            u[i][0] = -0.1
            u[i - 1][0] = -0.1
            u[i - 2][0] = -0.1

        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

        # double check penetration
        currentPos = evalPos + evalU[i]
        if (currentPos > study.grid.right + 0.1).any():
            print("Error! Right end penetrates boundary.")
        if (currentPos < study.grid.left - 0.1).any():
            print("Error! Left end penetrates boundary.")

    return times, u, fullU, evalU, iMat


# 2
def runCentralDifferenceMethodStrongContactBoundaryFittedLowMemory(study, dt, nt, u0, u1, evalPos):
    if study.config.extra != 0.0:
        print("Error! This function is designed for boundary fitted cases only.")
        return

    # get mass matrix
    M = study.getMassMatrix()

    # prepare result arrays
    u = np.zeros((3, M.shape[0]))
    fullU = np.zeros((3, study.ansatz.nDof()))
    evalU = np.zeros((nt + 2, len(evalPos)))

    times = np.zeros(nt + 2)

    iMat = study.ansatz.interpolationMatrix(evalPos)

    # set initial conditions
    times[0] = -dt
    times[1] = 0.0
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    for i in range(2):
        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    print("Factorization ... ", flush=True)
    factorized = scipy.sparse.linalg.splu(M)

    print("Time integration ... ", flush=True)
    study.F = study.F * 0
    onePercent = int(nt / 100)
    for i in range(2, nt + 2):
        if i % onePercent == 0:
            print(".", end="", flush=True)
            if i % (onePercent * 10) == 0:
                print("%d%%" % (i / onePercent))

        # solve
        times[i] = (i - 1) * dt
        u[2] = factorized.solve(M * (2 * u[1] - u[0]) + dt ** 2 * (study.F - study.K * u[1]))

        # check penetration
        if u[2][-1] > 0.1:
            u[2][-1] = 0.1
            u[2 - 1][-1] = 0.1
            u[2 - 2][-1] = 0.1

        if u[2][0] < -0.1:
            u[2][0] = -0.1
            u[2 - 1][0] = -0.1
            u[2 - 2][0] = -0.1

        # double check penetration
        currentPos = evalPos + evalU[i]
        if (currentPos > study.grid.right + 0.1).any():
            print("Error! Right end penetrates boundary.")
        if (currentPos < study.grid.left - 0.1).any():
            print("Error! Left end penetrates boundary.")

        fullU[2] = study.system.getFullVector(u[2])
        evalU[i] = iMat * fullU[2]

        u[0] = u[1]
        u[1] = u[2]

    return times, u, fullU, evalU, iMat


# 3
def runCentralDifferenceMethodWeakContactBoundaryFitted(study, dt, nt, u0, u1, evalPos, penaltyFactor):
    if study.config.extra != 0.0:
        print("Error! This function is designed for boundary fitted cases only.")
        return

    # get mass matrix
    M = study.getMassMatrix()

    # prepare result arrays
    u = np.zeros((nt + 2, M.shape[0]))
    fullU = np.zeros((nt + 2, study.ansatz.nDof()))
    evalU = np.zeros((nt + 2, len(evalPos)))
    times = np.zeros(nt + 2)

    # get interpolation matrix
    iMat = study.ansatz.interpolationMatrix(evalPos)

    # set initial conditions
    times[0] = -dt
    times[1] = 0.0
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    for i in range(2):
        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    print("Factorization ... ", flush=True)
    factorized = scipy.sparse.linalg.splu(M)

    print("Time integration ... ", flush=True)
    study.F = study.F * 0
    onePercent = int(nt / 100)
    for i in range(2, nt + 2):
        if i % onePercent == 0:
            print("%d / %d" % (i, nt))

        times[i] = (i-1) * dt
        u[i] = factorized.solve(M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (study.F - study.K * u[i - 1]))

        if u[i][-1] > 0.1:  # and u[i][-1] - u[i-1][-1] > 0:
            study.F[-1] = penaltyFactor * (0.1 - u[i][-1])
        else:
            study.F[-1] = 0

        if u[i][0] < -0.1:  # and u[i][0] - u[i-1][0] < 0:
            study.F[0] = penaltyFactor * (-0.1 - u[i][0])
        else:
            study.F[0] = 0

        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    return times, u, fullU, evalU, iMat


# 4
def runCentralDifferenceMethodWeakContactBoundaryFittedLowMemory(study, dt, nt, u0, u1, evalPos, penaltyFactor):
    M = study.getMassMatrix()

    # prepare result arrays
    u = np.zeros((3, M.shape[0]))
    fullU = np.zeros((3, study.ansatz.nDof()))
    evalU = np.zeros((nt + 2, len(evalPos)))

    times = np.zeros(nt + 2)

    iMat = study.ansatz.interpolationMatrix(evalPos)

    # set initial conditions
    times[0] = -dt
    times[1] = 0.0
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    for i in range(2):
        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    print("Factorization ... ", flush=True)
    factorized = scipy.sparse.linalg.splu(M)

    print("Time integration ... ", flush=True)
    study.F = study.F * 0
    onePercent = int(nt / 100)
    for i in range(2, nt + 2):
        if i % onePercent == 0:
            print(".", end="", flush=True)
            if i % (onePercent*10) == 0:
                print("%d%%" % (i / onePercent))

        # solve
        times[i] = (i-1) * dt
        u[2] = factorized.solve(M * (2 * u[1] - u[0]) + dt ** 2 * (study.F - study.K * u[1]))

        # check penetration
        if u[2][-1] > 0.1:  # and u[2][-1] - u[1][-1] > 0:
            study.F[-1] = penaltyFactor * (0.1 - u[2][-1])
        else:
            study.F[-1] = 0

        if u[2][0] < -0.1:  # and u[2][0] - u[1][0] < 0:
            study.F[0] = penaltyFactor * (-0.1 - u[2][0])
        else:
            study.F[0] = 0

        fullU[2] = study.system.getFullVector(u[2])
        evalU[i] = iMat * fullU[2]

        u[0] = u[1]
        u[1] = u[2]

    return times, u, fullU, evalU, iMat


# 5
def runCentralDifferenceMethodWeakContactImmersedLowMemory(study, dt, nt, u0, u1, evalPos, penaltyFactor):
    M = study.getMassMatrix()

    # prepare result arrays
    u = np.zeros((3, M.shape[0]))
    fullU = np.zeros((3, study.ansatz.nDof()))
    evalU = np.zeros((nt + 2, len(evalPos)))

    times = np.zeros(nt + 2)

    # compute interpolation matrix
    iMat = study.ansatz.interpolationMatrix(evalPos)

    # compute Neumann vectors
    leftF = study.system.getReducedVector(fem1d.createNeumannVector(study.system, [evalPos[0]], [1], [1]))
    rightF = study.system.getReducedVector(fem1d.createNeumannVector(study.system, [evalPos[-1]], [1], [1]))
    leftFactor = 0
    rightFactor = 0

    # set initial conditions
    times[0] = -dt
    times[1] = 0.0
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    for i in range(2):
        fullU[i] = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU[i]

    print("Factorization ... ", flush=True)
    factorized = scipy.sparse.linalg.splu(M)

    print("Time integration ... ", flush=True)
    onePercent = int(nt / 100)
    for i in range(2, nt + 2):
        if i % onePercent == 0:
            print(".", end="", flush=True)
            if i % (onePercent*10) == 0:
                print("%d%%" % (i / onePercent))

        # solve
        times[i] = i * dt
        u[2] = factorized.solve(
            M * (2 * u[1] - u[0]) + dt ** 2 * (leftFactor * leftF + rightFactor * rightF - study.K * u[1]))

        # evaluate solution
        fullU[2] = study.system.getFullVector(u[2])
        evalU[i] = iMat * fullU[2]

        # check penetration
        if evalU[i][-1] > 0.1+study.config.extra:  # and evalU[i][-1] - evalU[i-1][-1] > 0:
            rightFactor = penaltyFactor * (0.1+study.config.extra - evalU[i][-1])
        else:
            rightFactor = 0

        if evalU[i][0] < -0.1-study.config.extra:  # and evalU[i][0] - evalU[i-1][0] < 0:
            leftFactor = penaltyFactor * (-0.1-study.config.extra - evalU[i][0])
        else:
            leftFactor = 0

        # update solution vectors
        u[0] = u[1]
        u[1] = u[2]

    return times, u, fullU, evalU, iMat


# 5
def runCentralDifferenceMethodWeakContactImmersedLowMemoryPlastic(study, c, epsYield, hardening, dt, nt, u0, u1, evalPos,
                                                                  computeExternalLoad,
                                                                  earliestLumping=0, dampingM=0, dampingK=0):
    M = study.getMassMatrix()

    if earliestLumping > 0:
        M = study.M

    # prepare result arrays
    u = np.zeros((3, M.shape[0]))
    evalU = np.zeros((nt + 2, len(evalPos)))

    times = np.zeros(nt + 2)

    # compute interpolation matrix
    iMat = study.ansatz.interpolationMatrix(evalPos)

    # set initial conditions
    times[0] = -dt
    times[1] = 0.0
    u[0] = study.system.getReducedVector(u0)
    u[1] = study.system.getReducedVector(u1)
    for i in range(2):
        fullU = study.system.getFullVector(u[i])
        evalU[i] = iMat * fullU

    # compute initial external load
    externalLoad = computeExternalLoad(-dt, u[1], u[0])

    # internal variables
    epsPla = fem1d.createQuadraturePointData(study.quadratureK)
    alphaPla = fem1d.createQuadraturePointData(study.quadratureK)

    print("Factorization ... ", flush=True)
    C = 0.5 * dt * (dampingM * M + dampingK*study.K)
    factorized = scipy.sparse.linalg.splu(M + C)

    print("Time integration ... ", flush=True)
    onePercent = int(nt / 100)
    for i in range(2, nt + 2):
        if i == int(onePercent * earliestLumping):
            M = study.getMassMatrix()
            C = 0.5 * dt * (dampingM * M + dampingK * study.K)
            factorized = scipy.sparse.linalg.splu(M + C)

        if i % onePercent == 0:
            print(".", end="", flush=True)
            if i % (onePercent*10) == 0:
                minEpsPla = fem1d.minListElement(epsPla)
                maxEpsPla = fem1d.maxListElement(epsPla)
                maxDisp = np.max(np.abs(evalU))
                print("%d%%, minEpsPla: %e,  maxEpsPla: %e, maxDisp: %e" % (i / onePercent, minEpsPla, maxEpsPla, maxDisp))

        times[i] = (i - 1) * dt

        # solve
        if 0:
            internalLoad = study.K * u[1]
        else:
            #internalLoad = study.system.getReducedVector(fem1d.computePlasticInnerLoadVector(c, epsYield, hardening, study.ansatz, study.quadratureK, fullU, epsPla))
            internalLoad = study.system.getReducedVector(fem1d.computePlasticInnerLoadVectorIsotropic(c, epsYield, hardening, study.ansatz, study.quadratureK, fullU, epsPla, alphaPla))

        u[2] = factorized.solve(M * (2 * u[1] - u[0]) + C * u[0] + dt ** 2 * (externalLoad - internalLoad))

        if 0:
            velocity = 1
            if times[i] < 0.025:
                velocity *= (times[i]/0.025)
            u[2][-1] = u[1][-1] + dt*velocity

        # evaluate solution
        fullU = study.system.getFullVector(u[2])
        evalU[i] = iMat * fullU

        # compute external load
        externalLoad = computeExternalLoad(times[i], evalU[i], evalU[i-1])

        # update solution vectors
        u[0] = u[1]
        u[1] = u[2]

    return times, u, fullU, evalU, iMat, epsPla
