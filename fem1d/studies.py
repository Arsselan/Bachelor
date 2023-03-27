import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

import fem1d


class StudyConfig:

    def __init__(self, left, right, extra, n, p, ansatzType, continuity, mass, depth, stabilize, spectral, dual, smartQuadrature, source):
        self.left = left
        self.right = right
        self.extra = extra

        self.n = n
        self.p = p
        self.ansatzType = ansatzType
        self.continuity = continuity
        self.mass = mass

        self.depth = depth
        self.stabilize = stabilize
        self.spectral = spectral
        self.dual = dual
        self.smartQuadrature = smartQuadrature
        self.source = source

        if ansatzType == 'Lagrange':
            self.continuity = '0'

        if ansatzType == 'Spline':
            self.spectral = False


class EigenvalueStudy:

    def __init__(self, config):

        self.config = config

        # create grid and domain
        grid = fem1d.UniformGrid(config.left, config.right, config.n)

        left = config.left
        right = config.right
        extra = config.extra
        stabilize = config.stabilize

        def alpha(x):
            if left + extra <= x <= right - extra:
                return 1
            return stabilize

        domain = fem1d.Domain(alpha)

        # create ansatz and quadrature points
        ansatz = fem1d.createAnsatz(config.ansatzType, config.continuity, config.p, grid)
        gaussPointsM = fem1d.gll.computeGllPoints(config.p + 1)
        if config.smartQuadrature is False:
            quadratureM = fem1d.SpaceTreeQuadrature(grid, gaussPointsM, domain, config.depth)
        else:
            quadratureM = fem1d.SmartQuadrature(grid, gaussPointsM, domain, [extra, right-extra])

        gaussPointsK = np.polynomial.legendre.leggauss(config.p + 1)
        if config.smartQuadrature is False:
            quadratureK = fem1d.SpaceTreeQuadrature(grid, gaussPointsK, domain, config.depth)
        else:
            quadratureK = fem1d.SmartQuadrature(grid, gaussPointsK, domain, [extra, right-extra])

        # create system
        if config.spectral:
            system = fem1d.TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK, config.source.fx)
        else:
            if config.dual:
                print("DUAL! %e" % extra)
                system = fem1d.TripletSystem.fromOneQuadratureWithDualBasis(ansatz, quadratureK, config.source.fx)
            else:
                system = fem1d.TripletSystem.fromOneQuadrature(ansatz, quadratureK, config.source.fx)

        #    system.findZeroDof(-1e60, [0, 1, system.nDof()-2, system.nDof()-1])
        system.findZeroDof(0)
        if len(system.zeroDof) > 0:
            print("Warning! There were %d zero dof found." % len(system.zeroDof))
            #print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

        # get matrices
        self.M, self.K, self.MHRZ, self.MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)
        self.F = system.getReducedVector(system.F)

        self.grid = grid
        self.domain = domain
        self.ansatz = ansatz
        self.quadratureM = quadratureM
        self.quadratureK = quadratureK
        self.system = system

        self.w = 0
        self.v = 0
        self.nNegative = 0
        self.nComplex = 0

    def getMassMatrix(self):
        if self.config.mass == 'CON':
            return self.M
        elif self.config.mass == 'HRZ':
            return self.MHRZ
        elif self.config.mass == 'RS':
            return self.MRS
        else:
            print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

    def runDense(self, computeEigenvectors=False, sort=False):
        M = self.getMassMatrix()
        if computeEigenvectors:
            self.w, self.v = scipy.linalg.eig(self.K.toarray(), M.toarray(), right=True)
            # self.w, self.v = scipy.linalg.eigh(self.K.toarray(), M.toarray())
            # self.w, self.v = scipy.linalg.eig(np.float32(self.K.toarray()), np.float32(M.toarray()), right=True)
            # self.w, self.v = scipy.linalg.eig(np.float32(self.K.toarray()), np.float32(M.toarray()))
        else:
            self.w = scipy.linalg.eigvals(self.K.toarray(), self.M.toarray())

        wNegative = self.w < 0
        if wNegative.any():
            print("Warning! Found negative eigenvalue.")
        self.nNegative = wNegative.sum()

        wComplex = np.abs(self.w.imag) > 0
        if wComplex.any():
            print("Warning! Found complex eigenvalue.")
        self.nComplex = wComplex.sum()

        self.w = np.sqrt(np.abs(self.w))

        if sort:
            idx = self.w.argsort()[::1]
            self.w = self.w[idx]
            if computeEigenvectors:
                self.v = self.v[:, idx]

        return max(self.w)

    def runSparse(self, computeEigenvectors=False, sort=False):
        nEigen = self.K.shape[0] - 2
        M = self.getMassMatrix()
        if computeEigenvectors:
            # self.w, self.v = scipy.sparse.linalg.eigs(self.K, nEigen, M, which='SM', return_eigenvectors=True)
            self.w, self.v = scipy.sparse.linalg.eigsh(self.K, nEigen, M, which='SM', return_eigenvectors=True)
            # self.w, self.v = scipy.sparse.linalg.eigsh(self.K, nEigen, M, sigma=0, ncv=self.K.shape[0], maxiter=5000, which='LM', return_eigenvectors=True)
            # self.w, self.v = scipy.sparse.linalg.eigsh(self.K.astype("float32"), nEigen, M.astype("float32"), which='SM', return_eigenvectors=True)
        else:
            self.w = scipy.sparse.linalg.eigs(self.K, nEigen, M, which='SM', return_eigenvectors=False)

        self.w = np.sqrt(np.abs(self.w))

        if sort:
            idx = self.w.argsort()[::1]
            self.w = self.w[idx]
            if computeEigenvectors:
                self.v = self.v[:, idx]

        return max(self.w)

    def computeLargestEigenvalueSparse(self):
        M = self.getMassMatrix()
        self.w = scipy.sparse.linalg.eigs(self.K, 1, M, which='LM', return_eigenvectors=False)
        self.w = np.sqrt(np.abs(self.w))
        return max(self.w)

    def computeLargestEigenvalueDense(self):
        M = self.getMassMatrix()
        self.w = scipy.linalg.eigvals(self.K.toarray(), M.toarray())
        self.w = np.sqrt(np.abs(self.w))
        return max(self.w)

    def runCentralDifferenceMethod(self, dt, nt, u0, u1, evalPos):
        M = self.getMassMatrix()

        # prepare result arrays
        u = np.zeros((nt + 1, M.shape[0]))
        fullU = np.zeros((nt + 1, self.ansatz.nDof()))
        evalU = np.zeros((nt + 1, len(evalPos)))

        times = np.zeros(nt + 1)

        iMat = self.ansatz.interpolationMatrix(evalPos)

        # set initial conditions
        times[0] = -dt
        times[1] = 0.0
        u[0] = self.system.getReducedVector(u0)
        u[1] = self.system.getReducedVector(u1)
        for i in range(2):
            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

        #self.F = self.system.getReducedVector(self.system.F + createNeumannVector(self.system, [self.config.extra, self.config.right-self.config.extra], [-1, 1], [1, 1]))
        #print(self.F)

        nodes = self.grid.getNodes()
        nNodes = len(nodes)

        print("Factorization ... ", flush=True)
        factorized = scipy.sparse.linalg.splu(M)

        print("Time integration ... ", flush=True)
        for i in range(2, nt + 1):
            times[i] = i * dt
            u[i] = factorized.solve(
                M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (self.F * self.config.source.ft((i - 1) * dt) - self.K * u[i - 1]))
                #M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (self.F - self.K * u[i - 1]))

            #if u[i][-1] > 0.1:
            #    print("CONTACT!")
            #    u[i][-1] = 0.1
            #    u[i-1][-1] = 0.1

            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

        return u, fullU, evalU, iMat

    def runCentralDifferenceMethod2(self, dt, nt, u0, u1, evalPos):
        if self.config.mass == 'CON':
            M = self.M
        elif self.config.mass == 'HRZ':
            M = self.MHRZ
        elif self.config.mass == 'RS':
            M = self.MRS
        else:
            print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

        # prepare result arrays
        u = np.zeros((nt + 1, M.shape[0]))
        fullU = np.zeros((nt + 1, self.ansatz.nDof()))
        evalU = np.zeros((nt + 1, len(evalPos)))

        times = np.zeros(nt + 1)

        iMat = self.ansatz.interpolationMatrix(evalPos)

        # set initial conditions
        times[0] = -dt
        times[1] = 0.0
        u[0] = self.system.getReducedVector(u0)
        u[1] = self.system.getReducedVector(u1)
        for i in range(2):
            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

        nodes = self.grid.getNodes()
        nNodes = len(nodes)

        print("Factorization ... ", flush=True)
        factorized = scipy.sparse.linalg.splu(M)

        print("Time integration ... ", flush=True)
        for i in range(2, nt + 1):
            times[i] = i * dt
            u[i] = factorized.solve(
                M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (self.F * self.config.source.ft((i - 1) * dt) - self.K * u[i - 1]))
                #M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (self.F - self.K * u[i - 1]))

            if u[i][-1] > 0.1:
                u[i][-1] = 0.1
                u[i-1][-1] = 0.1

            if u[i][0] < -0.1:
                u[i][0] = -0.1
                u[i-1][0] = -0.1

            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

            currentPos = evalPos + evalU[i]
            if (currentPos > self.grid.right + 0.1).any():
                print("Error! Right end penetrates boundary.")
            if (currentPos < self.grid.left - 0.1).any():
                print("Error! Left end penetrates boundary.")

        return times, u, fullU, evalU, iMat

    def runCentralDifferenceMethod3(self, dt, nt, u0, u1, evalPos):
        M = self.getMassMatrix()

        # prepare result arrays
        u = np.zeros((nt + 1, M.shape[0]))
        fullU = np.zeros((nt + 1, self.ansatz.nDof()))
        evalU = np.zeros((nt + 1, len(evalPos)))

        times = np.zeros(nt + 1)

        iMat = self.ansatz.interpolationMatrix(evalPos)

        # set initial conditions
        times[0] = -dt
        times[1] = 0.0
        u[0] = self.system.getReducedVector(u0)
        u[1] = self.system.getReducedVector(u1)
        for i in range(2):
            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

        nodes = self.grid.getNodes()
        nNodes = len(nodes)

        print("Factorization ... ", flush=True)
        factorized = scipy.sparse.linalg.splu(M)

        print("Time integration ... ", flush=True)
        self.F = self.F * 0
        onepercent = int(nt / 100)
        for i in range(2, nt + 1):
            if i % onepercent == 0:
                print("%d / %d" % (i, nt))
            times[i] = i * dt
            u[i] = factorized.solve(M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (self.F - self.K * u[i - 1]))

            penalty = 1e3
            if u[i][-1] > 0.1 and u[i][-1] - u[i-1][-1] > 0:
                self.F[-1] = penalty * (0.1 - u[i][-1])
            else:
                self.F[-1] = 0

            if u[i][0] < -0.1 and u[i][0] - u[i-1][0] < 0:
                self.F[0] = penalty * (-0.1 - u[i][0])
            else:
                self.F[0] = 0

            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

        return times, u, fullU, evalU, iMat

    def runCentralDifferenceMethod4(self, dt, nt, u0, u1, evalPos):
        M = self.getMassMatrix()

        # prepare result arrays
        u = np.zeros((3, M.shape[0]))
        fullU = np.zeros((3, self.ansatz.nDof()))
        evalU = np.zeros((nt + 1, len(evalPos)))

        times = np.zeros(nt + 1)

        iMat = self.ansatz.interpolationMatrix(evalPos)

        # set initial conditions
        times[0] = -dt
        times[1] = 0.0
        u[0] = self.system.getReducedVector(u0)
        u[1] = self.system.getReducedVector(u1)
        for i in range(2):
            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

        nodes = self.grid.getNodes()
        nNodes = len(nodes)

        print("Factorization ... ", flush=True)
        factorized = scipy.sparse.linalg.splu(M)

        print("Time integration ... ", flush=True)
        self.F = self.F * 0
        onepercent = int(nt / 100)
        for i in range(2, nt + 1):
            if i % onepercent == 0:
                print("%d / %d" % (i, nt))
            times[i] = i * dt
            u[2] = factorized.solve(M * (2 * u[1] - u[0]) + dt ** 2 * (self.F - self.K * u[1]))

            penalty = 1e3
            if u[2][-1] > 0.1 and u[2][-1] - u[1][-1] > 0:
                self.F[-1] = penalty * (0.1 - u[2][-1])
            else:
                self.F[-1] = 0

            if u[2][0] < -0.1 and u[2][0] - u[1][0] < 0:
                self.F[0] = penalty * (-0.1 - u[2][0])
            else:
                self.F[0] = 0

            fullU[2] = self.system.getFullVector(u[2])
            evalU[i] = iMat * fullU[2]

            u[0] = u[1]
            u[1] = u[2]

        return times, u, fullU, evalU, iMat

    def runCentralDifferenceMethod5(self, dt, nt, u0, u1, evalPos):
        M = self.getMassMatrix()

        # prepare result arrays
        u = np.zeros((3, M.shape[0]))
        fullU = np.zeros((3, self.ansatz.nDof()))
        evalU = np.zeros((nt + 1, len(evalPos)))

        times = np.zeros(nt + 1)

        # compute interpolation matrix
        iMat = self.ansatz.interpolationMatrix(evalPos)

        # compute Neumann vectors
        leftF = self.system.getReducedVector(fem1d.createNeumannVector(self.system, [evalPos[0]], [1], [1]))
        rightF = self.system.getReducedVector(fem1d.createNeumannVector(self.system, [evalPos[-1]], [1], [1]))
        leftFactor = 0
        rightFactor = 0

        # set initial conditions
        times[0] = -dt
        times[1] = 0.0
        u[0] = self.system.getReducedVector(u0)
        u[1] = self.system.getReducedVector(u1)
        for i in range(2):
            fullU[i] = self.system.getFullVector(u[i])
            evalU[i] = iMat * fullU[i]

        print("Factorization ... ", flush=True)
        factorized = scipy.sparse.linalg.splu(M)

        print("Time integration ... ", flush=True)
        onePercent = int(nt / 100)
        for i in range(2, nt + 1):
            if i % onePercent == 0:
                print("%d / %d" % (i, nt))

            # solve
            times[i] = i * dt
            u[2] = factorized.solve(M * (2 * u[1] - u[0]) + dt ** 2 * (leftFactor * leftF + rightFactor * rightF - self.K * u[1]))

            # evaluate solution
            fullU[2] = self.system.getFullVector(u[2])
            evalU[i] = iMat * fullU[2]

            # check penetration
            penalty = 1e3
            if evalU[i][-1] > 0.1+self.config.extra and evalU[i][-1] - evalU[i-1][-1] > 0:
                rightFactor = penalty * (0.1+self.config.extra - evalU[i][-1])
            else:
                rightFactor = 0

            if evalU[i][0] < -0.1-self.config.extra and evalU[i][0] - evalU[i-1][0] < 0:
                leftFactor = penalty * (-0.1-self.config.extra - evalU[i][0])
            else:
                leftFactor = 0

            # update solution vectors
            u[0] = u[1]
            u[1] = u[2]

        return times, u, fullU, evalU, iMat


def findEigenvalue(w, eigenvalueSearch, eigenvalue, wExact):
    if eigenvalueSearch == 'nearest':
        wNum = fem1d.find_nearest(w, wExact)
        idx = fem1d.find_nearest_index(w, wExact)
        print("w index = %d" % idx)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
        idx = eigenvalue
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    return wNum, idx


def findEigenvector(v, search, index, iMatrix, system, vExact):
    if search == 'nearest':
        minIndex = 0
        minError = 1e10
        nEigen = v.shape[1]
        for idx in range(nEigen):
            # print(idx)
            if np.linalg.norm(np.imag(v[:, idx])) == 0:
                eVector = iMatrix * system.getFullVector(np.real(v[:, idx]))
                eVector = eVector / eVector[0]
                eVector *= np.linalg.norm(vExact) / np.linalg.norm(eVector)
                error = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
                if error < minError:
                    # plot(nodesEval, [vExact, eVector])
                    minError = error
                    minIndex = idx
            else:
                print("Warning! Found complex eigenvector %d." % idx)

        print("v index = %d" % minIndex)

    elif search == 'number':
        minIndex = index
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    vNum = iMatrix * system.getFullVector(np.real(v[:, minIndex]))
    vNum = vNum / vNum[0]
    vNum *= np.linalg.norm(vExact) / np.linalg.norm(vNum)

    return vNum, minIndex


def correctTimeStepSize(dt, tMax, critDeltaT, safety=0.9):
    if dt > critDeltaT * safety:
        dt = critDeltaT * safety
        nt = int(tMax / dt + 0.5)
        dt = tMax / nt
    return dt


# Plot animation
def postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed=4):
    figure, ax = plt.subplots()
    ax.set_xlim(study.grid.left, study.grid.right)
    ax.set_ylim(-2, 2)

    config = study.config
    ax.plot([config.left+config.extra, config.left+config.extra], [-0.1, 0.1], '--', label='left boundary')
    ax.plot([config.right-config.extra, config.right-config.extra], [-0.1, 0.1], '--', label='right boundary')

    ax.plot(evalNodes, evalU[1], '--', label='initial condition')

    line2, = ax.plot(0, 0, label='numerical')
    line2.set_xdata(evalNodes)

    ax.legend()

    plt.rcParams['axes.titleweight'] = 'bold'
    title = 'Solution'
    plt.title(title)
    plt.xlabel('solution')
    plt.ylabel('x')

    def prepareFrame(i):
        step = int(round(i / tMax * nt))
        plt.title(title + " time %3.2e step %d" % (i, step))
        #line2.set_ydata(evalU[step])
        line2.set_xdata(evalNodes + evalU[step])

    frames = np.linspace(0, tMax, round(tMax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000 / 60, repeat=False)
    plt.show()

