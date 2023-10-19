import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

import fem1d


class StudyConfig:

    def __init__(self, left, right, extra,
                 n, p, ansatzType, continuity, mass, depth, stabilize,
                 spectral, dual, smartQuadrature, source, fixedDof=[],
                 eigenvalueStabilizationM=0.0, eigenvalueStabilizationK=0.0):

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
        self.fixedDof = fixedDof
        self.eigenvalueStabilizationM = eigenvalueStabilizationM
        self.eigenvalueStabilizationK = eigenvalueStabilizationK

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
            system = fem1d.TripletSystem(ansatz)

            matrices = fem1d.WaveEquationStiffnessMatrixAndLoadVector(1.0, config.source.fx)
            fem1d.computeSystemMatrices(system, ansatz, quadratureK, matrices)

            matrices = fem1d.WaveEquationMassMatrix(1.0)
            matrices = fem1d.WaveEquationLumpedMatrices(matrices)
            fem1d.computeSystemMatrices(system, ansatz, quadratureM, matrices)
        else:
            system = fem1d.TripletSystem(ansatz)
            matrices = fem1d.WaveEquationStandardMatrices(1.0, 1.0, config.source.fx)
            matrices = fem1d.WaveEquationLumpedMatrices(matrices)

            if config.eigenvalueStabilizationM > 0.0:
                matrices = fem1d.WaveEquationStabilizedMatrices(matrices, config.eigenvalueStabilizationM)
            fem1d.computeSystemMatrices(system, ansatz, quadratureK, matrices)

        print("Matrix values: ", list(system.matrixValues.keys()))

        # disable certain dof
        #    system.findZeroDof(-1e60, [0, 1, system.nDof()-2, system.nDof()-1])
        system.findZeroDof(0, self.config.fixedDof)
        if len(system.zeroDof) > 0:
            print("Warning! There were %d zero dof found." % len(system.zeroDof))
            #print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

        # get matrices
        self.K = system.createSparseMatrix('K')
        if config.mass == 'CON':
            if config.eigenvalueStabilizationM > 0:
                self.M = system.createSparseMatrix('modM')
            else:
                self.M = system.createSparseMatrix('M')
        elif config.mass == 'RS':
            self.M = system.createSparseMatrix('MRS')
        elif config.mass == 'HRZ':
            if config.eigenvalueStabilizationM > 0:
                self.M = system.createSparseMatrix('modMHRZ')
            else:
                self.M = system.createSparseMatrix('MHRZ')
        else:
            print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

        print("Matrices: ", list(system.matrices.keys()))

        self.F = system.getReducedVector(system.vectors['F'])

        # store stuff
        self.grid = grid
        self.domain = domain
        self.ansatz = ansatz
        self.quadratureM = quadratureM
        self.quadratureK = quadratureK
        self.system = system

        # prepare result fields
        self.w = 0
        self.v = 0
        self.nNegative = 0
        self.nComplex = 0

    def getMassMatrix(self):
        return self.M

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
        self.w = scipy.sparse.linalg.eigs(self.K, 1, self.M, which='LM', return_eigenvectors=False)
        self.w = np.sqrt(np.abs(self.w))
        return max(self.w)

    def computeLargestEigenvalueDense(self):
        M = self.getMassMatrix()
        self.w = scipy.linalg.eigvals(self.K.toarray(), M.toarray())
        self.w = np.sqrt(np.abs(self.w))
        return max(self.w)


def findEigenvalue(w, eigenvalueSearch, eigenvalue, wExact, exclude=[]):
    if eigenvalueSearch == 'nearest':
        wToSearch = w
        wToSearch[exclude] = 1e60
        wNum = fem1d.find_nearest(wToSearch, wExact)
        idx = fem1d.find_nearest_index(wToSearch, wExact)
        print("w index = %d" % idx)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
        idx = eigenvalue
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    return wNum, idx


def findEigenvector(v, search, index, iMatrix, system, vExact, exclude=[]):
    if search == 'nearest':
        minIndex = 0
        minError = 1e10
        nEigen = v.shape[1]
        for idx in range(nEigen):
            if idx in exclude:
                continue
            # print(idx)
            if np.linalg.norm(np.imag(v[:, idx])) == 0:
                eVector = iMatrix * system.getFullVector(np.real(v[:, idx]))
                eVector = eVector / eVector[0]
                eVector *= np.linalg.norm(vExact) / np.linalg.norm(eVector)
                error = np.linalg.norm(eVector - vExact) / np.linalg.norm(vExact)
                error2 = np.linalg.norm(-eVector - vExact) / np.linalg.norm(vExact)
                if error2 < error:
                    error = error2
                    eVector *= -1
                if error < minError:
                    # plot(nodesEval, [vExact, eVector])
                    minError = error
                    minIndex = idx
            else:
                print("Warning! Found complex eigenvector %d." % idx)

        print("v index = %d" % minIndex)
    elif search == 'energy':
        minIndex = 0
        minError = 1e10
        nEigen = v.shape[1]
        for idx in range(nEigen):
            if idx in exclude:
                continue
            # print(idx)
            if np.linalg.norm(np.imag(v[:, idx])) == 0:
                geVector = iMatrix * system.getFullVector(np.real(v[:, idx]))
                geVector *= np.linalg.norm(vExact) / np.linalg.norm(geVector)
                error = np.linalg.norm(geVector - vExact) / np.linalg.norm(vExact)
                error2 = np.linalg.norm(-geVector - vExact) / np.linalg.norm(vExact)
                if error2 < error:
                    error = error2
                    geVector *= -1
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
    nt = int(tMax/dt + 0.5)
    if dt > critDeltaT * safety:
        dt = critDeltaT * safety
        nt = int(tMax / dt + 0.5)
        dt = tMax / nt
    return dt, nt


# Plot animation
def postProcessTimeDomainSolution(study, evalNodes, evalU, tMax, nt, animationSpeed=4, factor=1.0):
    figure, ax = plt.subplots()
    ax.set_xlim(study.grid.left, study.grid.right)
    ax.set_ylim(-2, 2)

    config = study.config
    ax.plot([config.left+config.extra, config.left+config.extra], [-0.1, 0.1], '--', label='left boundary')
    ax.plot([config.right-config.extra, config.right-config.extra], [-0.1, 0.1], '--', label='right boundary')

    ax.plot(evalNodes, evalU[1], '--', label='initial condition')

    line2, = ax.plot(0, 0, "-*", label='numerical')
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
        line2.set_ydata(evalU[step]*factor)
        #line2.set_xdata(evalNodes + evalU[step])

    frames = np.linspace(0, tMax, round(tMax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000 / 60, repeat=False)
    plt.show()

