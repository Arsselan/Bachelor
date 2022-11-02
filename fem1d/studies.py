from waves1d import *


class StudyConfig:

    def __init__(self, left, right, extra, n, p, ansatzType, continuity, mass, depth, stabilize, spectral, dual):
        self.left = left
        self.right = right
        self.extra = extra

        self.n = n
        self.p = p
        self.ansatzType = ansatzType
        self.continuity = continuity
        self.mass = mass
        self.spectral = spectral

        self.dual = dual
        self.stabilize = stabilize
        self.depth = depth

        if ansatzType == 'Lagrange':
            self.continuity = '0'

        if ansatzType == 'Spline':
            self.spectral = False


class EigenvalueStudy:

    def __init__(self, config):

        self.config = config

        # create grid and domain
        grid = UniformGrid(config.left, config.right, config.n)

        left = config.left
        right = config.right
        extra = config.extra
        stabilize = config.stabilize

        def alpha(x):
            if left + extra <= x <= right - extra:
                return 1
            return stabilize

        domain = Domain(alpha)

        # create ansatz and quadrature points
        ansatz = createAnsatz(config.ansatzType, config.continuity, config.p, grid)

        # gaussPointsM = gll.computeGllPoints(p + 1)
        gaussPointsM = GLL(config.p + 1)
        quadratureM = SpaceTreeQuadrature(grid, gaussPointsM, domain, config.depth)

        gaussPointsK = np.polynomial.legendre.leggauss(config.p + 1)
        quadratureK = SpaceTreeQuadrature(grid, gaussPointsK, domain, config.depth)

        # create system
        if config.spectral:
            system = TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK)
        else:
            if config.dual:
                print("DUAL! %e" % extra)
                system = TripletSystem.fromOneQuadratureWithDualBasis(ansatz, quadratureK)
            else:
                system = TripletSystem.fromOneQuadrature(ansatz, quadratureK)

        #    system.findZeroDof(-1e60, [0, 1, system.nDof()-2, system.nDof()-1])
        system.findZeroDof(0)
        if len(system.zeroDof) > 0:
            print("Warning! There were %d zero dof found." % len(system.zeroDof))
            #print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

        # get matrices
        self.M, self.K, self.MHRZ, self.MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)

        self.grid = grid
        self.domain = domain
        self.ansatz = ansatz
        self.quadratureM = quadratureM
        self.quadratureK = quadratureK
        self.system = system

        self.w = 0
        self.v = 0

    def runDense(self, computeEigenvectors=False, sort=False):
        if computeEigenvectors:
            if self.config.mass == 'CON':
                self.w, self.v = scipy.linalg.eig(self.K.toarray(), self.M.toarray(), right=True)
            elif self.config.mass == 'HRZ':
                self.w, self.v = scipy.linalg.eig(self.K.toarray(), self.MHRZ.toarray(), right=True)
            elif self.config.mass == 'RS':
                self.w, self.v = scipy.linalg.eig(self.K.toarray(), self.MRS.toarray(), right=True)
            else:
                print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")
        else:
            if self.config.mass == 'CON':
                self.w = scipy.linalg.eigvals(self.K.toarray(), self.M.toarray())
            elif self.config.mass == 'HRZ':
                self.w = scipy.linalg.eigvals(self.K.toarray(), self.MHRZ.toarray())
            elif self.config.mass == 'RS':
                self.w = scipy.linalg.eigvals(self.K.toarray(), self.MRS.toarray())
            else:
                print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

        self.w = np.sqrt(np.abs(self.w))

        if sort:
            idx = self.w.argsort()[::1]
            self.w = self.w[idx]
            if computeEigenvectors:
                self.v = self.v[:, idx]

        return max(self.w)

    def runSparse(self, computeEigenvectors=False, sort=False):
        nEigen = self.K.shape[0] - 2

        if computeEigenvectors:
            if self.config.mass == 'CON':
                self.w, self.v = scipy.sparse.linalg.eigs(self.K, nEigen, self.M, which='SM', return_eigenvectors=True)
            elif self.config.mass == 'HRZ':
                self.w, self.v = scipy.sparse.linalg.eigs(self.K, nEigen, self.MHRZ, which='SM', return_eigenvectors=True)
            elif self.config.mass == 'RS':
                self.w, self.v = scipy.sparse.linalg.eigs(self.K, nEigen, self.MRS, which='SM', return_eigenvectors=True)
            else:
                print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")
        else:
            if self.config.mass == 'CON':
                self.w = scipy.sparse.linalg.eigs(self.K, self.K.shape[0] - 2, self.M, which='SM', return_eigenvectors=False)
            elif self.config.mass == 'HRZ':
                self.w = scipy.sparse.linalg.eigs(self.K, self.K.shape[0] - 2, self.MHRZ, which='SM', return_eigenvectors=False)
            elif self.config.mass == 'RS':
                self.w = scipy.sparse.linalg.eigs(self.K, self.K.shape[0] - 2, self.MRS, which='SM', return_eigenvectors=False)
            else:
                print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

        self.w = np.sqrt(np.abs(self.w))

        if sort:
            idx = self.w.argsort()[::1]
            self.w = self.w[idx]
            if computeEigenvectors:
                self.v = self.v[:, idx]

        return max(self.w)

    def computeLargestEigenvalueSparse(self):
        if self.config.mass == 'CON':
            self.w = scipy.sparse.linalg.eigs(self.K, 1, self.M, which='LM', return_eigenvectors=False)
        elif self.config.mass == 'HRZ':
            self.w = scipy.sparse.linalg.eigs(self.K, 1, self.MHRZ, which='LM', return_eigenvectors=False)
        elif self.config.mass == 'RS':
            self.w = scipy.sparse.linalg.eigs(self.K, 1, self.MRS, which='LM', return_eigenvectors=False)
        else:
            print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

        self.w = np.sqrt(np.abs(self.w))

        return max(self.w)


def findEigenvalue(w, eigenvalueSearch, eigenvalue, wExact):
    if eigenvalueSearch == 'nearest':
        wNum = find_nearest(w, wExact)
        idx = find_nearest_index(w, wExact)
        print("w index = %d" % idx)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    return wNum


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

    return vNum


