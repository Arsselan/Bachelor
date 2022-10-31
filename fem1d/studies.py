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
        self.quadratureM = quadratureM
        self.quadratureK = quadratureK
        self.system = system

        self.w = np.zeros(self.M.shape[0])

    def runDense(self):
        if self.config.mass == 'CON':
            # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, M, which='SM', return_eigenvectors=False)
            self.w = scipy.linalg.eigvals(self.K.toarray(), self.M.toarray())
        elif self.config.mass == 'HRZ':
            # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MHRZ, which='SM', return_eigenvectors=False)
            self.w = scipy.linalg.eigvals(self.K.toarray(), self.MHRZ.toarray())
        elif self.config.mass == 'RS':
            # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MRS, which='SM', return_eigenvectors=False)
            self.w = scipy.linalg.eigvals(self.K.toarray(), self.MRS.toarray())
        else:
            print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

        self.w = np.sqrt(np.abs(self.w))
        self.w = np.sort(self.w)
        return max(self.w)

    def runSparse(self):
        if self.config.mass == 'CON':
            w = scipy.sparse.linalg.eigs(self.K, self.K.shape[0] - 2, self.M, which='SM', return_eigenvectors=False)
        elif self.config.mass == 'HRZ':
            w = scipy.sparse.linalg.eigs(self.K, self.K.shape[0] - 2, self.MHRZ, which='SM', return_eigenvectors=False)
        elif self.config.mass == 'RS':
            w = scipy.sparse.linalg.eigs(self.K, self.K.shape[0] - 2, self.MRS, which='SM', return_eigenvectors=False)
        else:
            print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

        self.w = np.sqrt(np.abs(self.w))
        self.w = np.sort(self.w)
        return max(self.w)


def findEigenvalue(w, eigenvalueSearch, eigenvalue, wExact):
    if eigenvalueSearch == 'nearest':
        wNum = find_nearest(w, wExact)
        idx = find_nearest_index(w, wExact)
        print("index = %d" % idx)
    elif eigenvalueSearch == 'number':
        wNum = w[eigenvalue]
    else:
        print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

    if np.imag(wNum) > 0:
        print("Warning! Chosen eigenvalue has imaginary part.")

    return wNum
