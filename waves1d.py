import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import lagrange
import bspline

from gll import *


class UniformGrid:
    def __init__(self, left, right, nElements):
        self.left = left
        self.right = right
        self.length = right - left
        self.elementSize = self.length / nElements
        self.nElements = nElements

    def pos(self, iElement, localCoord):
        return self.left + self.elementSize * (iElement + (localCoord + 1) / 2)

    def elementIndex(self, globalPos):
        return min(self.nElements - 1, int((globalPos - self.left) / self.length * self.nElements))

    def localPos(self, globalPos):
        return -1 + 2 * (globalPos - self.left - self.elementIndex(globalPos) * self.elementSize) / self.elementSize


class Domain:
    def __init__(self, alphaFunc):
        self.alphaFunc = alphaFunc

    def alpha(self, globalPos):
        return self.alphaFunc(globalPos)


class SpaceTreeQuadrature:
    def __init__(self, grid, localPointsAndWeights, domain, depth):
        self.grid = grid
        self.localPoints = localPointsAndWeights[0]
        self.localWeights = localPointsAndWeights[1]
        self.nPoints = len(self.localPoints)
        self.domain = domain
        self.depth = depth

        self.points = [None] * grid.nElements
        self.weights = [None] * grid.nElements
        self.cuts = [None] * grid.nElements

        for iElement in range(grid.nElements):
            result = self.createPointsAndWeights(grid.pos(iElement, -1), grid.pos(iElement, 1))
            self.points[iElement] = result[0]
            self.weights[iElement] = result[1]
            self.cuts[iElement] = result[2]

    def createPointsAndWeights(self, x1, x2, cuts=[], level=0):
        d = x2 - x1;
        if self.domain.alpha(x1) == self.domain.alpha(x2) or level >= self.depth:
            points = [0] * self.nPoints
            weights = [0] * self.nPoints
            for j in range(self.nPoints):
                points[j] = x1 + d * 0.5 * (self.localPoints[j] + 1)
                weights[j] = self.localWeights[j] * d / 2
            return points, weights, cuts
        else:
            cuts = cuts + [x1 + d / 2]
            pointsL, weightsL, cuts = self.createPointsAndWeights(x1, x1 + d / 2, cuts, level + 1)
            pointsR, weightsR, cuts = self.createPointsAndWeights(x1 + d / 2, x2, cuts, level + 1)
        return pointsL + pointsR, weightsL + weightsR, cuts


def createKnotVector(grid, p, k):
    extra = grid.length / grid.nElements * p
    t = np.linspace(grid.left - extra, grid.right + extra, grid.nElements + 1 + 2 * p)
    for i in range(p + 1):
        t[i] = grid.left
        t[-i - 1] = grid.right

    lower = min(p - k - 1, p - 1)
    for i in range(grid.nElements - 1):
        for j in range(lower):
            t = np.insert(t, p + grid.nElements - i - 1, t[p + grid.nElements - i - 1])

    return t


class SplineAnsatz:
    def __init__(self, grid, p, k):
        self.grid = grid
        self.p = p
        self.k = k
        self.knots = createKnotVector(grid, p, k)

    def spanIndex(self, iElement):
        return self.p + iElement * (self.p - self.k)

    def evaluate(self, pos, order, iElement):
        iSpan = self.spanIndex(iElement)
        return bspline.evaluateBSplineBases(iSpan, pos, self.p, order, self.knots)

    def locationMap(self, iElement):
        iShape = iElement * (self.p - self.k)
        return range(iShape, iShape + self.p + 1)

    def nDof(self):
        return self.grid.nElements * (self.p - self.k) + self.k + 1

    def interpolate(self, pos, globalVector):
        iElement = self.grid.elementIndex(pos)
        iSpan = self.spanIndex(iElement)
        basis = bspline.evaluateBSplineBases(iSpan, pos, self.p, 0, self.knots)
        lm = self.locationMap(iElement)
        return np.array(basis).dot(globalVector[lm])

    def interpolationMatrix(self, points):
        n = len(points)
        nVal = (self.p + 1)
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        val = np.zeros(nVal * n)
        for i in range(n):
            iElement = self.grid.elementIndex(points[i])
            iSpan = self.spanIndex(iElement)
            basis = bspline.evaluateBSplineBases(iSpan, points[i], self.p, 0, self.knots)
            lm = self.locationMap(iElement)
            pslice = slice(nVal * i, nVal * (i + 1))
            row[pslice] = i
            col[pslice] = lm
            val[pslice] = basis[0]
        return scipy.sparse.coo_matrix((val, (row, col)), shape=(n, self.nDof())).tocsc()


class LagrangeAnsatz:
    def __init__(self, grid, points):
        self.grid = grid
        self.points = points
        self.p = len(points) - 1
        self.knots = np.linspace(grid.left, grid.right, grid.nElements + 1)

        # lagrangeValues = np.identity(self.p + 1)
        # lagrange = lambda i : scipy.interpolate.lagrange(points, lagrangeValues[i])
        # self.shapesDiff0 = [lagrange(i) for i in range( self.p + 1 )]
        # self.shapesDiff1 = [np.polyder(shape) for shape in self.shapesDiff0];

    def evaluate(self, pos, order, iElement):
        basis = lagrange.evaluateLagrangeBases(iElement, pos, self.points, order, self.knots)
        return basis

    def locationMap(self, iElement):
        iShape = iElement * self.p
        return range(iShape, iShape + self.p + 1)

    def nDof(self):
        return self.grid.nElements * self.p + 1

    def interpolate(self, pos, globalVector):
        iElement = self.grid.elementIndex(pos)
        basis = lagrange.evaluateLagrangeBases(iElement, pos, self.points, 0, self.knots)
        lm = self.locationMap(iElement)
        return np.array(basis).dot(globalVector[lm])

    def interpolationMatrix(self, points):
        n = len(points)
        nVal = (self.p + 1)
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        val = np.zeros(nVal * n)
        for i in range(n):
            iElement = self.grid.elementIndex(points[i])
            basis = lagrange.evaluateLagrangeBases(iElement, points[i], self.points, 0, self.knots)
            lm = self.locationMap(iElement)
            pSlice = slice(nVal * i, nVal * (i + 1))
            row[pSlice] = i
            col[pSlice] = lm
            val[pSlice] = basis[0]
        return scipy.sparse.coo_matrix((val, (row, col)), shape=(n, self.nDof())).tocsc()


class TripletSystem:
    def __init__(self, ansatz, valM, valMHRZ, valMRS, valK, row, col, F):
        self.ansatz = ansatz
        self.valM = valM
        self.valMHRZ = valMHRZ
        self.valMRS = valMRS
        self.valK = valK
        self.row = row
        self.col = col
        self.F = F

    @classmethod
    def fromOneQuadrature(cls, ansatz, quadrature, bodyLoad=lambda x: 0.0):

        p = ansatz.p
        grid = ansatz.grid
        n = grid.nElements
        alpha = quadrature.domain.alpha

        nVal = (p + 1) * (p + 1)
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        valM = np.zeros(nVal * n)
        valMRS = np.zeros(nVal * n)
        valMHRZ = np.zeros(nVal * n)
        valK = np.zeros(nVal * n)
        F = np.zeros((ansatz.nDof(),))

        for iElement in range(n):
            lm = ansatz.locationMap(iElement)
            Me = np.zeros((p + 1, p + 1))
            Ke = np.zeros((p + 1, p + 1))
            Fe = np.zeros(p + 1)
            points = quadrature.points[iElement]
            weights = quadrature.weights[iElement]
            mass = 0
            for j in range(len(points)):
                shapes = ansatz.evaluate(points[j], 1, iElement)
                N = np.asarray(shapes[0])
                B = np.asarray(shapes[1])
                Me += np.outer(N, N) * weights[j] * alpha(points[j])
                Ke += np.outer(B, B) * weights[j] * alpha(points[j])
                Fe += N * bodyLoad(points[j]) * weights[j] * alpha(points[j])
                mass += weights[j] * alpha(points[j])

            eSlice = slice(nVal * iElement, nVal * (iElement + 1))
            row[eSlice] = np.broadcast_to(lm, (p + 1, p + 1)).T.ravel()
            col[eSlice] = np.broadcast_to(lm, (p + 1, p + 1)).ravel()
            valK[eSlice] = Ke.ravel()
            valM[eSlice] = Me.ravel()
            F[lm] += Fe

            diagMe = np.zeros(Me.shape)
            for iEntry in range(Me.shape[0]):
                diagMe[iEntry, iEntry] = sum(Me[iEntry, :])
            valMRS[eSlice] = diagMe.ravel()
            # print("Lump error: %e" % np.linalg.norm(diagMe - Me))

            diagMe = np.zeros(Me.shape)
            sumMe = 0
            for iEntry in range(Me.shape[0]):
                diagMe[iEntry, iEntry] = Me[iEntry, iEntry]
                sumMe += Me[iEntry, iEntry]
            c = mass * 1 / sumMe
            diagMe = diagMe * c
            valMHRZ[eSlice] = diagMe.ravel()
            # print("Lump error: %e" % np.linalg.norm(diagMe - Me))

        return cls(ansatz, valM, valMHRZ, valMRS, valK, row, col, F)

    @classmethod
    def fromTwoQuadratures(cls, ansatz, quadratureM, quadratureK, bodyLoad=lambda x: 0.0):

        p = ansatz.p
        grid = ansatz.grid
        n = grid.nElements
        alpha = quadratureM.domain.alpha

        nVal = (p + 1) * (p + 1)
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        valM = np.zeros(nVal * n)
        valMRS = np.zeros(nVal * n)
        valMHRZ = np.zeros(nVal * n)
        valK = np.zeros(nVal * n)
        F = np.zeros((ansatz.nDof(),))

        for iElement in range(n):
            lm = ansatz.locationMap(iElement)
            # print("ELEMENT " + str(i) + "lm: " + str(lm))
            Me = np.zeros((p + 1, p + 1))
            Ke = np.zeros((p + 1, p + 1))
            Fe = np.zeros(p + 1)
            pointsM = quadratureM.points[iElement]
            weightsM = quadratureM.weights[iElement]
            pointsK = quadratureK.points[iElement]
            weightsK = quadratureK.weights[iElement]
            for j in range(len(pointsM)):
                # print("M p: " + str(j) + " " + str(pointsM[j]))
                shapes = ansatz.evaluate(pointsM[j], 1, iElement)
                N = np.asarray(shapes[0])
                B = np.asarray(shapes[1])
                Me += np.outer(N, N) * weightsM[j] * alpha(pointsM[j])

            mass = 0
            for j in range(len(pointsK)):
                # print("K p: " + str(j) + " " + str(pointsK[j]))
                shapes = ansatz.evaluate(pointsK[j], 1, iElement)
                N = np.asarray(shapes[0])
                B = np.asarray(shapes[1])
                Ke += np.outer(B, B) * weightsK[j] * alpha(pointsK[j])
                Fe += N * bodyLoad(pointsK[j]) * weightsK[j] * alpha(pointsK[j])
                mass += weightsK[j] * alpha(pointsK[j])

            eSlice = slice(nVal * iElement, nVal * (iElement + 1))
            row[eSlice] = np.broadcast_to(lm, (p + 1, p + 1)).T.ravel()
            col[eSlice] = np.broadcast_to(lm, (p + 1, p + 1)).ravel()
            valK[eSlice] = Ke.ravel()
            valM[eSlice] = Me.ravel()
            F[lm] += Fe

            diagMe = np.zeros(Me.shape)
            for iEntry in range(Me.shape[0]):
                diagMe[iEntry, iEntry] = sum(Me[iEntry, :])
            valMRS[eSlice] = diagMe.ravel()
            # print("Lump error: %e" % np.linalg.norm(diagMe - Me))

            diagMe = np.zeros(Me.shape)
            sumMe = 0
            for iEntry in range(Me.shape[0]):
                diagMe[iEntry, iEntry] = Me[iEntry, iEntry]
                sumMe += Me[iEntry, iEntry]
            c = mass * 1 / sumMe
            diagMe = diagMe * c
            valMHRZ[eSlice] = diagMe.ravel()

        return cls(ansatz, valM, valMHRZ, valMRS, valK, row, col, F)

    def nDof(self):
        return int(max(self.row) + 1)

    def findZeroDof(self, tol=0):
        nDof = self.nDof()
        diag = [0] * nDof
        nVals = len(self.row)
        for i in range(nVals):
            iRow = self.row[i]
            diag[iRow] += self.valM[i]
        self.zeroDof = []
        self.dofMap = [0] * nDof
        nNonZeroDof = 0
        self.nonZeroDof = []
        for i in range(nDof):
            if diag[i] <= tol:
                self.zeroDof.append(i)
            else:
                self.dofMap[i] = nNonZeroDof
                nNonZeroDof += 1
                self.nonZeroDof.append(i)

    def getReducedRowAndCol(self):
        if hasattr(self, 'dofMap'):
            nVal = len(self.row)
            row = [0] * nVal
            col = [0] * nVal
            for i in range(nVal):
                row[i] = self.dofMap[self.row[i]]
                col[i] = self.dofMap[self.col[i]]
        else:
            row = self.row
            col = self.col
        return row, col

    def createSparseMatrices(self, returnHRZ=False, returnRS=False):
        row, col = self.getReducedRowAndCol()
        M = scipy.sparse.coo_matrix((self.valM, (row, col))).tocsc()
        K = scipy.sparse.coo_matrix((self.valK, (row, col))).tocsc()
        if returnHRZ is False and returnRS is False:
            return M, K
        elif returnHRZ is True and returnRS is False:
            MHRZ = scipy.sparse.coo_matrix((self.valMHRZ, (row, col))).tocsc()
            return M, K, MHRZ
        elif returnHRZ is False and returnRS is True:
            MRS = scipy.sparse.coo_matrix((self.valMRS, (row, col))).tocsc()
            return M, K, MRS
        else:
            MHRZ = scipy.sparse.coo_matrix((self.valMHRZ, (row, col))).tocsc()
            MRS = scipy.sparse.coo_matrix((self.valMRS, (row, col))).tocsc()
            return M, K, MHRZ, MRS

    def getFullVector(self, reducedVector):
        fullVector = np.zeros(self.ansatz.nDof())
        fullVector[self.nonZeroDof] = reducedVector
        return fullVector

    def getReducedVector(self, fullVector):
        return fullVector[self.nonZeroDof]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def plot(ptx, pty):
    figure, ax = plt.subplots()
    ax.plot(ptx, pty)
    plt.show()


def getReducedVector(systemF, systemS):
    FF = systemF.getReducedVector(systemF.F)
    FS = systemS.getReducedVector(systemS.F)
    return np.concatenate((FF, FS), axis=0)


def createSparseMatrices(systemF, systemS, boundary):
    rowF, colF = systemF.getReducedRowAndCol()
    rowS, colS = systemS.getReducedRowAndCol()
    nDofF = systemF.nDof()
    rowS += nDofF
    colS += nDofF
    row = np.concatenate((rowF, rowS), axis=0)
    col = np.concatenate((colF, colS), axis=0)
    valM = np.concatenate((systemF.valM, systemS.valM), axis=0)
    valK = np.concatenate((systemF.valK, systemS.valK), axis=0)
    M = scipy.sparse.coo_matrix((valM, (row, col))).tocsc()
    K = scipy.sparse.coo_matrix((valK, (row, col))).tocsc()

    iElementF = systemF.ansatz.grid.elementIndex(boundary)
    iElementS = systemS.ansatz.grid.elementIndex(boundary)

    shapesF = systemF.ansatz.evaluate(boundary, 0, iElementF)
    shapesS = systemS.ansatz.evaluate(boundary, 0, iElementS)

    lmF = systemF.ansatz.locationMap(iElementF)
    lmS = systemF.ansatz.locationMap(iElementS)

    Ce = np.outer(shapesF, shapesS)

    pF = systemF.ansatz.p
    pS = systemS.ansatz.p

    nVal = (pF + 1) * (pS + 1)
    rowC = np.zeros(nVal * 2, dtype=np.uint)
    colC = np.zeros(nVal * 2, dtype=np.uint)
    valC = np.zeros(nVal * 2)

    eSliceF = slice(0, nVal)
    eSliceS = slice(nVal, 2*nVal)

    rowC[eSliceF] = np.broadcast_to(lmF, (pS + 1, pF + 1)).T.ravel()
    colC[eSliceF] = np.broadcast_to(lmS, (pF + 1, pS + 1)).ravel()
    valC[eSliceF] = Ce.ravel()

    rowC[eSliceS] = np.broadcast_to(lmS, (pF + 1, pS + 1)).T.ravel()
    colC[eSliceS] = np.broadcast_to(lmF, (pS + 1, pF + 1)).ravel()
    valC[eSliceS] = Ce.T.ravel()

    C = scipy.sparse.coo_matrix((valC, (rowC, colC))).tocsc()

    return M, K, C


def createAnsatz(ansatzType, continuity, p, grid):
    if ansatzType == 'Spline':
        k = eval(continuity)
        k = max(0, min(k, p - 1))
        ansatz = SplineAnsatz(grid, p, k)
    elif ansatzType == 'Lagrange':
        gllPoints = GLL(p + 1)
        # gllPoints[0][0] += 1e-16
        # gllPoints[0][-1] -=1e-16
        ansatz = LagrangeAnsatz(grid, gllPoints[0])
    else:
        print("Error! Choose ansatzType 'Spline' or 'Lagrange'")
        return None

    return ansatz
