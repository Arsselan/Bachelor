import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import lagrange
import bspline
import gll


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

    def nShapesPerElement(self):
        return self.p + 1

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

    def interpolationMatrix(self, points, order=0):
        n = len(points)
        nVal = (self.p + 1)
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        val = np.zeros(nVal * n)
        for iPoint in range(n):
            iElement = self.grid.elementIndex(points[iPoint])
            iSpan = self.spanIndex(iElement)
            basis = bspline.evaluateBSplineBases(iSpan, points[iPoint], self.p, order, self.knots)
            lm = self.locationMap(iElement)
            pslice = slice(nVal * iPoint, nVal * (iPoint + 1))
            row[pslice] = iPoint
            col[pslice] = lm
            val[pslice] = basis[order]
        return scipy.sparse.coo_matrix((val, (row, col)), shape=(n, self.nDof())).tocsc()


def getGrevilleAbscissas(knots, p):
    ng = len(knots) - p - 1
    ga = np.zeros(ng)
    for i in range(ng):
        ga[i] = np.sum(knots[i+1:i+p+1]) / p
    return ga


class InterpolatorySplineAnsatz:
    def __init__(self, grid, p, k):
        self.grid = grid
        self.p = p
        self.k = k
        self.knots = createKnotVector(grid, p, k)
        self.greville = getGrevilleAbscissas(self.knots, p)

        if k != p-1:
            print("Warning! InterpolatorySplineAnsatz is only tested for maximum continuity.")

        self.T = self.splineInterpolationMatrix(self.greville, 0).toarray().T
        # self.T[T < 1e-30] = 0
        self.invT = np.linalg.inv(self.T)

    def spanIndex(self, iElement):
        return self.p + iElement * (self.p - self.k)

    def evaluate(self, pos, order, iElement):
        iSpan = self.spanIndex(iElement)
        basis = bspline.evaluateBSplineBases(iSpan, pos, self.p, order, self.knots)
        tBasis = [0] * (order+1)
        invTAB = self.invT[:, self.splineLocationMap(iElement)]
        for i in range(order+1):
            tBasis[i] = invTAB.dot(basis[i])
        return tBasis

    def nShapesPerElement(self):
        return self.nDof()

    def locationMap(self, iElement):
        return range(self.nDof())

    def splineLocationMap(self, iElement):
        iShape = iElement * (self.p - self.k)
        return range(iShape, iShape + self.p + 1)

    def nDof(self):
        return self.grid.nElements * (self.p - self.k) + self.k + 1

    def interpolate(self, pos, globalVector):
        splineVector = self.T.T.dot(globalVector)
        iElement = self.grid.elementIndex(pos)
        iSpan = self.spanIndex(iElement)
        basis = bspline.evaluateBSplineBases(iSpan, pos, self.p, 0, self.knots)
        lm = self.splineLocationMap(iElement)
        return np.array(basis).dot(splineVector[lm])

    def interpolationMatrix(self, points, order=0):
        n = len(points)
        nVal = self.nDof()
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        val = np.zeros(nVal * n)
        for iPoint in range(n):
            iElement = self.grid.elementIndex(points[iPoint])
            iSpan = self.spanIndex(iElement)
            basis = bspline.evaluateBSplineBases(iSpan, points[iPoint], self.p, order, self.knots)
            invTAB = self.invT[:, self.splineLocationMap(iElement)]
            tBasis = invTAB.dot(basis[order])
            lm = self.locationMap(iElement)
            pSlice = slice(nVal * iPoint, nVal * (iPoint + 1))
            row[pSlice] = iPoint
            col[pSlice] = lm
            val[pSlice] = tBasis
        return scipy.sparse.coo_matrix((val, (row, col)), shape=(n, self.nDof())).tocsc()

    def splineInterpolationMatrix(self, points, order=0):
        n = len(points)
        nVal = (self.p + 1)
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        val = np.zeros(nVal * n)
        for iPoint in range(n):
            iElement = self.grid.elementIndex(points[iPoint])
            iSpan = self.spanIndex(iElement)
            basis = bspline.evaluateBSplineBases(iSpan, points[iPoint], self.p, order, self.knots)
            lm = self.splineLocationMap(iElement)
            pslice = slice(nVal * iPoint, nVal * (iPoint + 1))
            row[pslice] = iPoint
            col[pslice] = lm
            val[pslice] = basis[order]
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

    def nShapesPerElement(self):
        return self.p + 1

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

    def interpolationMatrix(self, points, order=0):
        n = len(points)
        nVal = (self.p + 1)
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        val = np.zeros(nVal * n)
        for iPoint in range(n):
            iElement = self.grid.elementIndex(points[iPoint])
            basis = lagrange.evaluateLagrangeBases(iElement, points[iPoint], self.points, order, self.knots)
            lm = self.locationMap(iElement)
            pSlice = slice(nVal * iPoint, nVal * (iPoint + 1))
            row[pSlice] = iPoint
            col[pSlice] = lm
            val[pSlice] = basis[order]
        return scipy.sparse.coo_matrix((val, (row, col)), shape=(n, self.nDof())).tocsc()

