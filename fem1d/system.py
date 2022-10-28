import scipy.sparse
import scipy.sparse.linalg

from fem1d.ansatz import *


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
    def fromOneQuadrature(cls, ansatz, quadrature, bodyLoad=lambda x: 0.0, selectiveLumping=False):

        p = ansatz.p
        grid = ansatz.grid
        n = grid.nElements
        alpha = quadrature.domain.alpha

        nShapesPerElement = ansatz.nShapesPerElement()
        nVal = nShapesPerElement * nShapesPerElement
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        valM = np.zeros(nVal * n)
        valMRS = np.zeros(nVal * n)
        valMHRZ = np.zeros(nVal * n)
        valK = np.zeros(nVal * n)
        F = np.zeros((ansatz.nDof(),))

        for iElement in range(n):
            lm = ansatz.locationMap(iElement)
            Me = np.zeros((nShapesPerElement, nShapesPerElement))
            Ke = np.zeros((nShapesPerElement, nShapesPerElement))
            Fe = np.zeros(nShapesPerElement)
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
            row[eSlice] = np.broadcast_to(lm, (nShapesPerElement, nShapesPerElement)).T.ravel()
            col[eSlice] = np.broadcast_to(lm, (nShapesPerElement, nShapesPerElement)).ravel()
            valK[eSlice] = Ke.ravel()
            valM[eSlice] = Me.ravel()
            F[lm] += Fe

            if selectiveLumping is False or len(quadrature.cuts[iElement]) > 0:
                diagMe = np.zeros(Me.shape)
                for iEntry in range(Me.shape[0]):
                    diagMe[iEntry, iEntry] = sum(Me[iEntry, :])
                valMRS[eSlice] = diagMe.ravel()
                # print("Lump error RS: %e" % np.linalg.norm(diagMe - Me))

                diagMe = np.zeros(Me.shape)
                sumMe = 0
                for iEntry in range(Me.shape[0]):
                    diagMe[iEntry, iEntry] = Me[iEntry, iEntry]
                    sumMe += Me[iEntry, iEntry]
                if sumMe > 0:
                    c = mass * 1 / sumMe
                else:
                    c = 0
                diagMe = diagMe * c
                valMHRZ[eSlice] = diagMe.ravel()
                # print("Lump error HRZ: %e" % np.linalg.norm(diagMe - Me))
            else:
                valMRS[eSlice] = Me.ravel()
                valMHRZ[eSlice] = Me.ravel()

        return cls(ansatz, valM, valMHRZ, valMRS, valK, row, col, F)

    @classmethod
    def fromTwoQuadratures(cls, ansatz, quadratureM, quadratureK, bodyLoad=lambda x: 0.0, selectiveLumping=False):

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
            if sumMe > 0:
                c = mass * 1 / sumMe
            else:
                c = 0
            diagMe = diagMe * c
            valMHRZ[eSlice] = diagMe.ravel()

        return cls(ansatz, valM, valMHRZ, valMRS, valK, row, col, F)

    @classmethod
    def fromOneQuadratureWithDualBasis(cls, ansatz, quadrature, bodyLoad=lambda x: 0.0, selectiveLumping=False):

        grid = ansatz.grid
        n = grid.nElements
        alpha = quadrature.domain.alpha

        nShapesPerElement = ansatz.nShapesPerElement()
        nVal = nShapesPerElement * nShapesPerElement
        row = np.zeros(nVal * n, dtype=np.uint)
        col = np.zeros(nVal * n, dtype=np.uint)
        valM = np.zeros(nVal * n)
        valMRS = np.zeros(nVal * n)
        valMHRZ = np.zeros(nVal * n)
        valK = np.zeros(nVal * n)
        F = np.zeros((ansatz.nDof(),))

        for iElement in range(n):
            lm = ansatz.locationMap(iElement)
            Me = np.zeros((nShapesPerElement, nShapesPerElement))
            Ke = np.zeros((nShapesPerElement, nShapesPerElement))
            Fe = np.zeros(nShapesPerElement)
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

            # row summing
            MeRS = np.zeros(Me.shape)
            for iEntry in range(Me.shape[0]):
                MeRS[iEntry, iEntry] = sum(Me[iEntry, :])

            # hrz lumping
            MeHRZ = np.zeros(Me.shape)
            sumMe = 0
            for iEntry in range(Me.shape[0]):
                MeHRZ[iEntry, iEntry] = Me[iEntry, iEntry]
                sumMe += Me[iEntry, iEntry]
            if sumMe > 0:
                c = mass * 1 / sumMe
            else:
                c = 0
            MeHRZ = MeHRZ * c

            # dual matrices
            if mass > 0:
                dualMat = np.linalg.inv(Me).dot(MeRS)
                #check = np.linalg.norm(Me.dot(dualMat)-MeRS)
                #print("Check: %e" % check)
                Me = MeRS
                Ke = Ke.dot(dualMat)
                Fe = Fe.dot(dualMat)

            # assembly
            eSlice = slice(nVal * iElement, nVal * (iElement + 1))
            row[eSlice] = np.broadcast_to(lm, (nShapesPerElement, nShapesPerElement)).T.ravel()
            col[eSlice] = np.broadcast_to(lm, (nShapesPerElement, nShapesPerElement)).ravel()
            valK[eSlice] = Ke.ravel()
            valM[eSlice] = Me.ravel()
            F[lm] += Fe

            if selectiveLumping is False or len(quadrature.cuts[iElement]) > 0:
                valMRS[eSlice] = MeRS.ravel()
                # print("Lump error RS: %e" % np.linalg.norm(MeRS - Me))
                valMHRZ[eSlice] = MeHRZ.ravel()
                # print("Lump error HRZ: %e" % np.linalg.norm(MeHRZ - Me))
            else:
                valMRS[eSlice] = Me.ravel()
                valMHRZ[eSlice] = Me.ravel()

        return cls(ansatz, valM, valMHRZ, valMRS, valK, row, col, F)

    def nDof(self):
        return int(max(self.row) + 1)

    def findZeroDof(self, tol=0, deleteDof=[]):
        nDof = self.nDof()
        diag = [0] * nDof
        nVals = len(self.row)
        for i in range(nVals):
            iRow = self.row[i]
            # diag[iRow] += abs(self.valM[i])
            diag[iRow] += self.valM[i]
        self.zeroDof = []
        self.dofMap = [0] * nDof
        nNonZeroDof = 0
        self.nonZeroDof = []
        for i in range(nDof):
            if diag[i] == tol or i in deleteDof:
                self.zeroDof.append(i)
            else:
                self.dofMap[i] = nNonZeroDof
                nNonZeroDof += 1
                self.nonZeroDof.append(i)

        for i in range(nVals):
            if self.row[i] in self.zeroDof or self.col[i] in self.zeroDof:
                self.valM[i] = 0
                self.valMHRZ[i] = 0
                self.valMRS[i] = 0
                self.valK[i] = 0

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


def getReducedVector(systemF, systemS):
    FF = systemF.getReducedVector(systemF.F)
    FS = systemS.getReducedVector(systemS.F)
    return np.concatenate((FF, FS), axis=0)


def createSparseMatrices(systemF, systemS):
    rowF, colF = systemF.getReducedRowAndCol()
    rowS, colS = systemS.getReducedRowAndCol()
    nDofF = systemF.nDof()
    for i in range(len(rowS)):
        rowS[i] += nDofF
        colS[i] += nDofF
    row = np.concatenate((rowF, rowS), axis=0)
    col = np.concatenate((colF, colS), axis=0)
    valM = np.concatenate((systemF.valM, systemS.valM), axis=0)
    valK = np.concatenate((systemF.valK, systemS.valK), axis=0)
    M = scipy.sparse.coo_matrix((valM, (row, col))).tocsc()
    K = scipy.sparse.coo_matrix((valK, (row, col))).tocsc()

    return M, K


def createCouplingMatrix(systemF, systemS, boundaries):
    nDofF = systemF.nDof()
    nDofS = systemS.nDof()
    nDof = nDofF + nDofS

    pF = systemF.ansatz.p
    pS = systemS.ansatz.p
    nBoundaries = len(boundaries)

    nVal = (pF + 1) * (pS + 1)
    rowC = np.zeros(nVal * 2 * nBoundaries, dtype=np.uint)
    colC = np.zeros(nVal * 2 * nBoundaries, dtype=np.uint)
    valC = np.zeros(nVal * 2 * nBoundaries)

    normal = 1
    for iBoundary in range(nBoundaries):
        boundary = boundaries[iBoundary]

        iElementF = systemF.ansatz.grid.elementIndex(boundary)
        iElementS = systemS.ansatz.grid.elementIndex(boundary)

        shapesF = systemF.ansatz.evaluate(boundary, 0, iElementF)
        shapesS = systemS.ansatz.evaluate(boundary, 0, iElementS)

        lmF = systemF.ansatz.locationMap(iElementF)
        lmS = np.array(systemF.ansatz.locationMap(iElementS)) + nDofF

        Ce = np.outer(shapesF, shapesS) * normal
        normal *= -1

        startF = iBoundary * 2 * nVal
        eSliceF = slice(startF, startF + nVal)
        eSliceS = slice(startF + nVal, startF + 2 * nVal)

        rowC[eSliceF] = np.broadcast_to(lmF, (pS + 1, pF + 1)).T.ravel()
        colC[eSliceF] = np.broadcast_to(lmS, (pF + 1, pS + 1)).ravel()
        valC[eSliceF] = Ce.ravel()

        rowC[eSliceS] = np.broadcast_to(lmS, (pF + 1, pS + 1)).T.ravel()
        colC[eSliceS] = np.broadcast_to(lmF, (pS + 1, pF + 1)).ravel()
        valC[eSliceS] = -Ce.T.ravel()

    C = scipy.sparse.coo_matrix((valC, (rowC, colC)), shape=(nDof, nDof)).tocsc()

    return C
