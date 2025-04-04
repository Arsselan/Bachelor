import scipy.sparse
import scipy.sparse.linalg

from fem1d.ansatz import *
from fem1d.matrices import *


class TripletSystem:
    def __init__(self, ansatz):
        self.ansatz = ansatz

        self.row = np.zeros(0)
        self.col = np.zeros(0)

        self.matrixValues = {}
        self.matrices = {}
        self.vectors = {}

        self.zeroDof = []
        self.dofMap = []
        self.nonZeroDof = []

        self.minNonZeroMass = 0

    def nDof(self):
        return int(max(self.row) + 1)

    def findZeroDof(self, tol=0, deleteDof=[]):
        nDof = self.nDof()
        diag = [0] * nDof
        nVals = len(self.row)
        for i in range(nVals):
            iRow = self.row[i]
            diag[iRow] += self.matrixValues['MHRZ'][i]
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
                for key in self.matrixValues.keys():
                    if not key == "modM":
                        self.matrixValues[key][i] = 0

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

#    def createSparseMatrices(self):
#        row, col = self.getReducedRowAndCol()
#        for key in self.matrices.keys():
#            self.matrices[key] = scipy.sparse.coo_matrix((self.matrixValues[key], (row, col))).tocsc()

    def createSparseMatrix(self, name):
        row, col = self.getReducedRowAndCol()
        self.matrices[name] = scipy.sparse.coo_matrix((self.matrixValues[name], (row, col))).tocsc()
        return self.matrices[name]

    def getFullVector(self, reducedVector):
        fullVector = np.zeros(self.ansatz.nDof())
        fullVector[self.nonZeroDof] = reducedVector
        return fullVector

    def getReducedVector(self, fullVector):
        return fullVector[self.nonZeroDof]


def getReducedVector(systemF, systemS):
    FF = systemF.getReducedVector(systemF.vectors['F'])
    FS = systemS.getReducedVector(systemS.vectors['F'])
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
    valM = np.concatenate((systemF.matrixValues['M'], systemS.matrixValues['M']), axis=0)
    valK = np.concatenate((systemF.matrixValues['K'], systemS.matrixValues['K']), axis=0)
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


def createNeumannVector(system, boundaries, normals, forces):
    nDof = system.nDof()
    F = np.zeros(nDof)

    p = system.ansatz.p
    nBoundaries = len(boundaries)

    for iBoundary in range(nBoundaries):
        boundary = boundaries[iBoundary]

        iElement = system.ansatz.grid.elementIndex(boundary)
        shapes = system.ansatz.evaluate(boundary, 0, iElement)
        lm = system.ansatz.locationMap(iElement)

        F[lm] = np.array(shapes[0]) * normals[iBoundary] * forces[iBoundary]

        #print(F)

    return F


def createQuadraturePointData(quadrature):
    import copy
    data = copy.deepcopy(quadrature.points)
    nElements = len(data)
    for iElement in range(nElements):
        nPoints = len(data[iElement])
        for iPoint in range(nPoints):
            data[iElement][iPoint] = 0
    return data


def computePlasticInnerLoadVector(c, epsYield, hardening, ansatz, quadrature, solution, epsPla):
    grid = ansatz.grid
    n = grid.nElements
    alpha = quadrature.domain.alpha
    F = np.zeros((ansatz.nDof(),))
    nShapesPerElement = ansatz.nShapesPerElement()
    maxAbsEps = 0
    minAbsEps = 1e10
    for iElement in range(n):
        lm = ansatz.locationMap(iElement)
        points = quadrature.points[iElement]
        weights = quadrature.weights[iElement]
        Ue = solution[lm]
        Fe = np.zeros(nShapesPerElement)
        #Ke = np.zeros((nShapesPerElement, nShapesPerElement))
        for j in range(len(points)):
            shapes = ansatz.evaluate(points[j], 1, iElement)
            B = np.asarray(shapes[1])
            eps = (B @ Ue.transpose())
            if abs(eps) > maxAbsEps:
                maxAbsEps = abs(eps)
            if abs(eps) < minAbsEps:
                    minAbsEps = abs(eps)
            epsEla = eps - epsPla[iElement][j]
            if epsEla > epsYield:
                epsPla[iElement][j] += epsEla - epsYield
            if epsEla < -epsYield:
                epsPla[iElement][j] += epsEla + epsYield
                #print("P", eps)
            epsEla = eps - epsPla[iElement][j]
            sigma = c*c*(epsEla + epsPla[iElement][j]*hardening)
            Fe += B.transpose() * sigma * weights[j] * alpha(points[j])
            #Ke += np.outer(B, B) * weights[j] * alpha(points[j])
        #Fe = Ke @ Ue
        F[lm] += Fe
    #print("max/min abs eps: %e / %e" % (maxAbsEps, minAbsEps))
    return F


def radialReturn(E, sigmaYield, K, eps, epsPla, alphaPla):
    sigmaTrial = E * (eps - epsPla)
    fTrial = abs(sigmaTrial) - (sigmaYield + K * alphaPla)
    if fTrial <= 0:
        return epsPla, alphaPla, sigmaTrial
    deltaGamma = fTrial / (E + K)
    sigma = sigmaTrial - deltaGamma*E*np.sign(sigmaTrial)
    epsPla = epsPla + deltaGamma*np.sign(sigmaTrial)
    alphaPla = alphaPla + deltaGamma
    return epsPla, alphaPla, sigma


def computePlasticInnerLoadVectorIsotropic(c, epsYield, hardening, ansatz, quadrature, solution, epsPla, alphaPla):
    grid = ansatz.grid
    n = grid.nElements
    alpha = quadrature.domain.alpha
    F = np.zeros((ansatz.nDof(),))
    nShapesPerElement = ansatz.nShapesPerElement()
    for iElement in range(n):
        lm = ansatz.locationMap(iElement)
        points = quadrature.points[iElement]
        weights = quadrature.weights[iElement]
        Ue = solution[lm]
        Fe = np.zeros(nShapesPerElement)
        for j in range(len(points)):
            shapes = ansatz.evaluate(points[j], 1, iElement)
            B = np.asarray(shapes[1])
            eps = (B @ Ue.transpose())
            newEpsPla, newAlphaPla, sigma = radialReturn(c*c, c*c*epsYield, c*c*hardening, eps, epsPla[iElement][j], alphaPla[iElement][j])
            epsPla[iElement][j] = newEpsPla
            alphaPla[iElement][j] = newAlphaPla
            Fe += B.transpose() * sigma * weights[j] * alpha(points[j])
        F[lm] += Fe
    return F

