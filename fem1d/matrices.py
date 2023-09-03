import numpy as np
import scipy.sparse
import scipy.sparse.linalg


class WaveEquationStandardMatrices:
    def __init__(self, density, elasticity, bodyLoad):
        self.density = density
        self.elasticity = elasticity
        self.bodyLoad = bodyLoad

        self.Me = np.zeros(0)
        self.Ke = np.zeros(0)
        self.Fe = np.zeros(0)

    def initializeSystem(self, system, nDof, nVals, nElements):
        system.matrixValues['M'] = np.zeros(nVals * nElements)
        system.matrixValues['K'] = np.zeros(nVals * nElements)
        system.vectors['F'] = np.zeros((nDof,))

    def initializeElementMatrices(self, system, iElement, nShapes):
        self.Me = np.zeros((nShapes, nShapes))
        self.Ke = np.zeros((nShapes, nShapes))
        self.Fe = np.zeros((nShapes,))

    def addSystemIntegrands(self, system, point, weight, alpha, N, B):
        self.Me += np.outer(N, N) * weight * alpha
        self.Ke += np.outer(B, B) * weight * alpha
        self.Fe += N * self.bodyLoad(point) * weight * alpha

    def scatterElementMatrices(self, system, iElement, mass, lm, eSlice):
        system.matrixValues['K'][eSlice] = self.Ke.ravel()
        system.matrixValues['M'][eSlice] = self.Me.ravel()
        system.vectors['F'][lm] += self.Fe
        return

    def finalizeSystem(self, system):
        pass


class WaveEquationStiffnessMatrixAndLoadVector:
    def __init__(self, elasticity, bodyLoad):
        self.elasticity = elasticity
        self.bodyLoad = bodyLoad

        self.Ke = np.zeros(0)
        self.Fe = np.zeros(0)

    def initializeSystem(self, system, nDof, nVals, nElements):
        system.matrixValues['K'] = np.zeros(nVals * nElements)
        system.vectors['F'] = np.zeros((nDof,))

    def initializeElementMatrices(self, system, iElement, nShapes):
        self.Ke = np.zeros((nShapes, nShapes))
        self.Fe = np.zeros((nShapes,))

    def addSystemIntegrands(self, system, point, weight, alpha, N, B):
        self.Ke += np.outer(B, B) * weight * alpha
        self.Fe += N * self.bodyLoad(point) * weight * alpha

    def scatterElementMatrices(self, system, iElement, mass, lm, eSlice):
        system.matrixValues['K'][eSlice] = self.Ke.ravel()
        system.vectors['F'][lm] += self.Fe
        return

    def finalizeSystem(self, system):
        pass


class WaveEquationMassMatrix:
    def __init__(self, density):
        self.density = density

        self.Me = np.zeros(0)

    def initializeSystem(self, system, nDof, nVals, nElements):
        system.matrixValues['M'] = np.zeros(nVals * nElements)

    def initializeElementMatrices(self, system, iElement, nShapes):
        self.Me = np.zeros((nShapes, nShapes))

    def addSystemIntegrands(self, system, point, weight, alpha, N, B):
        self.Me += np.outer(N, N) * weight * alpha

    def scatterElementMatrices(self, system, iElement, mass, lm, eSlice):
        system.matrixValues['M'][eSlice] = self.Me.ravel()
        return

    def finalizeSystem(self, system):
        pass


def computeRowSummedMatrix(Me):
    diagMe = np.zeros(Me.shape)
    for iEntry in range(Me.shape[0]):
        diagMe[iEntry, iEntry] = sum(Me[iEntry, :])
    return diagMe


def computeHrzLumpedMatrix(Me, mass):
    diagMe = np.zeros(Me.shape)
    sumMe = 0
    for iEntry in range(Me.shape[0]):
        diagMe[iEntry, iEntry] = Me[iEntry, iEntry]
        sumMe += Me[iEntry, iEntry]
    if sumMe > 0:
        c = mass * 1.0 / sumMe
    else:
        c = 0
    diagMe = diagMe * c
    return diagMe


class WaveEquationLumpedMatrices:
    def __init__(self, stdMatrices):
        self.stdMatrices = stdMatrices
        self.Me = stdMatrices.Me

    def initializeSystem(self, system, nDof, nVals, nElements):
        self.stdMatrices.initializeSystem(system, nDof, nVals, nElements)
        system.matrixValues['MRS'] = np.zeros(nVals * nElements)
        system.matrixValues['MHRZ'] = np.zeros(nVals * nElements)

    def initializeElementMatrices(self, system, iElement, nShapes):
        self.stdMatrices.initializeElementMatrices(system, iElement, nShapes)
        self.Me = self.stdMatrices.Me

    def addSystemIntegrands(self, system, point, weight, alpha, N, B):
        self.stdMatrices.addSystemIntegrands(system, point, weight, alpha, N, B)

    def scatterElementMatrices(self, system, iElement, mass, lm, eSlice):
        self.stdMatrices.scatterElementMatrices(system, iElement, mass, lm, eSlice)
        rsMe = computeRowSummedMatrix(self.stdMatrices.Me)
        system.matrixValues['MRS'][eSlice] = rsMe.ravel()
        hrzMe = computeHrzLumpedMatrix(self.stdMatrices.Me, mass)
        system.matrixValues['MHRZ'][eSlice] = hrzMe.ravel()

    def finalizeSystem(self, system):
        self.stdMatrices.finalizeSystem(system)


def getEigenvalueDecomposition(mat, smallThreshold):
    w, v = scipy.linalg.eig(mat, right=True)
    n = mat.shape[0]
    smallIndices = []
    wMax = np.max(w)
    for i in range(n):
        if w[i] < smallThreshold * wMax:
            smallIndices.append(i)
    w = np.diag(np.real(w))
    error = np.linalg.norm(mat - v.dot(w).dot(v.transpose()))
    if error > 1e-10:
        print("Error! Eigenvalue decomposition failed, error is %e." % error)
    return w, v, smallIndices


def findRigidBodyMode(v, rbmMaxError=1e-10):
    rbmError = 1e6
    n = v.shape[0]
    rbm = np.ones(n)
    rbm = rbm / np.linalg.norm(rbm)
    rbmIndex = -1
    for i in range(n):
        rbmDeviation = min(np.linalg.norm(v[:, i] - rbm), np.linalg.norm(v[:, i] + rbm))
        if rbmDeviation < rbmError:
            rbmIndex = i
            rbmError = rbmDeviation
    if rbmError > rbmMaxError:
        print("Error! Could not identify rigid body mode.")
    return rbmIndex


def createStabilizationMatrix(v, w, indices):
    n = v.shape[0]
    for i in range(n):
        if i not in indices:
            v[:, i] *= 0
    mat = v.dot(v.transpose())
    return mat


class WaveEquationStabilizedMatrices:
    def __init__(self, stdMatrices, epsM):
        self.stdMatrices = stdMatrices
        self.epsM = epsM
        self.Me = stdMatrices.Me
        self.stabilizedModes = 0

    def initializeSystem(self, system, nDof, nVals, nElements):
        self.stdMatrices.initializeSystem(system, nDof, nVals, nElements)
        system.matrixValues['modM'] = np.zeros(nVals * nElements)
        system.matrixValues['modMHRZ'] = np.zeros(nVals * nElements)
        self.stabilizedModes = 0

    def addSystemIntegrands(self, system, point, weight, alpha, N, B):
        self.stdMatrices.addSystemIntegrands(system, point, weight, alpha, N, B)

    def initializeElementMatrices(self, system, iElement, nShapes):
        self.stdMatrices.initializeElementMatrices(system, iElement, nShapes)
        self.Me = self.stdMatrices.Me

    def scatterElementMatrices(self, system, iElement, mass, lm, eSlice):
        self.stdMatrices.scatterElementMatrices(system, iElement, mass, lm, eSlice)
        if mass > 0:
            eigenValsM, eigenVecsM, smallIndicesM = getEigenvalueDecomposition(self.Me, 1e-3)
            if len(smallIndicesM) > 0:
                self.stabilizedModes += len(smallIndicesM)
                #print("Me_%d\n" % iElement, self.Me)
                #print("vals\n", eigenValsM.diagonal())
                #print("vecs\n", eigenVecsM)
                #print("small indices: ", smallIndicesM)
                #print("Mstab_%d\n" % iElement, createStabilizationMatrix(eigenVecsM, eigenValsM, smallIndicesM))
                stabMe = self.epsM * createStabilizationMatrix(eigenVecsM, eigenValsM, smallIndicesM)
            else:
                stabMe = np.zeros(self.Me.shape)
            modMe = self.Me + stabMe
            system.matrixValues['modM'][eSlice] = modMe.ravel()
            # system.matrixValues['modMHRZ'][eSlice] = computeHrzLumpedMatrix(modMe, mass).ravel()
            system.matrixValues['modMHRZ'][eSlice] = computeHrzLumpedMatrix(self.Me, mass).ravel()
            system.matrixValues['modMHRZ'][eSlice] += computeRowSummedMatrix(stabMe).ravel()

    def finalizeSystem(self, system):
        self.stdMatrices.finalizeSystem(system)
        print("Stabilized %d modes" % self.stabilizedModes)


def computeSystemMatrices(system, ansatz, quadrature, matrices):
    grid = ansatz.grid
    n = grid.nElements
    alpha = quadrature.domain.alpha

    nShapesPerElement = ansatz.nShapesPerElement()
    nVal = nShapesPerElement * nShapesPerElement

    system.row = np.zeros(nVal * n, dtype=np.uint)
    system.col = np.zeros(nVal * n, dtype=np.uint)

    matrices.initializeSystem(system, ansatz.nDof(), nVal, n)

    system.minNonZeroMass = float('inf')
    system.minNonZeroMassElementIndex = -1
    for iElement in range(n):

        lm = ansatz.locationMap(iElement)
        points = quadrature.points[iElement]
        weights = quadrature.weights[iElement]
        mass = 0

        matrices.initializeElementMatrices(system, iElement, nShapesPerElement)

        for j in range(len(points)):
            shapes = ansatz.evaluate(points[j], 1, iElement)
            N = np.asarray(shapes[0])
            B = np.asarray(shapes[1])
            matrices.addSystemIntegrands(system, points[j], weights[j], alpha(points[j]), N, B)
            mass += weights[j] * alpha(points[j])

        if system.minNonZeroMass > mass > 0:
            system.minNonZeroMass = mass
            system.minNonZeroMassElementIndex = iElement

        eSlice = slice(nVal * iElement, nVal * (iElement + 1))
        system.row[eSlice] = np.broadcast_to(lm, (nShapesPerElement, nShapesPerElement)).T.ravel()
        system.col[eSlice] = np.broadcast_to(lm, (nShapesPerElement, nShapesPerElement)).ravel()

        matrices.scatterElementMatrices(system, iElement, mass, lm,  eSlice)

    matrices.finalizeSystem(system)
