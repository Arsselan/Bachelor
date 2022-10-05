import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg


from waves1d import *

p = 3
n = 5
k = p-1

grid = createGrid(0.0, L, n)

ansatz = createSplineAnsatz(grid, p, k)
ansatz = createLagrangeAnsatz(grid, np.linspace(-1, 1, p+1)
ansatz = createLagrangeAnsatz(grid, getGllPoints(p+1))
ansatz = createLegendreAnsatz(grid, p+1)

points = createGaussLegendrePointsAndWeights(p+1)
points = createGaussLobattoPointsAndWeights(p+1)

def domain(x):
    if x>=extra or x<=L-extra:
        return 1
    else
        return 0

cut = 0.4
def domainF(x):
    if x>=cut:
        return 1
    else
        return 0

def domainS(x):
    if x<=cut:
        return 1
    else
        return 0


quadrature = createSpaceTreeQuadrature(grid, domain, points)
quadrature = createMomentFittingQuadrature(grid, domain)

# uncoupled
tripletsM = assembleMassMatrixTriplets(rho, ansatz, quadrature)
tripletsK = assembleStiffnessMatrixTriplets(E*A, ansatz)

M = assembleSparseMatrix(tripletsM)
K = assembleSparseMatrix(trimplesK)

# coupled
tripletsMF = assembleMassMatrixTriplets(rho, ansatz, quadratureMF)
tripletsKF = assembleStiffnessMatrixTriplets(E*A, ansatz, quadratureMF)

tripletsMS = assembleMassMatrixTriplets(rho, ansatz, quadratureS)
tripletsKS = assembleStiffnessMatrixTriplets(rho*c*c, ansatz, quadratureS)

M = assembleSparseMatrix(tripletsMF, tripletsMS)
K = assembleSparseMatrix(tripletsKF, tripletsKS)
C = assembleSparseMatrix(tripletsCF, tripletsCF)

nOmega = K.shape[0]-2
omega = scipy.sparse.linalg.eigs(K, nOmega, M, which='SM', return_eigenvectors=False)



