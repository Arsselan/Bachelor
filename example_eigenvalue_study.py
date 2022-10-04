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
    if x>=extra or x<=extra:
        return 1
    else
        return 0

quadrature = createSpaceTreeQuadrature(grid, domain, points)
quadrature = createMomentFittingQuadrature(grid, domain)

tripletsM = assembleMassMatrixTriplets(rho, ansatz, quadrature)
tripletsK = assembleStiffnessMatrixTriplets(E*A, ansatz)

M = assembleSparseMatrix(tripletsM)
K = assembleSparseMatrix(trimplesK)

nOmega = K.shape[0]-2
omega = scipy.sparse.linalg.eigs(K, nOmega, M, which='SM', return_eigenvectors=False)



ax1.plot(t, np.zeros(t.size), '-o')
ax1.plot(t, np.ones(t.size), '-o')

ax1.plot(points, np.ones(points.size), 'x')


soundSpeed = 1#np.sqrt(E/rho);
freq = np.pi*np.linspace(0,n,n+1)*soundSpeed / length;

ax3.plot(w)
ax3.plot(freq,'--')

                          
plt.show()

