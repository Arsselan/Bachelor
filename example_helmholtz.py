import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
from waves1d import *

# problem
left = 0
right = 3.4
extra = 0.0

rho = 1.3
f = 200
w = 2*np.pi * f
c = 340
k = w / c

# method
depth = 40
p = 2
n = 40

# analysis
nw = n
indices = np.linspace(0, nw, nw + 1)
wExact = (indices * np.pi) / (1.2 - 2 * extra)

ansatzType = 'Lagrange'
continuity = 0

print("Running study...")
# create grid and domain
grid = UniformGrid(left, right, n)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 0


domain = Domain(alpha)

# create ansatz and quadrature
ansatz = createAnsatz(ansatzType, continuity, p, grid)

gaussPoints = np.polynomial.legendre.leggauss(p + 1)
quadrature = SpaceTreeQuadrature(grid, gaussPoints, domain, depth)
system = TripletSystem.fromOneQuadrature(ansatz, quadrature)

system.findZeroDof(0)
if len(system.zeroDof) > 0:
    print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

# solve sparse
M, K = system.createSparseMatrices()
factorized = scipy.sparse.linalg.splu(-k**2 * M + K)

rhs = np.zeros(M.shape[0])
rhs[0] = 1e-5
p = factorized.solve(rhs)

# free ends
Yleft = 0
Yright = 1

D = np.zeros(M.shape)
D[0, 0] = Yleft
D[-1, -1] = Yright

M = M.toarray()
K = K.toarray()

zero = np.zeros(M.shape)

A = np.block([[M, zero], [zero, -K]])
B = np.block([[zero, M], [M, D]])

values, vectors = scipy.linalg.eig(A, B, right=True)
values = values * c / 2 / np.pi


figure, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.linspace(left, right, p.shape[0]), p, '-', label='helmholtz solution')

ax2.plot(np.abs(np.real(values)), np.imag(values), '*', label='eigenvalues')

plt.show()
