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
n = 25

# analysis
nw = n
indices = np.linspace(0, nw, nw + 1)
wExact = (indices * np.pi) / (1.2 - 2 * extra)

ansatzType = 'Lagrange'
continuity = '0'

print("Running study...")
# create grid and domain
grid = UniformGrid(left, right, n)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 0


domain = Domain(alpha)

# create ansatz and quadrature
#ansatz = createAnsatz(ansatzType, continuity, p, grid)
#gllPoints = gll.computeGllPoints(p + 1)
ansatz = LagrangeAnsatz(grid, np.linspace(-1, 1, p+1))

gaussPoints = np.polynomial.legendre.leggauss(p + 1)
quadrature = SpaceTreeQuadrature(grid, gaussPoints, domain, depth)
system = TripletSystem.fromOneQuadrature(ansatz, quadrature)

system.findZeroDof(0)
if len(system.zeroDof) > 0:
    print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

# solve sparse
print("Integration...")
M, K = system.createSparseMatrices()

print("Factorization...")
factorized = scipy.sparse.linalg.splu(-k**2 * M + K)

print("Solving...")
rhs = np.zeros(M.shape[0])
rhs[0] = 1e-5
solution = factorized.solve(rhs)

print("Transformation...")
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

print("Eigenvalue computation...")
values, vectors = scipy.linalg.eig(A, B, right=True)
values = values * c / 2 / np.pi

idx = np.real(values).argsort()[::1]
values = values[idx]
vectors = vectors[:, idx]

nDof = int(vectors[0].shape[0] / 2)
nodes = np.linspace(left, right, nDof)
vectors = vectors[nDof:, :]

for idx in range(nDof):
    vectors[:, idx] = vectors[:, idx] / np.real(vectors[-1, idx])

title = ansatzType + " n=%d p=%d" % (n, p)
fileBaseName = getFileBaseNameAndCreateDir("results/example_helmholtz/", title.replace(' ', '_'))

np.savetxt(fileBaseName + "_values.dat", values)
np.savetxt(fileBaseName + "_vectors.dat", vectors)


np.savetxt(fileBaseName + "_nodes.dat", nodes)
np.savetxt(fileBaseName + "_vectors_abs.dat", np.abs(vectors))
np.savetxt(fileBaseName + "_vectors_real.dat", np.real(vectors))
np.savetxt(fileBaseName + "_vectors_imag.dat", np.imag(vectors))

print("Post processing...")


def plotEigenvalues():
    figure, ax = plt.subplots()
    ax.plot(np.abs(np.real(values)), np.abs(np.imag(values)), '.', label='eigenvalues')
    ax.set_xlabel("|real|")
    ax.set_ylabel("imag")
    #ax.set_title("eigenvalues " + title)
    plt.savefig(fileBaseName + "_Yl=%d_Yr=%d" % (Yleft, Yright) + '_values.pdf')
    plt.show()


def plotEigenvectors(sqrtn):
    plt.rcParams["figure.figsize"] = (13, 8)
    figure, ax = plt.subplots(sqrtn, sqrtn)
    figure.tight_layout(pad=3)
    for i in range(sqrtn):
        for j in range(sqrtn):
            idx = i * sqrtn + j
            ax[i, j].plot(nodes, np.absolute(vectors[:, idx]), '-', label='eigenvector abs')
            ax[i, j].plot(nodes, np.real(vectors[:, idx]), '--', label='eigenvector real')
            ax[i, j].plot(nodes, np.imag(vectors[:, idx]), '-.', label='eigenvector imag')
            ax[i, j].set_xlabel("x")
            ax[i, j].set_ylabel("vector")
            ax[i, j].set_title("%d: %3.2f%+3.2fi" % (idx, np.real(values[idx]), np.imag(values[idx])))
    plt.savefig(fileBaseName + "_Yl=%d_Yr=%d" % (Yleft, Yright) + '_vectors.pdf')
    plt.show()
