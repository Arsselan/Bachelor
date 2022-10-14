import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from waves1d import *

# problem
left = 0
right = 1.2
extra = 0.0
eigenVector = 49

# analysis
n = 25
p = 4

# method
ansatzType = 'Lagrange'
spectral = False

#ansatzType = 'Spline'
continuity = 'p-1'

mass = 'CON'
#mass = 'HRZ'
#mass = 'RS'

depth = 40

eigenvalueSearch = 'nearest'
# eigenvalueSearch = 'number'

if ansatzType == 'Lagrange':
    continuity = '0'

L = 1.2 - 2 * extra
wExact = eigenVector * np.pi / L

nodesEval = np.linspace(left + extra, right - extra, 1000)
vExact = np.cos(eigenVector * np.pi / L * (nodesEval - extra))


# plot(nodesEval, vExact)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 1e-12


domain = Domain(alpha)

# create grid and domain
grid = UniformGrid(left, right, n)

# create ansatz and quadrature
ansatz = createAnsatz(ansatzType, continuity, p, grid)

gaussPointsM = GLL(p + 1)
quadratureM = SpaceTreeQuadrature(grid, gaussPointsM, domain, depth)

gaussPointsK = np.polynomial.legendre.leggauss(p + 1)
quadratureK = SpaceTreeQuadrature(grid, gaussPointsK, domain, depth)

# create system
if spectral:
    system = TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK)
else:
    system = TripletSystem.fromOneQuadrature(ansatz, quadratureK)

system.findZeroDof()
if len(system.zeroDof) > 0:
    print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

# solve sparse
M, K, MHRZ, MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)

nEigen = min(K.shape[0] - 2, 2 * eigenVector)
if mass == 'CON':
    # w, v = scipy.sparse.linalg.eigs(K, nEigen, M, which='SM', return_eigenvectors=True)
    w, v = scipy.linalg.eig(K.toarray(), M.toarray(), right=True)
elif mass == 'HRZ':
    # w, v = scipy.sparse.linalg.eigs(K, nEigen, MHRZ, which='SM', return_eigenvectors=True)
    w, v = scipy.linalg.eig(K.toarray(), MHRZ.toarray(), right=True)
elif mass == 'RS':
    # w, v = scipy.sparse.linalg.eigs(K, nEigen, MRS, which='SM', return_eigenvectors=True)
    w, v = scipy.linalg.eig(K.toarray(), MRS.toarray(), right=True)
else:
    print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

# compute frequencies
w = np.real(w)
w = np.abs(w)
w = np.sqrt(w + 0j)
# w = np.sort(w)

dof = system.nDof()

idx = eigenVector
if eigenvalueSearch == 'nearest':
    wNum = find_nearest(w, wExact)
    idx = find_nearest_index(w, wExact)
elif eigenvalueSearch == 'number':
    wNum = w[eigenVector]
else:
    print("Error! Choose eigenvaluesSearch 'nearest' or 'number'")

if np.imag(wNum) > 0:
    print("Warning! Chosen eigenvalue has imaginary part.")

iMatrix = ansatz.interpolationMatrix(nodesEval)
eVector = iMatrix * v[:, idx]
eVector = eVector / eVector[0]

idx = find_nearest_index(w, max(w))
eVectorHighest = iMatrix * v[:, idx]
eVectorHighest = eVectorHighest / eVectorHighest[0]


#plt.rcParams["figure.figsize"] = (13, 6)

figure, ax = plt.subplots(1, 2)
plt.rcParams['axes.titleweight'] = 'bold'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#ax[0].set_xlim(20, 450)
#ax[0].set_ylim(1e-12, 0.1)

ax[0].plot(nodesEval, eVector, '-', label='p=' + str(p), color=colors[p - 1])
ax[1].plot(nodesEval, eVectorHighest, '-', label='p=' + str(p), color=colors[p - 1])

ax[0].legend()
ax[1].legend()

title = ansatzType
title += ' ' + continuity
title += ' ' + mass
title += ' d=' + str(extra)
title += ' ' + eigenvalueSearch
figure.suptitle(title)

ax[0].set_title('Eigenvector ' + str(eigenVector) + '/' + str(system.nDof()))
ax[1].set_title('Eigenvector ' + str(system.nDof()) + '/' + str(system.nDof()))

ax[0].set_xlabel('degrees of freedom')
ax[1].set_xlabel('degrees of freedom')
ax[0].set_ylabel('relative error in sixth eigenvalue ')
ax[1].set_ylabel('relative error in sixth eigenvector ')

#plt.savefig('results/eigen_' + title.replace(' ', '_') + '2.pdf')
plt.show()
