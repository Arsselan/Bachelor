import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline

from waves1d import *

# problem
left = 0
right = 1.2
extra = 0.219 * 0

# method
p = 2
ansatzType = 'Spline'
#ansatzType = 'Lagrange'

continuity = 'p-1'
spectral = False
mass = 'RS'
#mass ='CON'

depth = 40

if ansatzType == 'Lagrange':
    continuity = '0'

k = eval(continuity)
n = int(100 / (p-k))


axLimitLowY = -1
axLimitHighY = 4

#extra = 1.2 / n * 0.99
L = 1.2 - 2 * extra


# create grid and domain
grid = UniformGrid(left, right, n)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 0


domain = Domain(alpha)

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

system.findZeroDof(0)
#system.findZeroDof(-1e60, [0, system.nDof()-1])
if len(system.zeroDof) > 0:
    print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

# solve sparse
M, K, MHRZ, MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)

if mass == 'CON':
    # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, M, which='SM', return_eigenvectors=False)
    w, v = scipy.linalg.eig(K.toarray(), M.toarray(), right=True)
elif mass == 'HRZ':
    # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MHRZ, which='SM', return_eigenvectors=False)
    w, v = scipy.linalg.eig(K.toarray(), MHRZ.toarray(), right=True)
elif mass == 'RS':
    # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MRS, which='SM', return_eigenvectors=False)
    w, v = scipy.linalg.eig(K.toarray(), MRS.toarray(), right=True)
else:
    print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

if np.linalg.norm(np.imag(w)) > 0:
    print("Warning! There were imaginary eigenvalues: " + str(w))


# exact solution
nExact = system.nDof()-1 - len(system.zeroDof)
nEval = 5000

indices = np.linspace(0, nExact, nExact + 1)
wExact = (indices * np.pi) / L

nodesEval = np.linspace(left + extra, right - extra, nEval)
vExact = np.ndarray((nEval, nExact + 1))
gvExact = np.ndarray((nEval, nExact + 1))
for i in range(nExact + 1):
    vExact[:, i] = np.cos(i * np.pi / L * (nodesEval - extra))
    vNorm = np.linalg.norm(vExact[:, i])
    if vNorm > 0:
        vExact[:, i] = vExact[:, i] / vNorm
    gvExact[:, i] = - i * np.pi / L * np.sin(i * np.pi / L * (nodesEval - extra))
    gvNorm = np.linalg.norm(gvExact[:, i])
    if vNorm > 0:
        gvExact[:, i] = gvExact[:, i] / vNorm

# evaluation
iMatrix = ansatz.interpolationMatrix(nodesEval)
giMatrix = ansatz.interpolationMatrix(nodesEval, 1)

w = np.real(w)
w = np.abs(w)
#w = np.sqrt(w + 0j)
wExact = np.square(wExact)


if np.linalg.norm(np.imag(w)) > 0:
    print("Warning! There were negative eigenvalues: " + str(w))

# sort and compute errors
print("Sorting...", flush=True)
wSorted = np.sort(w)


def squaredNorm(a):
    return np.inner(a, a)


vSorted = 0 * v
vEval = 0 * iMatrix.toarray()
gvEval = 0 * vEval
wErrors = 0 * w
vErrors = 0 * w
eErrors = 0 * w
for i in range(len(w)):
    #idx = find_nearest_index(w, wExact[i])
    idx = find_nearest_index(w, wSorted[i])
    vSorted[:, i] = v[:, idx]
    vEval[:, i] = iMatrix * system.getFullVector(vSorted[:, i])
    gvEval[:, i] = giMatrix * system.getFullVector(vSorted[:, i])

    vNorm = np.linalg.norm(vEval[:, i])
    if vNorm > 0:
        vEval[:, i] = vEval[:, i] / vNorm

    gvNorm = np.linalg.norm(gvEval[:, i])
    if vNorm > 0:
        gvEval[:, i] = gvEval[:, i] / vNorm

    if vEval[0, i] < 0:
        vEval[:, i] = -vEval[:, i]
        gvEval[:, i] = - gvEval[:, i]

    if wExact[i] > 0:
        wErrors[i] = (wSorted[i] - wExact[i]) / wExact[i]

    energy = squaredNorm(gvExact[:, i])
    if energy > 0:
        eErrors[i] = squaredNorm(gvEval[:, i] - gvExact[:, i]) / energy

    vNorm = squaredNorm(vExact[:, i])
    if vNorm > 0:
        vErrors[i] = squaredNorm(vExact[:, i] - vEval[:, i]) / vNorm

print("Potting...", flush=True)

# plot
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams["figure.figsize"] = (8, 4)

figure, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_ylim(0, 600)

ax2.set_ylim(axLimitLowY, axLimitHighY)

# ax2.plot(indices[1:], wExact[1:] / wExact[1:], '-', label='reference')
#ax2.plot(nodesEval, vExact[:, 1], '--o', label='reference')

ax1.set_title('Eigenfrequencies')
ax2.set_title('Eigenvector errors')

ax1.plot(indices, np.sqrt(wExact), '-', label='reference', color='#000000')
ax1.plot(indices, np.sqrt(wSorted), '-', label='numeric')

ax2.plot(indices[1:], wErrors[1:], '-', label='value error')
ax2.plot(indices[1:], vErrors[1:], '-', label='vector error')
ax2.plot(indices[1:], eErrors[1:], '-', label='energy error')
ax2.plot(indices[1:], np.abs(wErrors[1:]) + vErrors[1:], '--', label='sum abs error')
ax2.plot(indices[1:], wErrors[1:] + vErrors[1:], '-.', label='sum error')

#ax2.plot(indices[1:], np.abs(wErrors[1:]) + np.abs(vErrors[1:]) / eErrors[1:], '--*', label='sum error')

#ax2.plot(nodesEval, vEval[:, 1], '--o', label='numeric')

ax1.legend()
ax2.legend()

title = ansatzType + ' C' + continuity + ' ' + mass + ' p=' + str(p) + ' n=' + str(n) + " d=" + str(extra)
figure.suptitle(title)

ax1.set_xlabel('index')
ax1.set_ylabel('eigenvalue')

fileBaseName = getFileBaseNameAndCreateDir("results/example_pythagoras/", title.replace(' ', '_'))
#np.savetxt(fileBaseName + '.dat', res)

plt.savefig(fileBaseName + '.pdf')
plt.show()


plt.rcParams["figure.figsize"] = (12, 3)

figure, ax = plt.subplots(1, 3)
plt.rcParams['axes.titleweight'] = 'bold'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
figure.tight_layout(pad=3)

for i in range(3):
    ax[i].plot(nodesEval, vEval[:, -3+i], '-', label='p=' + str(p))
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('eigenvector')
    ax[i].set_title('eigenvector ' + str(system.nDof()-3+1+i) + ' / ' + str(system.nDof()))

plt.savefig(fileBaseName + '_high_vectors.pdf')

plt.show()