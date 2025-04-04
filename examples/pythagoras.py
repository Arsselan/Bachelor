import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

from context import fem1d

# problem
problemType = 'N'
left = 0
right = 1.2
extra = 0.0

# method
p = 3
ansatzType = 'Spline'
#ansatzType = 'InterpolatorySpline'
#ansatzType = 'Lagrange'

continuity = 'p-1'
spectral = False
dual = False

selective = False
mass = 'CON'
#mass = 'RS'
#mass = 'HRZ'

#eigenvalueSearch = 'nearest'
eigenvalueSearch = 'number'
#eigenvalueSearch = 'vector'
#eigenvalueSearch = 'individual'

depth = 40

if ansatzType == 'Lagrange':
    continuity = '0'

k = eval(continuity)
n = int(100 / (p-k))

#extra = 1.2 / n * 4.99
L = 1.2 - 2 * extra

axLimitLowY = -1
axLimitHighY = 4


# create grid and domain
grid = fem1d.UniformGrid(left, right, n)


def alpha(x):
    if left + extra <= x <= right - extra:
        return 1.0
    return 0


domain = fem1d.Domain(alpha)

# create ansatz and quadrature
ansatz = fem1d.createAnsatz(ansatzType, continuity, p, grid)

gaussPointsM = fem1d.gll.computeGllPoints(p + 1)
quadratureM = fem1d.SpaceTreeQuadrature(grid, gaussPointsM, domain, depth)

gaussPointsK = np.polynomial.legendre.leggauss(p + 1)
quadratureK = fem1d.SpaceTreeQuadrature(grid, gaussPointsK, domain, depth)

# create system
if spectral:
    system = fem1d.TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK)
else:
    if dual:
        system = fem1d.TripletSystem.fromOneQuadratureWithDualBasis(ansatz, quadratureK, selectiveLumping=selective)
    else:
        system = fem1d.TripletSystem.fromOneQuadrature(ansatz, quadratureK, selectiveLumping=selective)

if problemType == 'N':
    system.findZeroDof(0)
elif problemType == 'D':
    system.findZeroDof(0, [0, system.nDof()-1])
elif problemType == 'DN':
    system.findZeroDof(0, [0])
else:
    print("Error! Choose problem type 'N', 'D', or 'DN'.")

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
nEval = 5001

indices = np.linspace(0, nExact, nExact + 1)

if problemType == 'N':
    wExact = (indices * np.pi) / L
    getExactV = lambda nodes, index : np.cos(index * np.pi / L * (nodes - extra))
    getExactGV = lambda nodes, index : - index * np.pi / L * np.sin(index * np.pi / L * (nodes - extra))
elif problemType == 'D':
    wExact = ((indices+1) * np.pi) / L
    getExactV = lambda nodes, index : np.sin((index+1) * np.pi / L * (nodes - extra))
    getExactGV = lambda nodes, index : (i+1) * np.pi / L * np.cos((i+1) * np.pi / L * (nodes - extra))
elif problemType == 'DN':
    wExact = ( (2*indices+1) * np.pi) / (2 * L)
    getExactV = lambda nodes, index : np.sin((2*index+1) * np.pi / (2 * L) * (nodes - extra))
    getExactGV = lambda nodes, index: (2*index+1) * np.pi / (2 * L) * np.cos((2*index+1) * np.pi / (2 * L) * (nodes - extra))
else:
    print("Error! Choose problem type 'N', 'D', o r'DN'.")


nodesEval = np.linspace(left + extra, right - extra, nEval)
vExact = np.ndarray((nEval, nExact + 1))
gvExact = np.ndarray((nEval, nExact + 1))
for i in range(nExact + 1):
    vExact[:, i] = getExactV(nodesEval, i)
    vNorm = np.linalg.norm(vExact[:, i]) / np.sqrt(nEval)
    if vNorm > 0:
        vExact[:, i] = vExact[:, i] / vNorm
    gvExact[:, i] = getExactGV(nodesEval, i)
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


wNearest = 0 * w
vSorted = 0 * v
vEval = 0 * iMatrix.toarray()
gvEval = 0 * vEval
wErrors = 0 * w
vErrors = 0 * w
eErrors = 0 * w
for i in range(len(w)):
    if eigenvalueSearch == 'nearest':
        idx = fem1d.find_nearest_index(w, wExact[i])
        widx = idx
    elif eigenvalueSearch == 'number':
        idx = fem1d.find_nearest_index(w, wSorted[i])
        widx = idx
    elif eigenvalueSearch == 'vector':
        vEval[:, i], idx = fem1d.findEigenvector(v, 'nearest', i, iMatrix, system, vExact[:, i])
        widx = idx
    elif eigenvalueSearch == 'individual':
        vEval[:, i], idx = fem1d.findEigenvector(v, 'nearest', i, iMatrix, system, vExact[:, i])
        widx = fem1d.find_nearest_index(w, wExact[i])
    else:
        print("Error! Choose eigenvalueSearch 'nearest' or 'number'")

    wNearest[i] = w[widx]

    vSorted[:, i] = v[:, idx]
    vEval[:, i] = iMatrix * system.getFullVector(vSorted[:, i])
    gvEval[:, i] = giMatrix * system.getFullVector(vSorted[:, i])

    # normalize
    vNorm = np.linalg.norm(vEval[:, i]) / np.sqrt(nEval)
    if vNorm > 0:
        vEval[:, i] = vEval[:, i] / vNorm
        gvEval[:, i] = gvEval[:, i] / vNorm

    # correct sign
    if vEval[0, i] < 0 and problemType == 'N':
        vEval[:, i] = -vEval[:, i]
        gvEval[:, i] = - gvEval[:, i]

    if gvEval[0, i] < 0 and (problemType == 'D' or problemType == 'DN'):
        vEval[:, i] = -vEval[:, i]
        gvEval[:, i] = - gvEval[:, i]

    # compute errors
    if wExact[i] > 0:
        wErrors[i] = (wNearest[i] - wExact[i]) / wExact[i]

    vNorm = squaredNorm(vExact[:, i])
    if vNorm > 0:
        vErrors[i] = squaredNorm(vExact[:, i] - vEval[:, i]) / vNorm
        error = squaredNorm(vExact[:, i] + vEval[:, i]) / vNorm
        if error < vErrors[i]:
            vErrors[i] = error
            vEval[:, i] *= -1

    energy = squaredNorm(gvExact[:, i])
    if energy > 0:
        eErrors[i] = squaredNorm(gvEval[:, i] - gvExact[:, i]) / energy

print("Potting...", flush=True)

# plot
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams["figure.figsize"] = (8, 4)

figure, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_ylim(0, 600)

ax2.set_ylim(axLimitLowY, axLimitHighY)

ax1.set_title('Eigenfrequencies')
ax2.set_title('Eigenvector errors')

ax1.plot(indices, np.sqrt(wExact), '-', label='reference', color='#000000')
ax1.plot(indices, np.sqrt(wSorted), '-', label='numeric (number)')
ax1.plot(indices, np.sqrt(wNearest), '--', label='numeric (taken)')

ax2.plot(indices[1:], wErrors[1:], '-', label='value error')
ax2.plot(indices[1:], vErrors[1:], '-', label='vector error')
ax2.plot(indices[1:], eErrors[1:], '-', label='energy error')
ax2.plot(indices[1:], np.abs(wErrors[1:]) + vErrors[1:], '--', label='sum abs error')
ax2.plot(indices[1:], wErrors[1:] + vErrors[1:], '-.', label='sum error')

ax1.legend()
ax2.legend()

title = ansatzType + ' C' + continuity + ' ' + mass + ' p=' + str(p) + ' n=' + str(n) + " d=" + str(extra) + " " + eigenvalueSearch
figure.suptitle(title)

ax1.set_xlabel('index')
ax1.set_ylabel('eigenvalue')

fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/example_pythagoras/", title.replace(' ', '_'))

fem1d.writeColumnFile(fileBaseName + '_frequencies_p=' + str(p) + '.dat', (indices, np.sqrt(wExact), np.sqrt(wSorted), np.sqrt(wNearest)))
fem1d.writeColumnFile(fileBaseName + '_pythagoras_p=' + str(p) + '.dat', (indices[1:], wErrors[1:], vErrors[1:], eErrors[1:], wErrors[1:] + vErrors[1:], np.abs(wErrors[1:]) + vErrors[1:]))

plt.savefig(fileBaseName + '.pdf')
plt.show()


#plt.rcParams["figure.figsize"] = (12, 3)

nRows = 3
nCols = 4
figure, ax = plt.subplots(nRows, nCols)
plt.rcParams['axes.titleweight'] = 'bold'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
figure.tight_layout(pad=1)

for j in range(nRows):

    for i in range(nCols):
        index = system.nDof() - len(system.zeroDof) - 3 + i
        index = 78 + i + nCols*j

        ax[j][i].plot(nodesEval, vEval[:, index], '-', label='numeric')
        ax[j][i].plot(nodesEval, vExact[:, index], '--', label='reference')

        ax[j][i].set_xlabel('x')
        ax[j][i].set_ylabel('eigenvector')
        ax[j][i].set_title('v ' + str(index+1) + ' / ' + str(system.nDof() - len(system.zeroDof)) + ' e=' + str(vErrors[index]))

        fem1d.writeColumnFile(fileBaseName + '_vector' + str(index+1) + '_p=' + str(p) + '.dat', (nodesEval, vEval[:, index], vExact[:, index]))

plt.savefig(fileBaseName + '_high_vectors.pdf')

plt.show()

