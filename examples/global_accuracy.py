import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

from context import fem1d


config = fem1d.StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0.0,

    # method
    ansatzType='Spline',
    # ansatzType='Lagrange',
    # ansatzType = 'InterpolatorySpline'
    n=100,
    p=3,
    continuity='p-1',
    mass='RS',

    depth=30,
    spectral=False,
    dual=False,
    stabilize=0,
    smartQuadrature=False,
    source=fem1d.sources.NoSource()
  )



# problem
problemType = 'N'

selective = False

#eigenvalueSearch = 'value'
#eigenvalueSearch = 'value_exclude'
#eigenvalueSearch = 'number'
#eigenvalueSearch = 'vector'
#eigenvalueSearch = 'vector_exclude'
eigenvalueSearch = 'vector_energy'
#eigenvalueSearch = 'vector_energy_exclude'
#eigenvalueSearch = 'individual'

p = config.p
k = eval(config.continuity)
config.n = int(100 / (config.p-k))

L = 1.2 - 2 * config.extra

axLimitLowY = -1
axLimitHighY = 4

# create study
study = fem1d.EigenvalueStudy(config)

if 0:
    if problemType == 'N':
        study.system.findZeroDof(0)
    elif problemType == 'D':
        study.system.findZeroDof(0, [0, study.system.nDof()-1])
    elif problemType == 'DN':
        study.system.findZeroDof(0, [0])
    else:
        print("Error! Choose problem type 'N', 'D', or 'DN'.")

    if len(study.system.zeroDof) > 0:
        print("Warning! There were %d zero dof found: " % len(study.system.zeroDof) + str(study.system.zeroDof))


study.runDense(computeEigenvectors=True, sort=True)

# exact solution
nExact = study.system.nDof()-1 - len(study.system.zeroDof)
nEval = 5001

indices = np.linspace(0, nExact, nExact + 1)

if problemType == 'N':
    wExact = (indices * np.pi) / L
    getExactV = lambda nodes, index : np.cos(index * np.pi / L * (nodes - config.extra))
    getExactGV = lambda nodes, index : - index * np.pi / L * np.sin(index * np.pi / L * (nodes - config.extra))
elif problemType == 'D':
    wExact = ((indices+1) * np.pi) / L
    getExactV = lambda nodes, index : np.sin((index+1) * np.pi / L * (nodes - config.extra))
    getExactGV = lambda nodes, index : (i+1) * np.pi / L * np.cos((i+1) * np.pi / L * (nodes - config.extra))
elif problemType == 'DN':
    wExact = ( (2*indices+1) * np.pi) / (2 * L)
    getExactV = lambda nodes, index : np.sin((2*index+1) * np.pi / (2 * L) * (nodes - config.extra))
    getExactGV = lambda nodes, index: (2*index+1) * np.pi / (2 * L) * np.cos((2*index+1) * np.pi / (2 * L) * (nodes - extra))
else:
    print("Error! Choose problem type 'N', 'D', or 'DN'.")


nodesEval = np.linspace(config.left + config.extra, config.right - config.extra, nEval)
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
iMatrix = study.ansatz.interpolationMatrix(nodesEval)
giMatrix = study.ansatz.interpolationMatrix(nodesEval, 1)


w = np.square(study.w)
w = np.abs(w)
wExact = np.square(wExact)

if np.linalg.norm(np.imag(w)) > 0:
    print("Warning! There were negative eigenvalues: " + str(w))

# sort and compute errors
print("Sorting...", flush=True)
wSorted = np.sort(w)


def squaredNorm(a):
    return np.inner(a, a)


wNearest = 0 * w
vSorted = 0 * study.v
vEval = 0 * iMatrix.toarray()
gvEval = 0 * vEval
wErrors = 0 * w
vErrors = 0 * w
eErrors = 0 * w
vIndices = []
wIndices = []
for i in range(len(w)):
    if eigenvalueSearch == 'value':
        wNum, wIdx = fem1d.findEigenvalue(w, "nearest", i, wExact[i])
        #wIdx = fem1d.find_nearest_index(w, wExact[i])
        vIdx = wIdx
    elif eigenvalueSearch == 'value_exclude':
        wNum, wIdx = fem1d.findEigenvalue(w, "nearest", i, wExact[i], wIndices)
        #wIdx = fem1d.find_nearest_index(w, wExact[i])
        vIdx = wIdx
    elif eigenvalueSearch == 'number':
        wIdx = fem1d.find_nearest_index(w, wSorted[i])
        if wIdx != i:
            print("ERROR!!!")
        vIdx = wIdx
    elif eigenvalueSearch == 'vector_exclude':
        vEval[:, i], vIdx = fem1d.findEigenvector(study.v, 'nearest', i, iMatrix, study.system, vExact[:, i], vIndices)
        wIdx = vIdx
    elif eigenvalueSearch == 'vector':
        vEval[:, i], vIdx = fem1d.findEigenvector(study.v, 'nearest', i, iMatrix, study.system, vExact[:, i])
        wIdx = vIdx
    elif eigenvalueSearch == 'vector_energy':
        vEval[:, i], vIdx = fem1d.findEigenvector(study.v, 'energy', i, giMatrix, study.system, gvExact[:, i])
        wIdx = vIdx
    elif eigenvalueSearch == 'vector_energy_exclude':
        vEval[:, i], vIdx = fem1d.findEigenvector(study.v, 'energy', i, giMatrix, study.system, gvExact[:, i], vIndices)
        wIdx = vIdx
    elif eigenvalueSearch == 'individual':
        vEval[:, i], vIdx = fem1d.findEigenvector(study.v, 'nearest', i, iMatrix, study.system, vExact[:, i], vIndices)
        wIdx = fem1d.find_nearest_index(w, wExact[i])
    else:
        print("Error! Choose eigenvalueSearch 'nearest' or 'number'")

    vIndices.append(vIdx)
    wIndices.append(wIdx)

    wNearest[i] = w[wIdx]

    vSorted[:, i] = study.v[:, vIdx]
    vEval[:, i] = iMatrix * study.system.getFullVector(vSorted[:, i])
    gvEval[:, i] = giMatrix * study.system.getFullVector(vSorted[:, i])

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

ax1.set_title('Eigenvalues')
ax2.set_title('Errors')

ax1.plot(indices, np.sqrt(wExact), '-', label='reference', color='#000000')
ax1.plot(indices, np.sqrt(wSorted), '-', label='numeric (number)')
ax1.plot(indices, np.sqrt(wNearest), '--', label='numeric (taken)')
ax1.plot(indices, wIndices, '-', label='value index')
ax1.plot(indices, vIndices, '--', label='vector index')

ax2.plot(indices[1:], wErrors[1:], '-', label='value error (valE)')
ax2.plot(indices[1:], vErrors[1:], '-', label='l2 norm vector error (vecE)')
ax2.plot(indices[1:], eErrors[1:], '-', label='energy norm vector error')
ax2.plot(indices[1:], np.abs(wErrors[1:]) + vErrors[1:], '--', label='|valE + vecE|')
ax2.plot(indices[1:], wErrors[1:] + vErrors[1:], '-.', label='valE + vecE')

ax1.legend()
ax2.legend()

title = config.ansatzType + ' C' + config.continuity + ' ' + config.mass + ' p=' + str(config.p) + ' n=' + str(config.n) + " d=" + str(config.extra) + " " + eigenvalueSearch
figure.suptitle(title)

ax1.set_xlabel('index')
ax1.set_ylabel('eigenvalue')

fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/global_accuracy/", title.replace(' ', '_'))

fem1d.writeColumnFile(fileBaseName + '_frequencies_p=' + str(config.p) + '.dat', (indices, np.sqrt(wExact), np.sqrt(wSorted), np.sqrt(wNearest)))
fem1d.writeColumnFile(fileBaseName + '_pythagoras_p=' + str(config.p) + '.dat', (indices[1:], wErrors[1:], vErrors[1:], eErrors[1:], wErrors[1:] + vErrors[1:], np.abs(wErrors[1:]) + vErrors[1:]))

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
        index = study.system.nDof() - len(study.system.zeroDof) - 3 + i
        index = 78 + i + nCols*j

        ax[j][i].plot(nodesEval, vEval[:, index], '-', label='numeric')
        ax[j][i].plot(nodesEval, vExact[:, index], '--', label='reference')

        ax[j][i].set_xlabel('x')
        ax[j][i].set_ylabel('eigenvector')
        ax[j][i].set_title('v ' + str(index+1) + ' / ' + str(study.system.nDof() - len(study.system.zeroDof)) + ' e=' + str(vErrors[index]))

        fem1d.writeColumnFile(fileBaseName + '_vector' + str(index+1) + '_p=' + str(p) + '.dat', (nodesEval, vEval[:, index], vExact[:, index]))

plt.savefig(fileBaseName + '_high_vectors.pdf')

plt.show()

