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
extra = 0.119*0

# method
depth = 40
p = 5
n = 120*p

# analysis
nw = n
indices = np.linspace(0, nw, nw + 1)
wExact = (indices * np.pi) / (1.2 - 2 * extra)


def runStudy(n, p, extra, spectral, mass):
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

    gaussPointsM = GLL(p + 1)
    quadratureM = SpaceTreeQuadrature(grid, gaussPointsM, domain, depth)

    gaussPointsK = np.polynomial.legendre.leggauss(p + 1)
    quadratureK = SpaceTreeQuadrature(grid, gaussPointsK, domain, depth)

    # create system
    if spectral:
        system = TripletSystem.fromTwoQuadratures(ansatz, quadratureM, quadratureK)
    else:
        system = TripletSystem.fromOneQuadrature(ansatz, quadratureK)

    system.findZeroDof(-1e60)
    #system.findZeroDof(-1e60, [0, system.nDof()-1])
    if len(system.zeroDof) > 0:
        print("Warning! There were %d zero dof found: " % len(system.zeroDof) + str(system.zeroDof))

    # solve sparse
    M, K, MHRZ, MRS = system.createSparseMatrices(returnHRZ=True, returnRS=True)

    if mass == 'CON':
        # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, M, which='SM', return_eigenvectors=False)
        w = scipy.linalg.eigvals(K.toarray(), M.toarray())
    elif mass == 'HRZ':
        # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MHRZ, which='SM', return_eigenvectors=False)
        w = scipy.linalg.eigvals(K.toarray(), MHRZ.toarray())
    elif mass == 'RS':
        # w = scipy.sparse.linalg.eigs(K, K.shape[0] - 2, MRS, which='SM', return_eigenvectors=False)
        w = scipy.linalg.eigvals(K.toarray(), MRS.toarray())
    else:
        print("Error! Choose mass 'CON' or 'HRZ' or 'RS'")

    if np.linalg.norm(np.imag(w)) > 0:
        print("Warning! There were imaginary eigenvalues: " + str(w))

    # compute frequencies
    w = np.real(w)
    w = np.abs(w)
    w = np.sqrt(w + 0j)
    w = np.sort(w)

    if np.linalg.norm(np.imag(w)) > 0:
        print("Warning! There were negative eigenvalues: " + str(w))

    return np.real(w), system.nDof(), system.zeroDof


def createLegend():
    leg = ansatzType + ' C' + str(continuity)
    leg += ' ' + mass
    leg += ' d=' + str(extra)
    leg += ' dof=' + str(dof-len(zeroDof)) + " / " + str(dof)
    return leg


def plotStudy(lineStyle):
    global wNum
    #wNum = wNum[0:nw + 1]
    print(wNum)
    indices = np.linspace(0, len(wNum-1), len(wNum))
    wExact = (indices * np.pi) / (1.2 - 2 * extra)
    ax1.plot(indices, wNum, lineStyle, label=createLegend())
    ax2.plot(indices[1:], wNum[1:] / wExact[1:], '--o', label=createLegend())


# plot
figure, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(indices, wExact, '-', label='reference')
ax2.plot(indices[1:], wExact[1:] / wExact[1:], '-', label='reference')

# studies
ansatzType = 'Lagrange'
continuity = '0'
k = eval(continuity)
mass = 'RS'
wNum, dof, zeroDof = runStudy(int(n/(p-k)), p, extra, False, mass)
plotStudy('--o')


ansatzType = 'Spline'
continuity = 'p-1'
k = eval(continuity)
mass = 'RS'
wNum, dof, zeroDof = runStudy(int(n/(p-k)), p, extra, False, mass)
plotStudy('--x')


ansatzType = 'Lagrange'
continuity = '0'
k = eval(continuity)
mass = 'CON'
wNum, dof, zeroDof = runStudy(int(n/(p-k)), p, extra, False, mass)
plotStudy('-o')


ansatzType = 'Spline'
continuity = 'p-1'
k = eval(continuity)
mass = 'CON'
wNum, dof, zeroDof = runStudy(int(n/(p-k)), p, extra, False, mass)
plotStudy('-x')


ansatzType = 'Lagrange'
continuity = '0'
k = eval(continuity)
mass = 'CON'
wNum, dof, zeroDof = runStudy(int(n/(p-k)), p, extra, True, mass)
plotStudy('-.+')

ax1.legend()
ax2.legend()

plt.rcParams['axes.titleweight'] = 'bold'

title = 'Spectrum for p=' + str(p) + ' n=' + str(n) + " d=" + str(extra)
figure.suptitle(title)

plt.xlabel('eigenvalue index')
plt.ylabel('eigenvalue ')

plt.savefig('results/' + title.replace(' ', '_') + '.pdf')
plt.show()
