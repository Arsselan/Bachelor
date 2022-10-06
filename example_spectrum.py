import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline
#from scipy.interpolate import BSpline

from waves1d import *

# problem
left = 0
right = 1.2

# analysis
nw = 10
depth = 10


def runStudy(n, p, extra):
    
    # create grid and domain
    grid = UniformGrid(left, right, n)
    
    def alpha(x):
        if x>=left+extra and x<=right-extra:
            return 1.0
        return 0
    
    domain = Domain(alpha)
        
    # create ansatz
    if ansatzType == 'Spline':
        if continuity == 'p-1':
            k = p-1
        else:
            k = int(continuity)
        k = max(0, min(k, p-1))
        ansatz = SplineAnsatz(grid, p, k)
    elif ansatzType == 'Lagrange':
        gllPoints = GLL(p+1)
        ansatz = LagrangeAnsatz(grid, gllPoints[0])
    else:
        print("Error! Choose ansatzType 'Spline' or 'Lagrange'")
    
    #print(ansatz.knots)

    # create quadrature points
    gaussPoints = np.polynomial.legendre.leggauss(p+1)
    quadrature = SpaceTreeQuadrature(grid, gaussPoints, domain, depth)

    # create system
    system = TripletSystem(ansatz, quadrature, lump)
    system.findZeroDof()

    # solve sparse
    #M, K = system.createSparseMatrices()
    #w = scipy.sparse.linalg.eigs(K, K.shape[0]-2, M.toarray(), which='SM', return_eigenvectors=False)

    # solve dense
    fullM, fullK = system.createDenseMatrices()
    w = scipy.linalg.eigvals(fullK, fullM)    
    
    # compute frequencies
    #w = np.real(w)
    #w = np.abs(w)
    w = np.sqrt(w + 0j)
    w = np.sort(w)
    #print(w)

    return w
    

def createLegend():
    title = ansatzType + ' C' + str(continuity)
    if lump:
        title = title + ' lumped'
    else:
        title = title + ' consistent'
    title += ' d=' + str(extra)
    return title
    




p = 3
n = 20
extra = 0.20

indices = np.linspace(0, nw, nw+1)
wexact = (indices*np.pi)/(1.2-2*extra)   

figure, ax = plt.subplots()
    
ax.plot(indices, wexact,'-', label='reference')

#method
ansatzType = 'Lagrange'
continuity = '0'
lump = True
wnum = runStudy(n, p, extra)
wnum = wnum[0:nw+1]
ax.plot(indices, wnum,'--o', label=createLegend())


ansatzType = 'Spline'
continuity = '0'
lump = True
wnum = runStudy(n, p, extra)
wnum = wnum[0:nw+1]
ax.plot(indices, wnum,'--o', label=createLegend())

ansatzType = 'Lagrange'
continuity = '0'
lump = False
wnum = runStudy(n, p, extra)
wnum = wnum[0:nw+1]
ax.plot(indices, wnum,'--x', label=createLegend())


ansatzType = 'Spline'
continuity = '0'
lump = False
wnum = runStudy(n, p, extra)
wnum = wnum[0:nw+1]
ax.plot(indices, wnum,'--+', label=createLegend())


#errors = np.abs(wnum - wexact) / wexact

ax.legend()

plt.rcParams['axes.titleweight'] = 'bold'

title = 'Spectrum for p=' + str(p) + ' n=' + str(n) + " d=" + str(extra)
plt.title(title)

plt.xlabel('eigenvalue index')  
plt.ylabel('eigenvalue ')  

plt.savefig(title.replace(' ', '_') + '.pdf')
plt.show()

    
