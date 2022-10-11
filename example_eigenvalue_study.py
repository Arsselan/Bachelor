import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import example_plot_bspline
#from scipy.interpolate import BSpline

from waves1d import *

# problem
left = 0
right = 1.2

#method
ansatzType = 'Lagrange'
#ansatzType = 'Spline'
continuity = '0'
lump = True
depth = 40

# analysis
n = 12
axLimitY = 500

if ansatzType == 'Lagrange':
    continuity = '0'
        
        
def runStudy(p, extra):
    
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
    system.findZeroDof(0.0)

    # solve sparse
    #M, K = system.createSparseMatrices()
    #w = scipy.sparse.linalg.eigs(K, K.shape[0]-2, M.toarray(), which='LM', return_eigenvectors=False)

    # solve dense
    fullM, fullK = system.createDenseMatrices()
    #removeZeroDof(fullK, fullM)
    
    w = scipy.linalg.eigvals(fullK, fullM)    
    
    w = np.sqrt(np.abs(w))
    w = np.sort(w)

    return max(w)
    

figure, ax = plt.subplots()
ax.set_ylim(5, axLimitY)
for p in range(4):
    ne = 11
    extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [0.3]
    ne = len(extras)
    maxw = [0]*ne
    for i in range(ne):
        maxw[i] = runStudy(p+1, extras[i])
        print("e = %e, wmax = %e" % (extras[i], maxw[i]))        
    ax.plot(extras, maxw,'-o', label='p=' + str(p+1))

ax.legend()

plt.rcParams['axes.titleweight'] = 'bold'

title = ansatzType + ' C' + str(continuity)
if lump:
    title = title + ' lumped'
else:
    title = title + ' consistent'
plt.title(title)

plt.xlabel('ficticious domain size')  
plt.ylabel('largest eigenvalue')  

plt.savefig(title.replace(' ', '_') + '.pdf')
plt.show()

    
