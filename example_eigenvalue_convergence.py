import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline
#from scipy.interpolate import BSpline

from waves1d import *

left = 0
right = 1.2
depth = 40

ansatzType = 'Lagrange'
#ansatzType = 'Spline'
continuity = '0'

lump = True
#eigenvalue = 10

trueContinuity = None

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
    w = np.sqrt(np.abs(w))
    w = np.sort(w)
    #print(w)
    
    dofs = system.nDof()
    
    wexact = (10*np.pi)/(1.2-2*extra)
        
    return dofs, find_nearest(w, wexact) # w[1+5]
    
    
figure, ax = plt.subplots()
#ax.set_ylim(5, 500)

extra = 0.20

nh = 8

wexact = (10*np.pi)/(1.2-2*extra)

for p in range(4):
    print("p = %d" % p)
    minw = [0]*nh
    errors = [0]*nh
    dofs = [0]*nh
    for i in range(nh):
        dofs[i], minw[i] = runStudy(12*int(1.5**(i)), p+1, extra)
        errors[i] = np.abs(minw[i] - wexact) / wexact
        print("dof = %e, wmin = %e, , e = %e" % (dofs[i], minw[i], errors[i]))        
    ax.loglog(dofs, errors,'-o', label='p=' + str(p+1))

ax.legend()

plt.rcParams['axes.titleweight'] = 'bold'

title = ansatzType + ' C' + str(continuity)
if lump:
    title = title + ' lumped'
else:
    title = title + ' consistent'
plt.title(title)

plt.xlabel('degrees of freedom')  
plt.ylabel('relative error in sixth eigenvalue ')  

#plt.savefig('eigenvalue_convergence_consistent_20.pdf')
plt.savefig('eigenvalue_convergence_lumped_20.pdf')
plt.show()

    
