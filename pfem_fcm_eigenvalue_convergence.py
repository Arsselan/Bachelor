import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline
from scipy.interpolate import BSpline


def evaluateLagrangeBases(i, x, k, maxDerOrder, t):
    lagrangeCoords = np.linspace(t[i], t[i+1], k + 1)
    lagrangeValues = np.identity(k + 1)
    lagrange = lambda j : scipy.interpolate.lagrange(lagrangeCoords, lagrangeValues[j])
    shapesDiff0 = [lagrange(j) for j in range( k + 1 )]    
    shapesDiff1 = [np.polyder(shape) for shape in shapesDiff0];
    
    diff0 = np.array([shape(x) for shape in shapesDiff0])
    diff1 = np.array([shape(x) for shape in shapesDiff1])
    
    return [ diff0, diff1 ]

def runStudy(n, k, extra):
    #k = 1
    #n = 12
    depth = 10

    left = 0
    right = 1.2
    #extra = 0.0

    def salpha(x):
        if x>=left+extra and x<=right-extra:
            return 1.0
        return 0


    #print("Meshing...", flush=True)

    # create nodes
    length = right - left
    t = np.linspace(left, right, n+1)


    # create quadrature points
    gaussPoints = np.polynomial.legendre.leggauss(k+1)
    def qpoints(x1, x2, level=0):
        d = x2-x1;
        if salpha(x1)==salpha(x2) or level>=depth:
            points = [0]*(k+1)
            weights = [0]*(k+1)
            for j in range(k+1):
                points[j] = x1 + d * 0.5 * ( gaussPoints[0][j] + 1 )
                weights[j] = gaussPoints[1][j] * d / 2 * salpha(points[j])
            return points, weights
        else:
            pointsL, weightsL = qpoints(x1, x1+d/2, level+1)
            pointsR, weightsR = qpoints(x1+d/2, x2, level+1)
        return pointsL+pointsR, weightsL+weightsR

    # create matrices
    nval = (k+1)*(k+1)
    row  = np.zeros(nval*n, dtype=np.uint)
    col  = np.zeros(nval*n, dtype=np.uint)
    valM = np.zeros(nval*n)
    valK = np.zeros(nval*n)

    for i in range(n):
        lm = range(i*(k), (i+1)*(k)+1)
        #print("lm %d: " % (i) + str(list(lm)))
        Me = np.zeros( ( k+1, k+1 ) ) 
        Ke = np.zeros( ( k+1, k+1 ) )
        x1 = t[i]
        x2 = t[1+i]
        points, weights = qpoints(x1,x2)
        for j in range(len(points)):
            shapes = evaluateLagrangeBases(i, points[j], k, 1, t)
            N = np.asarray(shapes[0])
            B = np.asarray(shapes[1])
            Me += np.outer(N, N) * weights[j]
            Ke += np.outer(B, B) * weights[j]
        eslice = slice(nval * i, nval * (i + 1))
        row[eslice] = np.broadcast_to( lm, (k+1, k+1) ).T.ravel()
        col[eslice] = np.broadcast_to( lm, (k+1, k+1) ).ravel()
        valM[eslice] = Me.ravel()
        valK[eslice] = Ke.ravel()

    M = scipy.sparse.coo_matrix( (valM, (row, col)) ).tocsc( )
    K = scipy.sparse.coo_matrix( (valK, (row, col)) ).tocsc( )

    #w = scipy.sparse.linalg.eigs(K, K.shape[0]-2, M.toarray(), which='SM', return_eigenvectors=False)

    fullK = K.toarray();
    fullM = M.toarray();
    
    deleted = 1
    while deleted==1:    
        deleted = 0
        if(fullM[0,0]<1e-60):
            fullM=np.delete(fullM, 0, 0)
            fullM=np.delete(fullM, 0, 1)
            fullK=np.delete(fullK, 0, 0)
            fullK=np.delete(fullK, 0, 1)
            deleted = 1
            #print("Deleted left")
        if(fullM[-1,-1]<1e-60):
            fullM=np.delete(fullM, -1, 0)
            fullM=np.delete(fullM, -1, 1)
            fullK=np.delete(fullK, -1, 0)
            fullK=np.delete(fullK, -1, 1)
            deleted = 1
            #print("Deleted right")
    
    diagM = np.zeros(fullM.shape);
    for i in range(fullM.shape[0]):
        diagM[i,i] = sum(fullM[i,:])
    
    w = scipy.linalg.eigvals(fullK, diagM)    
    w = np.sqrt(np.abs(w))
    w = np.sort(w)
    #print(w)
    
    dofs = fullM.shape[0]
    
    return dofs, w[1+5]
    
    
def plot(ptx,pty):
    figure, ax = plt.subplots()
    ax.plot(ptx, pty)
    plt.show()
    
figure, ax = plt.subplots()
#ax.set_ylim(5, 500)

extra = 0.2



wexact = (6*np.pi)/(1.2-2*extra)

for p in range(4):
    nh = int(6.5-p)
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
#plt.title("consistent mass matrix")
plt.title("lumped mass matrix")
plt.xlabel('degrees of freedom')  
plt.ylabel('relative error in sixth eigenvalue ')  

#plt.savefig('pfem_eigenvalue_convergence_consistent_20.pdf')
plt.savefig('pfem_eigenvalue_convergence_lumped_20.pdf')
plt.show()

    
