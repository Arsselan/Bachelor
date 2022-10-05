import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline
from scipy.interpolate import BSpline


def runStudy(k, extra):
    #k = 1
    n = 12
    depth = 10

    left = 0
    right = 1.2
    #extra = 0.0

    def salpha(x):
        if x>=left+extra and x<=right-extra:
            return 1.0
        return 0


    print("Meshing...", flush=True)

    # create knot span
    length = right - left
    nextra = length / n * k
    t = np.linspace(left-nextra, right+nextra, n+1+2*k)
    for i in range(k+1):
        t[i] = left
        t[-i-1] = right

    #t[5] = t[6]
    lower = min(0,k-1)
    for i in range(n):
        for j in range(lower):
            t = np.insert(t, k+n-i-1, t[k+n-i-1])
    n = n + (lower)*n
    print(t)
    #t = np.insert(t, 5, t[5])
    #t = np.insert(t, 4, t[4])
    #t = np.insert(t, 4, t[4])
    #n=n+4

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
        lm = range(i, i+k+1)
        #print("lm %d: " % (i) + str(list(lm)))
        Me = np.zeros( ( k+1, k+1 ) ) 
        Ke = np.zeros( ( k+1, k+1 ) )
        x1 = t[k+i]
        x2 = t[k+1+i]
        if x1==x2:
            continue
        points, weights = qpoints(x1,x2)
        for j in range(len(points)):
            shapes = bspline.evaluateBSplineBases(k+i, points[j], k, 1, t)
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
            print("Deleted left")
        if(fullM[-1,-1]<1e-60):
            fullM=np.delete(fullM, -1, 0)
            fullM=np.delete(fullM, -1, 1)
            fullK=np.delete(fullK, -1, 0)
            fullK=np.delete(fullK, -1, 1)
            deleted = 1
            print("Deleted right")
    
    diagM = np.zeros(fullM.shape);
    for i in range(fullM.shape[0]):
        diagM[i,i] = sum(fullM[i,:])
    
    w = scipy.linalg.eigvals(fullK, diagM)    
    w = np.sqrt(np.abs(w))
    w = np.sort(w)
    print(w)
    
    return max(w)
    
    
def plot(ptx,pty):
    figure, ax = plt.subplots()
    ax.plot(ptx, pty)
    plt.show()
    
figure, ax = plt.subplots()
ax.set_ylim(5, 50)
for p in range(6):
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
plt.title("consistent mass matrix")
plt.xlabel('ficticious domain size')  
plt.ylabel('heighest eigenvalue')  

plt.savefig('eigenvalue_consistent.pdf')  
plt.show()

    
