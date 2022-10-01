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
        return 1e-10


    print("Meshing...", flush=True)

    # create knot span
    length = right - left
    nextra = length / n * k
    t = np.linspace(left-nextra, right+nextra, n+1+2*k)
    for i in range(k+1):
        t[i] = left
        t[-i-1] = right

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
    diagM = np.zeros(M.shape);
    
    for i in range(M.shape[0]):
        diagM[i,i] = sum(fullM[i,:])
    
    deleted = 1
    while deleted==1:    
        deleted = 0
        if(diagM[0,0]<1e-10):
            diagM=np.delete(diagM, 0, 0)
            diagM=np.delete(diagM, 0, 1)
            fullK=np.delete(fullK, 0, 0)
            fullK=np.delete(fullK, 0, 1)
            deleted = 1
            print("Deleted left")
        if(diagM[-1,-1]<1e-10):
            diagM=np.delete(diagM, -1, 0)
            diagM=np.delete(diagM, -1, 1)
            fullK=np.delete(fullK, -1, 0)
            fullK=np.delete(fullK, -1, 1)
            deleted = 1
            print("Deleted right")
    
    
    w = scipy.linalg.eigvals(fullK, diagM)    
    w = np.sqrt(np.abs(w))
    w = np.sort(w)
    print(w)
    
    return max(w)
    
    
def plot(ptx,pty):
    figure, ax = plt.subplots()
    ax.plot(ptx, pty)
    plt.show()
    
    
ne = 31
extras = np.linspace(0, 0.3, ne)
maxw = [0]*ne
for i in range(ne):
    maxw[i] = runStudy(2, extras[i])
    print("e = %e, wmax = %e" % (extras[i], maxw[i]))
    
plot(extras, maxw)
    
