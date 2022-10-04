import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import lagrange

# Gauss-Lobatto points and weights taken from here:
# https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb
def lgP (n, xi):
  if n == 0:
    return np.ones (xi.size)
  
  elif n == 1:
    return xi

  else:
    fP = np.ones (xi.size); sP = xi.copy (); nP = np.empty (xi.size)
    for i in range (2, n + 1):
      nP = ((2 * i - 1) * xi * sP - (i - 1) * fP) / i
      fP = sP; sP = nP

    return nP
    
def GLL(n, epsilon = 1e-15):
  if n < 2:
    print('Error: n must be larger than 1')
  
  else:
    x = np.empty (n)
    w = np.empty (n)
    
    x[0] = -1; x[n - 1] = 1
    w[0] = w[0] = 2.0 / ((n * (n - 1))); w[n - 1] = w[0];
    
    n_2 = n // 2
    
    dLgP  = lambda n, xi: n * (lgP (n - 1, xi) - xi * lgP (n, xi)) / (1 - xi ** 2)
    d2LgP = lambda n, xi: (2 * xi * dLgP (n, xi) - n * (n + 1) * lgP (n, xi)) / (1 - xi ** 2)
    d3LgP = lambda n, xi: (4 * xi * d2LgP (n, xi) - (n * (n + 1) - 2) * dLgP (n, xi)) / (1 - xi ** 2)                             
                                      
    for i in range (1, n_2):
      xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) *\
           np.cos ((4 * i + 1) * np.pi / (4 * (n - 1) + 1))
      
      error = 1.0
      
      while error > epsilon:
        y  =  dLgP (n - 1, xi)
        y1 = d2LgP (n - 1, xi)
        y2 = d3LgP (n - 1, xi)
        
        dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)
        
        xi -= dx
        error = abs (dx)
        
      x[i] = -xi
      x[n - i - 1] =  xi
      
      w[i] = 2 / (n * (n - 1) * lgP (n - 1, x[i]) ** 2)
      w[n - i - 1] = w[i]
    
    if n % 2 != 0:
      x[n_2] = 0;
      w[n_2] = 2.0 / ((n * (n - 1)) * lgP (n - 1, np.array (x[n_2])) ** 2)
    
  return np.array(x), np.array(w)
  
def evaluateLagrangeBases2(i, x, k, maxDerOrder, t):
    lagrangeCoords = np.linspace(t[i], t[i+1], k + 1)
    lagrangeValues = np.identity(k + 1)
    lagrange = lambda j : scipy.interpolate.lagrange(lagrangeCoords, lagrangeValues[j])
    shapesDiff0 = [lagrange(j) for j in range( k + 1 )]    
    shapesDiff1 = [np.polyder(shape) for shape in shapesDiff0];
    
    diff0 = np.array([shape(x) for shape in shapesDiff0])
    diff1 = np.array([shape(x) for shape in shapesDiff1])
    
    return [ diff0, diff1 ]

def runStudy(k, extra):
    #k = 1
    n = 12
    depth = 10

    left = 0
    right = 1.2
    #extra = 0.0
    
    #samppoints = np.linspace(-1, 1, k+1);
    gllPoints = GLL(k+1)
    samppoints = gllPoints[0]

    def salpha(x):
        if x>=left+extra and x<=right-extra:
            return 1.0
        return 0


    print("Meshing...", flush=True)

    # create nodes
    length = right - left
    t = np.linspace(left, right, n+1)


    # create quadrature points
    #gaussPoints = np.polynomial.legendre.leggauss(k+1)
    gaussPoints = GLL(k+1)
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
            #shapes = evaluateLagrangeBases2(i, points[j], k, 1, t)
            shapes = lagrange.evaluateLagrangeBases(i, points[j], samppoints, 1, t)
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
ax.set_ylim(5, 500)
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
plt.title("lumped mass matrix")
plt.xlabel('ficticious domain size')  
plt.ylabel('heighest eigenvalue')  

plt.savefig('pfem_eigenvalue_lumped.pdf')  
plt.show()

    
