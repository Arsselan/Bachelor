import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline
from scipy.interpolate import BSpline


k = 5
n = 12
depth = 3

left = 0
right = 1.2
extra = 0.04

def salpha(x):
    if x>=left+extra and x<=right-extra:
        return 1.0
    return 1e-6


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
    print("lm %d: " % (i) + str(list(lm)))
    Me = np.zeros( ( 2, 2 ) ) 
    Ke = np.zeros( ( 2, 2 ) )
    x1 = t[k+i]
    x2 = t[k+1+i]
    points, weights = qpoints(x1,x2)
    for j in range(len(points)):
        shapes = evalbspline(t, k, i, points[j], 1)
        N = np.asarray(shapes[0])
        B = np.asarray(shapes[1])
        Me += np.outer(N, N) * weights[j]
        Ke += np.outer(B, B) * weights[j]        
    eslice = slice(nval * i, nval * (i + 1))
    row[eslice] = np.broadcast_to( locationMap, (k+1, k+1) ).T.ravel()
    col[eslice] = np.broadcast_to( locationMap, (k+1, k+1) ).ravel()
    valM[eslice] = Me.ravel()
    valK[eslice] = Ke.ravel()

M = scipy.sparse.coo_matrix( (valM, (row, col)) ).tocsc( )
K = scipy.sparse.coo_matrix( (valK, (row, col)) ).tocsc( )

        
        
allpoints = []
allweights = []
for i in range(n):
    x1 = t[k+i]
    x2 = t[k+1+i]
    print("Element from %e to %e" % (x1, x2))
    points, weights = qpoints(x1, x2)
    allpoints = allpoints + points
    allweights = allweights + weights
    
# plot
print("Plotting...", flush=True)
figure, ax = plt.subplots()

ax.set_xlim(left, right)
ax.set_ylim(-0.5, 1.5)


ax.plot(t, np.zeros(t.size), '-o')
ax.plot(t, np.ones(t.size), '-o')
ax.plot(allpoints, allweights, 'x')

plt.show()

