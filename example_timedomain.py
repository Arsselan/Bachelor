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
right = 2.0

#method
#ansatzType = 'Lagrange'
ansatzType = 'Spline'
continuity = 'p-1'
lump = True
depth = 40

# analysis
nw = 10

p = 3
n = 40
extra = 0.0

tmax = 8.4
nt = 1000
dt = tmax / nt


# create grid and domain
grid = UniformGrid(left, right, n)

def alpha(x):
    if x>=left+extra and x<=right-extra-0.15:
        return 1.0
    return 0

domain = Domain(alpha)
    
# Source function (Ricklers wavelet)
frequency = 1
t0 = 1.0 / frequency

sigmaT = 1.0 / ( 2.0 * np.pi * frequency )
sigmaS = 0.03

ft = lambda t : -( t - t0 ) / ( np.sqrt( 2 * np.pi ) * sigmaT**3 ) * np.exp( -( t - t0 )**2 / ( 2 * sigmaT**2 ) )
fx = lambda x : 0.25 * alpha( x ) / np.sqrt( 2 * np.pi * sigmaS**2 ) * np.exp( -x**2 / ( 2 * sigmaS**2 ) )

if False:
    xx = np.linspace(left, right, 100)
    yy = xx * 0
    for i in range(len(xx)):
        yy[i] = fx(xx[i])

    figure, ax = plt.subplots()
    ax.plot(xx, yy) 
    plt.show()

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
system = TripletSystem(ansatz, quadrature, lump, fx)
system.findZeroDof()
M, K = system.createSparseMatrices()
F = system.getReducedVector(system.F)

# compute critical time step size
w = scipy.sparse.linalg.eigs(K, 1, M.toarray(), which='LM', return_eigenvectors=False)
w = np.sqrt(w[0] + 0j)

critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)

dt = critDeltaT * 0.9
nt = int(tmax / dt + 0.5)
dt = tmax / nt
print("Corrected time step size is %e" % dt)


# solve sparse
factorized = scipy.sparse.linalg.splu( M )

print( "Time integration ... ", flush=True )

u = np.zeros( ( nt + 1, M.shape[0] ) )
fullU = np.zeros( ( nt + 1, ansatz.nDof() ) )


for i in range( 2, nt + 1 ):
    u[i] = factorized.solve( M * ( 2 * u[i - 1] - u[i - 2] ) + dt**2 * ( F * ft( i * dt ) - K * u[i - 1] ) )
    fullU[i] = system.getFullVector(u[i])


# Plot animation
figure, ax = plt.subplots()
ax.set_xlim(grid.left, grid.right)
ax.set_ylim(-0.5, 2.1)
line,  = ax.plot(0, 0) 
line.set_xdata( np.linspace( grid.left, grid.right, ansatz.nDof() ) )

#ax.plot([0, xmax],[1, 1], '--b')
#ax.plot([interface, interface],[-0.5, 2.1], '--r')

#ax.legend()


plt.rcParams['axes.titleweight'] = 'bold'
title = 'Solution'
plt.title(title)
plt.xlabel('solution')  
plt.ylabel('x')  

animationSpeed = 4

def prepareFrame(i):
    line.set_ydata( fullU[int( round(i / tmax * nt) )] )
    return line,

frames = np.linspace(0, tmax, round( tmax * 60 / animationSpeed))
animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000/60, repeat=False)
                          
plt.show()





    
