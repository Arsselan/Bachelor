import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline
#from scipy.interpolate import BSpline

from waves1d import *
from progress import *

# problem
left = 0
right = 2.0
extra = 0.0

#method
ansatzType = 'Lagrange'
#ansatzType = 'Spline'
continuity = 'p-1'
lump = False
depth = 40

def runStudy(n, p):

    #p = 4
    #n = 200

    tmax = 0.7
    nt = 1000
    dt = tmax / nt

    # create grid and domain
    grid = UniformGrid(left, right, n)
    rightBoundary = right-extra-grid.elementSize * 5.5 * 0
    L = rightBoundary
    pi = np.pi

    def alpha(x):
        if x>=left+extra and x<=rightBoundary:
            return 1.0
        return 1e-20

    domain = Domain(alpha)
        
    # Source function (Ricklers wavelet)
    frequency = 1
    t0 = 1.0 / frequency

    sigmaT = 1.0 / ( 2.0 * np.pi * frequency )
    sigmaS = 0.03

    #ft = lambda t : -( t - t0 ) / ( np.sqrt( 2 * np.pi ) * sigmaT**3 ) * np.exp( -( t - t0 )**2 / ( 2 * sigmaT**2 ) )
    #fx = lambda x : 0.25 * alpha( x ) / np.sqrt( 2 * np.pi * sigmaS**2 ) * np.exp( -x**2 / ( 2 * sigmaS**2 ) )

    # manufactured solution
    #u(x,t) = cos(2*pi*x/L) * sin(2*pi*t)
    #u'(x,t) = -2*pi*/L*sin(2*pi*x/L) * sin(2*pi*t)
    #u''(x,t) = -4*pi^2/L^2*cos(2*pi*x/L)  * sin(2*pi*t)
    #dudt(x,t) = cos(2*pi*x/L) * 2*pi*cos(2*pi*t)
    #ddudt^2(x,t) = -cos(2*pi*x/L) * 4*pi^2*sin(2*pi*t)


    def uxt(x,t):
        return 1e-1 * np.cos(2*pi*x/L) * np.sin(2*pi*t)

    def fx(x):
        return np.cos(2*pi*x/L)
        
    def ft(t):
        return -1e-2 * 4*pi**2 * ( np.sin(2*pi*t) + 1/L**2 * np.sin(2*pi*t) )
        
    def fxt(x,t):
        return fx(x)*ft(t)


    # manufactured solution 2
    #u(x,t) = cos(2*pi*x/L) * 1/3*sin(2*pi*t)**3
    #u'(x,t) = -2*pi*/L*sin(2*pi*x/L) * 1/3*sin(2*pi*t)**3
    #u''(x,t) = -4*pi**2/L**2*cos(2*pi*x/L)  * 1/3*sin(2*pi*t)**3
    #dudt(x,t) = cos(2*pi*x/L) * sin(2*pi*t)**2 * 2*pi*cos(2*pi*t)
    #ddudt^2(x,t) = cos(2*pi*x/L) * ( 2*sin(2*pi*t) * 4*pi**2*cos(2*pi*t)**2 - sin(2*pi*t)**2 * 4*pi**2*sin(2*pi*t) )

    wx = 2*pi/L*4
    wt = 2*pi

    def uxt(x,t):
        return np.cos(wx*x) * 1/3*np.sin(wt*t)**3

    def fx(x):
        return np.cos(wx*x)
        
    def ft(t):
        sin = np.sin(wt*t)
        cos = np.cos(wt*t)
        return ( 2*sin * wt**2*cos**2 - sin**2 * wt**2*sin + wx**2 * 1/3*sin**3 )
        
    def fxt(x,t):
        return fx(x)*ft(t)


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

    dt = 1e-5#critDeltaT * 0.1
    nt = int(tmax / dt + 0.5)
    dt = tmax / nt
    print("Corrected time step size is %e" % dt)


    # solve sparse
    factorized = scipy.sparse.linalg.splu( M )

    print( "Time integration ... ", flush=True )

    u = np.zeros( ( nt + 1, M.shape[0] ) )
    fullU = np.zeros( ( nt + 1, ansatz.nDof() ) )
    evalU = 0*fullU

    nodes = np.linspace( grid.left, grid.right, ansatz.nDof() )
    I = ansatz.interpolationMatrix( nodes )
        
    nodes2 = np.linspace( grid.left, rightBoundary, 1000 )
    I2 = ansatz.interpolationMatrix( nodes2 )
    
    #printProgressBar(0, nt+1, prefix = 'Progress:', suffix = 'Complete', length = 50)
   
    errorSum = 0 
    for i in range( 2, nt + 1 ):
        u[i] = factorized.solve( M * ( 2 * u[i - 1] - u[i - 2] ) + dt**2 * ( F * ft( i * dt ) - K * u[i - 1] ) )
        fullU[i] = system.getFullVector(u[i])
        evalU[i] = I*fullU[i]
        evalU2 = I2 * fullU[i]
        errorSum += dt*np.linalg.norm( (evalU2 - uxt(nodes2, (i+1)*dt ))/system.nDof() )
            
        #if i % int(nt / 100) == 0:
        #    print( np.linalg.norm(evalU[i] - uxt(nodes, i*dt )) )
            #printProgressBar(i, nt + 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

    
    print( "Error: %e " % errorSum )
    
    return errorSum, system.nDof(), dt

# Plot animation
def postprocess():
    figure, ax = plt.subplots()
    ax.set_xlim(grid.left, grid.right)
    ax.set_ylim(-2, 2)

    ax.plot([rightBoundary, rightBoundary], [-0.1, 0.1], '--', label='domain boundary')

    line,  = ax.plot(0, 0, label='conrrol points') 
    line.set_xdata( np.linspace( grid.left, grid.right, ansatz.nDof() ) )

    line2,  = ax.plot(0, 0, label='numerical') 
    line2.set_xdata( nodes )

    line3,  = ax.plot(0, 0, '--', label='analytical') 
    line3.set_xdata( nodes )

    #ax.plot([0, xmax],[1, 1], '--b')
    #ax.plot([interface, interface],[-0.5, 2.1], '--r')

    ax.legend()


    plt.rcParams['axes.titleweight'] = 'bold'
    title = 'Solution'
    plt.title(title)
    plt.xlabel('solution')  
    plt.ylabel('x')  

    animationSpeed = 1

    def prepareFrame(i):
        plt.title(title + " time %3.2e" % i)
        line.set_ydata( fullU[int( round(i / tmax * nt) )] )
        line2.set_ydata( evalU[int( round(i / tmax * nt) )] )
        line3.set_ydata( uxt(nodes, i ) )
        return line,

    frames = np.linspace(0, tmax, round( tmax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000/60, repeat=False)
                              
    plt.show()

#postprocess()

figure, ax = plt.subplots()

nRef = 4
errors = [0]*nRef
dofs = [0]*nRef
dts = [0]*nRef
for p in [1,2,3,4]:
    print("p=%d" % p)
    for i in range(nRef):
        errors[i], dofs[i], dts[i] = runStudy(int(10*2**i), p)
    ax.loglog(dofs, errors,'-o', label='p=' + str(p))


ax.legend()

plt.rcParams['axes.titleweight'] = 'bold'

title = ansatzType + ' C' + str(continuity)
if lump:
    title = title + ' lumped'
else:
    title = title + ' consistent'
title += ' d=' + str(extra)
plt.title(title)

plt.xlabel('degrees of freedom')  
plt.ylabel('relative error in sixth eigenvalue ')  

plt.savefig(title.replace(' ', '_') + '.pdf')
plt.show()

