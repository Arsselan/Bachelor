import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
  
#  PDE: rho * du/dt - d/dx ( E * du/dx ) = f     on Omega x T = [0, xmax] x [0, tmax]
#
#  x = 0              x = 0.5            x = 1
#    |------------------|------------------|

xmax = 1
tmax = 8.4
interface = 0.601

animationSpeed = 4

# Coefficients
alpha = lambda x : 1.0 if x <= interface else 1e-6

rho = lambda x : 1.0 * alpha( x )
E  = lambda x : 1.0 * alpha( x )

# Source function (Ricklers wavelet)
frequency = 1
t0 = 1.0 / frequency

sigmaT = 1.0 / ( 2.0 * np.pi * frequency )
sigmaS = 0.03

ft = lambda t : -( t - t0 ) / ( np.sqrt( 2 * np.pi ) * sigmaT**3 ) * np.exp( -( t - t0 )**2 / ( 2 * sigmaT**2 ) )
fx = lambda x : 0.25 * alpha( x ) / np.sqrt( 2 * np.pi * sigmaS**2 ) * np.exp( -x**2 / ( 2 * sigmaS**2 ) )
    
# Discretization
nx = 160
nt = 1000

dt = tmax / nt
dx = xmax / nx

# Matrix computation (assemble in coordiante format and convert to CSC)
ndofelement = 4

row  = np.zeros( ndofelement * nx, dtype=np.uint )
col  = np.zeros( ndofelement * nx, dtype=np.uint )
Mdata = np.zeros( ndofelement * nx )
Kdata = np.zeros( ndofelement * nx )

Fx = np.zeros( ( nx + 1, ) )
    
print( "Assembling ... ", flush=True )
    
for ielement in range( nx ):
    locationMap = [ielement, ielement + 1]
    gaussPoints = np.polynomial.legendre.leggauss( 2 )
    
    Me = np.zeros( ( 2, 2 ) ) 
    Ke = np.zeros( ( 2, 2 ) )
    Fe = np.zeros( ( 2, ) )
        
    for r, w in zip( *gaussPoints ):
        x = ( ielement + r / 2.0 + 0.5 ) * dx
        J = 2 * dx
        
        # Shape function vectors
        N = [0.5 * (1.0 - r), 0.5 * (1.0 + r)]
        dN = [-0.5 / J, 0.5 / J]
        
        # Add contributions
        Me += np.outer( N, N ) * rho(x) * w * J
        Ke += np.outer( dN, dN) * E(x) * w * J
        Fe += np.array( N ) * fx(x) * w * J

    elementSlice = slice(4 * ielement, 4 * (ielement + 1))

    # Repeat locationMap twice and linearize to obtain matrix entry coordinates
    row[elementSlice] = np.broadcast_to( locationMap, (2, 2) ).T.ravel( )
    col[elementSlice] = np.broadcast_to( locationMap, (2, 2) ).ravel( )

    Mdata[elementSlice] = Me.ravel( )
    Kdata[elementSlice] = Ke.ravel( )
    
    # Add to spatial rhs
    Fx[locationMap] += Fe

M = scipy.sparse.coo_matrix( ( Mdata, (row, col) ) ).tocsc( )
K = scipy.sparse.coo_matrix( ( Kdata, (row, col) ) ).tocsc( )

# Time integration and visualization

print( "Factorizing ... ", flush=True )

factorized = scipy.sparse.linalg.splu( M )

print( "Time integration ... ", flush=True )

u = np.zeros( ( nt + 1, nx + 1 ) )

for i in range( 2, nt + 1 ):
    u[i] = factorized.solve( M * ( 2 * u[i - 1] - u[i - 2] ) + dt**2 * ( Fx * ft( i * dt ) - K * u[i - 1] ) )

print( "Plotting ... ", flush=True )

# Plot animation
figure, ax = plt.subplots()
ax.set_xlim(0, xmax)
ax.set_ylim(-0.5, 2.1)
line,  = ax.plot(0, 0) 
line.set_xdata( np.linspace( 0, xmax, nx + 1 ) )

ax.plot([0, xmax],[1, 1], '--b')
ax.plot([interface, interface],[-0.5, 2.1], '--r')

def prepareFrame(i):
    line.set_ydata( u[int( round(i / tmax * nt) )] )
    return line,

frames = np.linspace(0, tmax, round( tmax * 60 / animationSpeed))
animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000/60)
                          
plt.show()

## Plot single frame
#t = 3 # 3.4
#plt.plot(np.linspace( 0, xmax, nx + 1 ), u[round(t / tmax * nt)]) 
#plt.plot([0, xmax],[1, 1], '--b')
#plt.plot([interface, interface],[-0.5, 2.1], '--r')
#plt.show()
