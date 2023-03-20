import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
  
# Domain definition
lengths = [2, 2]
duration = 12

center = [1.0, 1.0]
radius = 0.4

domain = lambda x, y : (x - center[0])**2 + (y - center[1])**2 >= radius**2

animationSpeed = 1

# Coefficients
alpha = lambda x, y : 1.0 if domain( x, y ) else 1e-6

rho = lambda x, y : alpha( x, y )
E  = lambda x, y : alpha( x, y )

# Source function (Ricklers wavelet)
frequency = 1
t0 = 1.0 / frequency

sigmaT = 1.0 / ( 2.0 * np.pi * frequency )
sigmaS = 0.06

ft = lambda t : -(t - t0) / (np.sqrt(2 * np.pi) * sigmaT**3) * np.exp( -(t - t0)**2 / (2 * sigmaT**2) )
fx = lambda x, y : 10*np.exp( -(x**2 + y**2)/(2*sigmaS**2)) if domain(x, y) else 0.0
    
# Discretization
nelements = [24, 24]
nsteps = 2000
polynomialDegree = 3
quadratureOrder = polynomialDegree + 1

treedepth = polynomialDegree + 1
nseedpoints = polynomialDegree + 2
 
dt = duration / nsteps
dx = lengths[0] / nelements[0]
dy = lengths[1] / nelements[1]
     
# Prepare Lagrange polynomials
lagrangeCoords = np.linspace(-1, 1, polynomialDegree + 1)
lagrangeValues = np.identity(polynomialDegree + 1)

lagrange = lambda i : scipy.interpolate.lagrange(lagrangeCoords, lagrangeValues[i])

shapesDiff0 = [lagrange(i) for i in range( polynomialDegree + 1 )]    
shapesDiff1 = [np.polyder(shape) for shape in shapesDiff0];
 
ndofelement = (polynomialDegree + 1)**2
ndofdirection = [nelements[0] * polynomialDegree + 1, nelements[1] * polynomialDegree + 1]
ndofglobal = ndofdirection[0] * ndofdirection[1]
    
print( "Assembling (" + str( ndofglobal ) + " dofs, " + str( nelements[0] * nelements[1] ) + " elements) ... ", flush=True )

# Allocate data structure for coordinate format
row   = np.zeros( ndofelement**2 * nelements[0] * nelements[1], dtype=np.uint )
col   = np.zeros( ndofelement**2 * nelements[0] * nelements[1], dtype=np.uint )
Mdata = np.zeros( ndofelement**2 * nelements[0] * nelements[1] )
Kdata = np.zeros( ndofelement**2 * nelements[0] * nelements[1] )

Fx = np.zeros( ndofglobal )

# Assembly
for ielement in range( nelements[0] ):
    for jelement in range( nelements[1] ):
      
        # Mapping function
        mapX = lambda r : ( ielement + r / 2.0 + 0.5 ) * dx
        mapY = lambda s : ( jelement + s / 2.0 + 0.5 ) * dy
      
        # Prepare location map and element linear system
        dofIndicesI = np.arange(polynomialDegree + 1) + polynomialDegree * ielement
        dofIndicesJ = np.arange(polynomialDegree + 1) + polynomialDegree * jelement
        locationMap = np.add.outer(dofIndicesI * ndofdirection[1], dofIndicesJ).ravel( )
               
        Me = np.zeros( ( ndofelement, ndofelement ) ) 
        Ke = np.zeros( ( ndofelement, ndofelement ) )
        Fe = np.zeros( ( ndofelement, ) )
        
        # Quad tree partitioning
        begin, end, depth = 0, 1, 0
        tree = [((-1.0, 1.0), (-1.0, 1.0))]
        
        partitions = []
        
        while begin != end and depth < treedepth:
            for (r0, r1), (s0, s1) in tree[begin:end]:
            
                # Create grid of points and evaluate embedded domain
                X, Y = np.meshgrid( np.linspace(mapX(r0), mapX(r1), nseedpoints), 
                                    np.linspace(mapY(s0), mapY(s1), nseedpoints), indexing='ij')
                
                result = domain(X.ravel( ), Y.ravel( ))
                
                # Subdivide if some are inside and some outside                
                if not np.all(result) and not np.all(np.logical_not(result)):
                    tree += [((r0, (r0 + r1)/2), (s0, (s0 + s1)/2)),
                             (((r0 + r1)/2, r1), (s0, (s0 + s1)/2)),
                             ((r0, (r0 + r1)/2), ((s0 + s1)/2, s1)),
                             (((r0 + r1)/2, r1), ((s0 + s1)/2, s1))]
                else:
                    partitions.append(((r0, r1), (s0, s1))) 
                             
            begin, end, depth = end, len(tree), depth + 1
        
        partitions += tree[begin:end]
            
        for (r0, r1), (s0, s1) in partitions:
            
            coordinates, weights = np.polynomial.legendre.leggauss( quadratureOrder )
            
            rValues = [xi*(r1 - r0)/2 + (r1 + r0)/2 for xi in coordinates]
            sValues = [et*(s1 - s0)/2 + (s1 + s0)/2 for et in coordinates]
            
            # Evaluate Lagrange polynomials in both directions
            shapesRDiff0 = [np.array([shape(r) for shape in shapesDiff0]) for r in rValues]
            shapesRDiff1 = [np.array([shape(r) for shape in shapesDiff1]) for r in rValues]
            shapesSDiff0 = [np.array([shape(s) for shape in shapesDiff0]) for s in sValues]
            shapesSDiff1 = [np.array([shape(s) for shape in shapesDiff1]) for s in sValues]
            
            for igauss, (r, w0) in enumerate(zip(rValues, weights)):
                for jgauss, (s, w1) in enumerate(zip(sValues, weights)):
                
                    x, y = mapX(r), mapY(s)
                    weight = (r1 - r0) * (s1 - s0) * dx * dy * w0 * w1
                    
                    # Compute tensor product and derivatives w.r.t. x and y
                    N    = np.outer( shapesRDiff0[igauss], shapesSDiff0[jgauss] ).ravel( )
                    dNdx = np.outer( shapesRDiff1[igauss] / ( 2 * dx ), shapesSDiff0[jgauss] ).ravel( )
                    dNdy = np.outer( shapesRDiff0[igauss], shapesSDiff1[jgauss] / ( 2 * dy ) ).ravel( )
                                    
                    # Add contributions
                    Me += np.outer( N, N ) * rho(x, y) * weight
                    Ke += ( np.outer( dNdx, dNdx ) + np.outer( dNdy, dNdy ) ) * E(x, y) * weight
                    Fe += N * fx(x, y) * weight
        
        elementIndex = jelement * nelements[0] + ielement
        elementSlice = slice(ndofelement**2 * elementIndex, ndofelement**2 * (elementIndex + 1))

        # Repeat locationMaps as colums/rows linearize the result to obtain matrix entry coordinates
        row[elementSlice] = np.broadcast_to( locationMap, (ndofelement, ndofelement) ).T.ravel( )
        col[elementSlice] = np.broadcast_to( locationMap, (ndofelement, ndofelement) ).ravel( )
        
        # Set data
        Mdata[elementSlice] = Me.ravel( )
        Kdata[elementSlice] = Ke.ravel( )
        
        # Add to spatial rhs
        Fx[locationMap] += Fe
        
# Create coordinate matrix and convert to compressed sparse column format
M = scipy.sparse.coo_matrix( (Mdata, (row, col)), shape=(ndofglobal, ndofglobal) ).tocsc( )
K = scipy.sparse.coo_matrix( (Kdata, (row, col)), shape=(ndofglobal, ndofglobal) ).tocsc( )

print( "Time integration (" + str( nsteps ) + " time steps) ... ", flush=True )

# Sparse LU factorization and time integration
factorized = scipy.sparse.linalg.splu( M )

u = np.zeros((nsteps + 1, ndofglobal))

for i in range( 2, nsteps + 1 ):
    u[i] = factorized.solve( M * ( 2 * u[i - 1] - u[i - 2] ) + dt**2 * ( Fx * ft( i * dt ) - K * u[i - 1] ) )

print( "Plotting ... ", flush=True )

# Plot animation (at Lagrange interpolation points)
X, Y = np.meshgrid(np.linspace(0, lengths[0], ndofdirection[0]), 
                   np.linspace(0, lengths[1], ndofdirection[1]), indexing='ij')
                   
Xe, Ye = np.meshgrid(np.linspace(0, lengths[0], nelements[0] + 1),
                     np.linspace(0, lengths[0], nelements[0] + 1), indexing='ij')
               
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')

def animate(i):
    ax.clear()
    Z = np.reshape( u[i], (ndofdirection[0], ndofdirection[1]) )
    ax.contourf(X, Y, Z, levels=np.linspace(-1.0, 1.0, 24), cmap='PuOr', extend='both')
    
    plt.plot(Xe, Ye, 'black')
    plt.plot(Ye, Xe, 'black')
    ax.add_artist(plt.Circle(center, radius, fill=False))
        
anim = animation.FuncAnimation(fig, animate, range(0, nsteps, 32), interval=1)

plt.show()
