import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import lagrange
import bspline

from gll import *

class UniformGrid:
    def __init__(self, left, right, nElements):
        self.left = left
        self.right = right
        self.length = right - left
        self.elementSize = self.length / nElements
        self.nElements = nElements
        
    def pos(self, iElement, localCoord):
        return self.left + self.elementSize * (iElement + (localCoord + 1 ) / 2)

    def elementIndex(self, globalPos):
        return min(self.nElements - 1, int( (globalPos - self.left) / self.length * self.nElements ) )

    def localPos(self, globalPos):
        return -1 + 2 * (globalPos - self.left) / self.length

class Domain:
    def __init__(self, alphaFunc):
        self.alphaFunc = alphaFunc

    def alpha(self, globalPos):
        return self.alphaFunc(globalPos)


class SpaceTreeQuadrature:
    def __init__(self, grid, localPointsAndWeights, domain, depth):
        self.grid = grid
        self.localPoints = localPointsAndWeights[0]
        self.localWeights = localPointsAndWeights[1]
        self.nPoints = len(self.localPoints)
        self.domain = domain
        self.depth = depth
        
        self.points = [None] * grid.nElements
        self.weights = [None] * grid.nElements
        self.cuts =  [None] * grid.nElements
        
        for iElement in range(grid.nElements):
            result = self.createPointsAndWeights(grid.pos(iElement, -1), grid.pos(iElement, 1))
            self.points[iElement] = result[0]
            self.weights[iElement] = result[1]
            self.cuts[iElement] = result[2]
            
    def createPointsAndWeights(self, x1, x2, cuts = [], level = 0):
        d = x2 - x1;
        if self.domain.alpha(x1) == self.domain.alpha(x2) or level >= self.depth:
            points = [0]*self.nPoints
            weights = [0]*self.nPoints
            for j in range(self.nPoints):
                points[j] = x1 + d * 0.5 * ( self.localPoints[j] + 1 )
                weights[j] = self.localWeights[j] * d / 2 * self.domain.alpha(points[j])
            return points, weights, cuts
        else:
            cuts = cuts + [x1+d/2]
            pointsL, weightsL, cuts = self.createPointsAndWeights(x1, x1+d/2, cuts, level+1)
            pointsR, weightsR, cuts = self.createPointsAndWeights(x1+d/2, x2, cuts, level+1)
        return pointsL+pointsR, weightsL+weightsR, cuts
        

def createKnotVector(grid, p, k):
    extra = grid.length / grid.nElements * p
    t = np.linspace(grid.left-extra, grid.right+extra, grid.nElements+1+2*p)
    for i in range(p+1):
        t[i] = grid.left
        t[-i-1] = grid.right

    lower = min(p-k-1, p-1)
    for i in range(grid.nElements-1):
        for j in range(lower):
            t = np.insert(t, p+grid.nElements-i-1, t[p+grid.nElements-i-1])
    
    return t


class SplineAnsatz:
    def __init__(self, grid, p, k):
        self.grid = grid
        self.p = p
        self.k = k
        self.knots = createKnotVector(grid, p, k)
          
    def spanIndex(self, iElement):
        return self.p + iElement * (self.p - self.k)
        
    def evaluate(self, pos, order):
        iElement = self.grid.elementIndex(pos)
        iSpan = self.spanIndex(iElement)
        #print("e=%d, s=%s" % (iElement, iSpan))
        return bspline.evaluateBSplineBases(iSpan, pos, self.p, order, self.knots)
    
    def locationMap(self, iElement):
        iShape = (self.spanIndex(iElement) - self.p)# * (self.p - self.k)
        return range(iShape, iShape+self.p+1)

class LagrangeAnsatz:
    def __init__(self, grid, points):
        self.grid = grid
        self.points = points
        self.p = len(points) - 1
        self.knots = np.linspace(grid.left, grid.right, grid.nElements+1)
              
    def evaluate(self, pos, order):
        # Better do this
        #localPos = self.grid.localPos(pos)
        #basis = lagrange.evaluateLagrangeBases(self.points, localPos, order)
        #if order > 0:
        #    basis[1] = basis[1] * 2 / grid.elementSize
        # than this:
        iElement = self.grid.elementIndex(pos)
        basis = lagrange.evaluateLagrangeBases(iElement, pos, self.points, order, self.knots)
        return basis
    
    def locationMap(self, iElement):
        iShape = iElement*self.p
        return range(iShape, iShape+self.p+1)
        
    
class TripletSystem:
    def __init__(self, ansatz, quadrature, lump = False):
        self.lump = lump
        
        p = ansatz.p
        grid = ansatz.grid
        n = grid.nElements
        
        nval = (p+1)*(p+1)
        row  = np.zeros(nval*n, dtype=np.uint)
        col  = np.zeros(nval*n, dtype=np.uint)
        valM = np.zeros(nval*n)
        valK = np.zeros(nval*n)
        
        for i in range(n):
            lm = ansatz.locationMap(i)
            #print("lm %d: " % (i) + str(list(lm)))
            Me = np.zeros( ( p+1, p+1 ) ) 
            Ke = np.zeros( ( p+1, p+1 ) )
            x1 = grid.pos(i, -1)
            x2 = grid.pos(i, 1)
            points = quadrature.points[i]
            weights = quadrature.weights[i]
            for j in range(len(points)):
                shapes = ansatz.evaluate(points[j], 1)
                #print(shapes[0])
                N = np.asarray(shapes[0])
                B = np.asarray(shapes[1])
                Me += np.outer(N, N) * weights[j]
                Ke += np.outer(B, B) * weights[j]
                
            eslice = slice(nval * i, nval * (i + 1))
            row[eslice] = np.broadcast_to( lm, (p+1, p+1) ).T.ravel()
            col[eslice] = np.broadcast_to( lm, (p+1, p+1) ).ravel()
            valK[eslice] = Ke.ravel()

            if(lump):
                diagMe = np.zeros(Me.shape);
                for i in range(Me.shape[0]):
                    diagMe[i,i] = sum(Me[i,:])
                valM[eslice] = diagMe.ravel()
            else:                
                valM[eslice] = Me.ravel()

        
        self.valM = valM
        self.valK = valK
        self.row = row
        self.col = col
    
    def nDof(self):
        return int(max(self.row) + 1)
    
    def findZeroDof(self):
        nDof = self.nDof()
        diag = [0]*nDof
        nVals = len(self.row) 
        for i in range(nVals):
            iRow = self.row[i]
            diag[iRow] += self.valM[i]
        self.zeroDof = []
        self.dofMap = [0]*nDof
        self.nNonZeroDof = 0
        for i in range(nDof):
            if diag[i] == 0:
                self.zeroDof.append(i)
            else:
                self.dofMap[i] = self.nNonZeroDof
                self.nNonZeroDof += 1
    
    def getReducedRowAndCol(self):
        if hasattr(self, 'dofMap'):
            nVals = len(self.row)
            row = [0] * nVals
            col = [0] * nVals
            for i in range(nVals):
                row[i] = self.dofMap[self.row[i]]
                col[i] = self.dofMap[self.col[i]]
        else:
            row = self.row
            col = self.col
        return row, col
        
    def createDenseMatrices(self):
        row, col = self.getReducedRowAndCol()
        M = scipy.sparse.coo_matrix( (self.valM, (row, col)) ).tocsc( )
        K = scipy.sparse.coo_matrix( (self.valK, (row, col)) ).tocsc( )
        fullK = K.toarray();
        fullM = M.toarray();
        return fullM, fullK
        
    def createSparseMatrices(self):
        row, col = self.getReducedRowAndCol()            
        M = scipy.sparse.coo_matrix( (self.valM, (row, col)) ).tocsc( )
        K = scipy.sparse.coo_matrix( (self.valK, (row, col)) ).tocsc( )
        return M, K
        

def removeZeroDof(fullK, fullM):
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
    return fullK, fullM
    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    

def plot(ptx,pty):
    figure, ax = plt.subplots()
    ax.plot(ptx, pty)
    plt.show()
