import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import lagrange
import scipy.interpolate


def createGrid(left, right, n):
    return np.linspace(left, right, n)


class SplineAnsatz:
    def __init__(self, grid, p, k):
        self.grid = grid
        self.knots = createKnotVector(grid, p, k)
          
    def evaluate(self, pos, order):
        return evaluateSplineBasis(self.knots, span, pos, order)
    
    
class LagrangeAnsatz:
    def __init__(self, grid, points):
        self.grid = grid
              
    def evaluate(self, pos, order):
        localPos = -1 + 2 * (pos - grid.left) / grid.length
        basis = evaluateLagrangeBasis(points, pos, order)
        if order > 0:
            basis[1] = basis[1] * 2 / grid
        return basis
    
 

