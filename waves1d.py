import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import lagrange
import bspline
import gll

from sandbox.gllTemp import *
from fem1d.ansatz import *
from fem1d.system import *
from fem1d.utilities import *


class UniformGrid:
    def __init__(self, left, right, nElements):
        self.left = left
        self.right = right
        self.length = right - left
        self.elementSize = self.length / nElements
        self.nElements = nElements

    def pos(self, iElement, localCoord):
        return self.left + self.elementSize * (iElement + (localCoord + 1) / 2)

    def elementIndex(self, globalPos):
        return min(self.nElements - 1, int((globalPos - self.left) / self.length * self.nElements))

    def localPos(self, globalPos):
        return -1 + 2 * (globalPos - self.left - self.elementIndex(globalPos) * self.elementSize) / self.elementSize


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
        self.cuts = [None] * grid.nElements

        for iElement in range(grid.nElements):
            result = self.createPointsAndWeights(grid.pos(iElement, -1), grid.pos(iElement, 1))
            self.points[iElement] = result[0]
            self.weights[iElement] = result[1]
            self.cuts[iElement] = result[2]

    def createPointsAndWeights(self, x1, x2, cuts=[], level=0):
        d = x2 - x1;
        if self.domain.alpha(x1) == self.domain.alpha(x2) or level >= self.depth:
            points = [0] * self.nPoints
            weights = [0] * self.nPoints
            for j in range(self.nPoints):
                points[j] = x1 + d * 0.5 * (self.localPoints[j] + 1)
                weights[j] = self.localWeights[j] * d / 2
            return points, weights, cuts
        else:
            cuts = cuts + [x1 + d / 2]
            pointsL, weightsL, cuts = self.createPointsAndWeights(x1, x1 + d / 2, cuts, level + 1)
            pointsR, weightsR, cuts = self.createPointsAndWeights(x1 + d / 2, x2, cuts, level + 1)
        return pointsL + pointsR, weightsL + weightsR, cuts


def createAnsatz(ansatzType, continuity, p, grid):
    if ansatzType == 'Spline':
        k = eval(continuity)
        k = max(0, min(k, p - 1))
        ansatz = SplineAnsatz(grid, p, k)
    elif ansatzType == 'InterpolatorySpline':
        k = eval(continuity)
        k = max(0, min(k, p - 1))
        ansatz = InterpolatorySplineAnsatz(grid, p, k)
    elif ansatzType == 'Lagrange':
        gllPoints = gll.computeGllPoints(p + 1)
        ansatz = LagrangeAnsatz(grid, gllPoints[0])
    else:
        print("Error! Choose ansatzType 'Spline' or 'Lagrange' or 'InterpolatorySpline'.")
        return None

    return ansatz
