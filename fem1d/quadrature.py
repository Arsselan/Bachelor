import numpy as np


def createGaussLegendreQuadraturePoints(nPoints):
    return np.polynomial.legendre.leggauss(nPoints)


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
        d = x2 - x1
        if d < 1e-14:
            print("Warning! Almost zero cell.")
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


class SmartQuadrature:
    def __init__(self, grid, localPointsAndWeights, domain, knownCuts):
        self.grid = grid
        self.localPoints = localPointsAndWeights[0]
        self.localWeights = localPointsAndWeights[1]
        self.nPoints = len(self.localPoints)
        self.domain = domain
        self.knownCuts = knownCuts

        self.points = [None] * grid.nElements
        self.weights = [None] * grid.nElements
        self.cuts = [None] * grid.nElements

        for iElement in range(grid.nElements):
            result = self.createPointsAndWeights(grid.pos(iElement, -1), grid.pos(iElement, 1))
            self.points[iElement] = result[0]
            self.weights[iElement] = result[1]
            self.cuts[iElement] = result[2]

    def createPointsAndWeights(self, x1, x2):
        d = x2 - x1
        if d < 1e-14:
            print("Warning! Almost zero cell.")

        cuts = []
        for cut in self.knownCuts:
            if x1 < cut < x2:
                cuts.append(cut)

        if len(cuts) == 0:
            points = [0] * self.nPoints
            weights = [0] * self.nPoints
            for j in range(self.nPoints):
                points[j] = x1 + d * 0.5 * (self.localPoints[j] + 1)
                weights[j] = self.localWeights[j] * d / 2
            return points, weights, cuts
        elif len(cuts) == 1:
            pointsL, weightsL, cutsL = self.createPointsAndWeights(x1, cuts[0])
            pointsR, weightsR, cutsR = self.createPointsAndWeights(cuts[0], x2)
            if len(cutsL) > 0 or len(cutsR) > 0:
                print("Error! Found cuts on deeper level.")
            return pointsL + pointsR, weightsL + weightsR, cuts
        else:
            print("Error! Element from %e to %e is cut multiple times." % (x1, x2))
        return [], [], []

