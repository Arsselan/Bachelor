import unittest

import numpy as np
from context import fem1d


# Reference Gauss-Lobatto points and weights taken from here:
# https://colab.research.google.com/github/caiociardelli/sphglltools/blob/main/doc/L3_Gauss_Lobatto_Legendre_quadrature.ipynb
def lgP(nPoints, xi):
    if nPoints == 0:
        return np.ones(xi.size)
    elif nPoints == 1:
        return xi
    else:
        fP = np.ones(xi.size)
        sP = xi.copy()
        nP = np.empty(xi.size)
        for iPoint in range(2, nPoints + 1):
            nP = ((2 * iPoint - 1) * xi * sP - (iPoint - 1) * fP) / iPoint
            fP = sP
            sP = nP
        return nP


def GLL(nPoints, epsilon=1e-15):
    if nPoints < 2:
        print('Error: n must be larger than 1')

    else:
        x = np.empty(nPoints)
        w = np.empty(nPoints)

        x[0] = -1
        x[nPoints - 1] = 1
        w[0] = w[0] = 2.0 / (nPoints * (nPoints - 1))
        w[nPoints - 1] = w[0]

        n_2 = nPoints // 2

        dLgP = lambda n, xi: n * (lgP(n - 1, xi) - xi * lgP(n, xi)) / (1 - xi ** 2)
        d2LgP = lambda n, xi: (2 * xi * dLgP(n, xi) - n * (n + 1) * lgP(n, xi)) / (1 - xi ** 2)
        d3LgP = lambda n, xi: (4 * xi * d2LgP(n, xi) - (n * (n + 1) - 2) * dLgP(n, xi)) / (1 - xi ** 2)

        for i in range(1, n_2):
            xi = (1 - (3 * (nPoints - 2)) / (8 * (nPoints - 1) ** 3)) * \
                 np.cos((4 * i + 1) * np.pi / (4 * (nPoints - 1) + 1))

            error = 1.0

            while error > epsilon:
                y = dLgP(nPoints - 1, xi)
                y1 = d2LgP(nPoints - 1, xi)
                y2 = d3LgP(nPoints - 1, xi)

                dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)

                xi -= dx
                error = abs(dx)

            x[i] = -xi
            x[nPoints - i - 1] = xi

            w[i] = 2 / (nPoints * (nPoints - 1) * lgP(nPoints - 1, x[i]) ** 2)
            w[nPoints - i - 1] = w[i]

        if nPoints % 2 != 0:
            x[n_2] = 0
            w[n_2] = 2.0 / ((nPoints * (nPoints - 1)) * lgP(nPoints - 1, np.array(x[n_2])) ** 2)

    return np.array(x), np.array(w)


class TestGll(unittest.TestCase):
    def test_gll(self):
        for i in range(2, 20):
            eps = 1e-15
            pyGLL = GLL(i, eps)
            cppGLL = fem1d.gll.computeGllPoints(i, eps)

            pyPoints = np.array(pyGLL[0])
            cppPoints = np.array(cppGLL[0])

            pyWeights = np.array(pyGLL[1])
            cppWeights = np.array(cppGLL[1])

            places = 17
            for iPoint in range(i):
                self.assertAlmostEqual(pyPoints[iPoint], cppPoints[iPoint], places=places, msg="Point %d" % iPoint)

            places = 15
            for iPoint in range(i):
                self.assertAlmostEqual(pyWeights[iPoint], cppWeights[iPoint], places=places, msg="Weight %d" % iPoint)


if __name__ == '__main__':
    unittest.main()

