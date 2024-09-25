import unittest

import numpy as np
from context import fem1d


class TestSystem(unittest.TestCase):
    def test_system_linear(self):
        grid = fem1d.UniformGrid(0, 30, 10)

        def alpha(x):
            return 1

        ansatz = fem1d.createAnsatz("Spline", "p-1", 1, grid)
        domain = fem1d.Domain(alpha)
        glPoints = np.polynomial.legendre.leggauss(2)
        quadrature = fem1d.SpaceTreeQuadrature(grid, glPoints, domain, 1)

        def source(x):
            return x*x

        system = fem1d.TripletSystem(ansatz)
        matrices = fem1d.WaveEquationStandardMatrices(1.0, 1.0, source)
        matrices = fem1d.WaveEquationLumpedMatrices(matrices)
        fem1d.computeSystemMatrices(system, ansatz, quadrature, matrices)

        def entryM(iEntry):
            if iEntry % 4 == 0 or iEntry % 4 == 3:
                return 1.0
            else:
                return 0.5

        def entryMRS(iEntry):
            if iEntry % 4 == 0 or iEntry % 4 == 3:
                return 1.5
            else:
                return 0.0

        def entryK(iEntry):
            if iEntry % 4 == 0 or iEntry % 4 == 3:
                return 1.0/3.0
            else:
                return -1.0/3.0

        places = 14
        for i in range(system.matrixValues['M'].size):
            self.assertAlmostEqual(system.matrixValues['M'][i], entryM(i), places=places, msg="Mass matrix entry %d" % i)
            self.assertAlmostEqual(system.matrixValues['MRS'][i], entryMRS(i), places=places, msg="Row-summed mass matrix entry %d" % i)
            self.assertAlmostEqual(system.matrixValues['MHRZ'][i], entryMRS(i), places=places, msg="HRZ lumped mass matrix entry %d" % i)
            self.assertAlmostEqual(system.matrixValues['K'][i], entryK(i), places=places, msg="Stiffness matrix entry %d" % i)

        for i in range(system.row.size):
            self.assertEqual(system.row[i], int((i + 2) / 4))
            self.assertEqual(system.col[i], int(i/4) + (i % 2))

        ref = [9.0, 126.0, 450.0, 990.0, 1746.0, 2718.0, 3906.0, 5310.0, 6930.0, 8766.0, 5049.0]
        for i in range(system.vectors['F'].size):
            print(system.vectors['F'][i]*4.0)
            self.assertAlmostEqual(system.vectors['F'][i], ref[i] / 4.0, places=12, msg="Load vector  entry %d" % i)

    def test_system_quadratic(self):
        grid = fem1d.UniformGrid(0, 30, 1)

        def alpha(x):
            return 1.0 if x <= 15.0 else 0.0

        ansatz = fem1d.createAnsatz("Lagrange", "p-1", 2, grid)
        domain = fem1d.Domain(alpha)
        glPoints = np.polynomial.legendre.leggauss(7)
        quadrature = fem1d.SpaceTreeQuadrature(grid, glPoints, domain, 20)

        def source(x):
            return 0

        system = fem1d.TripletSystem(ansatz)
        matrices = fem1d.WaveEquationStandardMatrices(1.0, 1.0, source)
        matrices = fem1d.WaveEquationLumpedMatrices(matrices)
        fem1d.computeSystemMatrices(system, ansatz, quadrature, matrices)

        def entryM(iEntry):
            ref = [3.875,  2.875, -0.5, 2.875, 8.0, -0.875, -0.5, -0.875, 0.125]

            return ref[iEntry % 9]

        def entryMRS(iEntry):
            ref = [6.25,  0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -1.25]
            return ref[iEntry % 9]

        def entryMHRZ(iEntry):
            ref = [4.84375, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.15625]
            return ref[iEntry % 9]

        def entryK(iEntry):
            ref = [6.5, -7.0, 0.5, -7.0, 8.0, -1.0, 0.5, -1.0, 0.5]
            return ref[iEntry % 9] / 90.0

        places = 13
        for i in range(system.matrixValues['M'].size):
            self.assertAlmostEqual(system.matrixValues['M'][i], entryM(i), places=places, msg="Mass matrix entry %d" % i)
            self.assertAlmostEqual(system.matrixValues['MRS'][i], entryMRS(i), places=places, msg="Row-summed mass matrix entry %d" % i)
            self.assertAlmostEqual(system.matrixValues['MHRZ'][i], entryMHRZ(i), places=places-1, msg="HRZ lumped mass matrix entry %d" % i)
            self.assertAlmostEqual(system.matrixValues['K'][i], entryK(i), places=places, msg="Stiffness matrix entry %d" % i)

        self.assertAlmostEqual(system.minNonZeroMass, 15, places=places, msg="Minimum non-zero mass")

    def test_system_benchmark(self):

        for p in range(1):
            grid = fem1d.UniformGrid(0, 30, 1000)

            def alpha(x):
                return 1.0 if x <= 15.0 else 0.0

            ansatz = fem1d.createAnsatz("Lagrange", "p-1", p, grid)
            domain = fem1d.Domain(alpha)
            glPoints = np.polynomial.legendre.leggauss(p+1)
            quadrature = fem1d.SpaceTreeQuadrature(grid, glPoints, domain, 20)

            def source(x):
                return 0

            system = fem1d.TripletSystem(ansatz)
            matrices = fem1d.WaveEquationStandardMatrices(1.0, 1.0, source)
            matrices = fem1d.WaveEquationLumpedMatrices(matrices)
            fem1d.computeSystemMatrices(system, ansatz, quadrature, matrices)

    def test_system_spectral(self):

        p = 3

        grid = fem1d.UniformGrid(0, 30, 10)

        def alpha(x):
            return 1.0 if x <= 50.0 else 0.0

        ansatz = fem1d.createAnsatz("Lagrange", "p-1", p, grid)
        domain = fem1d.Domain(alpha)
        glPoints = np.polynomial.legendre.leggauss(p + 1)
        quadratureK = fem1d.SpaceTreeQuadrature(grid, glPoints, domain, 20)
        gllPoints = fem1d.gll.computeGllPoints(p + 1)
        quadratureM = fem1d.SpaceTreeQuadrature(grid, gllPoints, domain, 20)

        def source(x):
            return 0

        system = fem1d.TripletSystem(ansatz)

        matrices = fem1d.WaveEquationStiffnessMatrixAndLoadVector(1.0, 1.0, source)
        fem1d.computeSystemMatrices(system, ansatz, quadratureK, matrices)

        matrices = fem1d.WaveEquationMassMatrix(1.0)
        matrices = fem1d.WaveEquationLumpedMatrices(matrices)
        fem1d.computeSystemMatrices(system, ansatz, quadratureM, matrices)

        ref = [0.25, 0.0, 0.0, 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.25]
        for i in range(system.matrixValues['M'].size):
            self.assertAlmostEqual(system.matrixValues['M'][i], ref[i % 16], places=16, msg="Spectral mass matrix entry %d" % i)

        system = fem1d.TripletSystem(ansatz)
        matrices = fem1d.WaveEquationStandardMatrices(1.0, 1.0, source)
        matrices = fem1d.WaveEquationLumpedMatrices(matrices)
        fem1d.computeSystemMatrices(system, ansatz, quadratureK, matrices)

        for i in range(system.matrixValues['M'].size):
            self.assertAlmostEqual(system.matrixValues['MHRZ'][i], ref[i % 16], places=14, msg="HRZ lumped mass matrix entry %d" % i)
            self.assertAlmostEqual(system.matrixValues['MRS'][i], ref[i % 16], places=14, msg="Row summed mass matrix entry %d" % i)


if __name__ == '__main__':
    unittest.main()

