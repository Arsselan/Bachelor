import unittest

import numpy as np
from context import fem1d


def createBaseConfig():
    config = fem1d.StudyConfig(
        # problem
        left=0,
        right=1.2,
        # extra=0.8*1.2/120,
        extra=0.01495,

        # method
        ansatzType='Lagrange',
        # ansatzType='Spline',
        # ansatzType = 'InterpolatorySpline',
        n=80,
        p=3,
        continuity='p-1',
        mass='CON',

        depth=15,
        spectral=False,
        dual=False,
        stabilize=0,
        smartQuadrature=True,
        source=fem1d.sources.NoSource()
    )
    config.density = 1.0
    config.elasticity = 1.0
    return config 


class TestStudy(unittest.TestCase):
    def test_study(self):
        config = createBaseConfig()
        config.ansatzType = "Lagrange"
        config.mass = "CON"
        study = fem1d.EigenvalueStudy(config)
        w = study.computeLargestEigenvalueSparse()
        self.assertAlmostEqual(w, 209198.5764866068, places=13, msg="Largest eigenvalue lagrange consistent")

        config = createBaseConfig()
        config.ansatzType = "Spline"
        config.mass = "CON"
        study = fem1d.EigenvalueStudy(config)
        w = study.computeLargestEigenvalueSparse()
        self.assertAlmostEqual(w, 209041.52086941496, places=13, msg="Largest eigenvalue spline consistent")

        config = createBaseConfig()
        config.ansatzType = "Lagrange"
        config.mass = "HRZ"
        study = fem1d.EigenvalueStudy(config)
        w = study.computeLargestEigenvalueSparse()
        self.assertAlmostEqual(w, 59213.97071297006, places=13, msg="Largest eigenvalue lagrange lumped")

        config = createBaseConfig()
        config.ansatzType = "Lagrange"
        config.mass = "RS"
        study = fem1d.EigenvalueStudy(config)
        w = study.computeLargestEigenvalueSparse()
        self.assertAlmostEqual(w, 3985.982861715098, places=15, msg="Largest eigenvalue spline lumped")


if __name__ == '__main__':
    unittest.main()

