import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

from context import fem1d

config = fem1d.StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0.1,

    # method
    ansatzType='Spline',
    n=24,
    mass='RS',

    #ansatzType='InterpolatorySpline',
    #ansatzType='Lagrange',
    #n=12,
    #mass='HRZ',

    p=2,

    continuity='1',
    #mass='CON',

    depth=15,
    stabilize=1e-8,
    spectral=False,
    dual=False,
    smartQuadrature=False,

    source=fem1d.sources.NoSource()
)

# study
eigenvalue = 6

# title
title = config.ansatzType
title += ' ' + config.continuity
title += ' ' + config.mass
title += ' a=%2.1e' % config.stabilize
title += ' d=' + str(config.extra)

fileBaseName = fem1d.getFileBaseNameAndCreateDir("results/example_spectrum/", title.replace(' ', '_'))

study = fem1d.EigenvalueStudy(config)

study.runDense(False, True)

nw = study.w.size

indices = np.linspace(0, nw-1, nw)

wExact = (indices * np.pi) / (1.2 - 2 * config.extra)

w = []
for i in range(nw):
    wi, idx = fem1d.findEigenvalue(study.w, "nearest", i, wExact[i])
    w.append(wi)

fem1d.plot(indices, [wExact, w, study.w], ["exact", "nearest", "number"])

fem1d.writeColumnFile(fileBaseName + ".dat", [indices, wExact, w, study.w])

