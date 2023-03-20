import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from waves1d import *

from fem1d.studies import *

config = StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0.0,

    # method
    # ansatzType = 'Lagrange'
    # ansatzType = 'InterpolatorySpline'
    n=12,
    p=3,
    ansatzType='Spline',
    continuity='p-1',
    mass='CON',

    depth=40,
    spectral=False,
    dual=False,
    stabilize=0,
  )

axLimitY = 500
# axLimitY = 50


# extra values
ne = 101
extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
    0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]
ne = len(extras)

extras = list(np.array(extras))

# prepare result data
maxP = 4
res = np.zeros((ne, maxP + 1))
res[:, 0] = extras

# run studies
for p in range(1, maxP + 1):
    maxw = [0] * ne
    print("p = %d" % p)
    for i in range(ne):
        config.extra = extras[i]
        config.p = p
        study = EigenvalueStudy(config)
        #maxw[i] = study.runDense()
        #maxw[i] = study.runSparse()
        maxw[i] = study.computeLargestEigenvalueSparse()

        print("e = %e, wmax = %e" % (extras[i], maxw[i]))
    res[:, p] = maxw

# save result
title = config.ansatzType + ' C' + str(config.continuity) + ' ' + config.mass + ' ' + str(config.stabilize)
fileBaseName = getFileBaseNameAndCreateDir("results/example_eigenvalue_study/", title.replace(' ', '_'))
np.savetxt(fileBaseName + '.dat', res)


# plot
figure, ax = plt.subplots()
# ax.set_ylim(5, axLimitY)

for p in range(1, maxP + 1):
    ax.plot(extras, res[:,p], '-o', label='p=' + str(p))

ax.legend()
plt.xlabel('ficticious domain size')
plt.ylabel('largest eigenvalue')

plt.rcParams['axes.titleweight'] = 'bold'

plt.title(title)

plt.savefig(fileBaseName + '.pdf')
plt.show()
