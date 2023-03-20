import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

import sources
from waves1d import *
from fem1d.studies import *

config = StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0,

    # method
    ansatzType='Lagrange',
    # ansatzType = 'InterpolatorySpline',
    n=120,
    p=2,
    # ansatzType='Spline',
    continuity='p-1',
    mass='HRZ',

    depth=35,
    spectral=False,
    dual=False,
    stabilize=0,
    smartQuadrature=False,
    source=sources.NoSource()
)

L = config.right - 2*config.extra
tMax = L*2
nt = 1200*20
dt = tMax / nt

# create study
temp = config.extra
for i in range(0):
    config.extra = i / 10 * 1.2/120
    print("extra = %e" % config.extra)
    study = EigenvalueStudy(config)
config.extra = temp
study = EigenvalueStudy(config)

# compute critical time step size
w = study.computeLargestEigenvalueSparse()
critDeltaT = 2 / abs(w)
print("Critical time step size is %e" % critDeltaT)
print("Chosen time step size is %e" % dt)
dt = correctTimeStepSize(dt, tMax, critDeltaT)
print("Corrected time step size is %e" % dt)

# solve sparse

boundaries = [study.config.extra, study.config.right-study.config.extra]
normals = [-1, 1]
forces = [1, 1]
study.F = study.system.getReducedVector(study.system.F + createNeumannVector(study.system, boundaries, normals, forces))
print(study.F)

u0, u1 = sources.applyGaussianInitialConditions(study.ansatz, dt, -0.6, config.stabilize)
evalNodes = np.linspace(study.grid.left + config.extra, study.grid.right - config.extra, study.ansatz.nDof())
u, fullU, evalU, iMat = study.runCentralDifferenceMethod(dt, nt, u0*0, u1*0, evalNodes)


# Plot animation
def postProcess(animationSpeed=4):
    figure, ax = plt.subplots()
    ax.set_xlim(study.grid.left, study.grid.right)
    ax.set_ylim(-2, 2)

    ax.plot([config.left+config.extra, config.left+config.extra], [-0.1, 0.1], '--', label='left boundary')
    ax.plot([config.right-config.extra, config.right-config.extra], [-0.1, 0.1], '--', label='right boundary')

    line, = ax.plot(0, 0, label='control points')
    line.set_xdata(np.linspace(study.grid.left, study.grid.right, study.ansatz.nDof()))

    line2, = ax.plot(0, 0, label='numerical')
    line2.set_xdata(evalNodes)

    ax.legend()

    plt.rcParams['axes.titleweight'] = 'bold'
    title = 'Solution'
    plt.title(title)
    plt.xlabel('solution')
    plt.ylabel('x')

    def prepareFrame(i):
        step = int(round(i / tMax * nt))
        plt.title(title + " time %3.2e step %d" % (i, step))
        line.set_ydata(fullU[step])
        line2.set_ydata(evalU[step])
        return line,

    frames = np.linspace(0, tMax, round(tMax * 60 / animationSpeed))
    animation = anim.FuncAnimation(figure, func=prepareFrame, frames=frames, interval=1000 / 60, repeat=False)
    plt.show()


# postProcess(1)
