import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from waves1d import *

def salpha(x):
    if x>=0.5 and x<=4.5:
        return 1.0
    return 0
    
d = Domain(salpha)

g = UniformGrid(0, 5, 10)

#a = SplineAnsatz(g, 3, 2)
gllPoints = GLL(4)
a = LagrangeAnsatz(g, gllPoints[0])

gaussPoints = np.polynomial.legendre.leggauss(a.p+1)
    
q = SpaceTreeQuadrature(g, gaussPoints, d, 2)

system = TripletSystem(a, q, True)

figure, ax = plt.subplots()
ax.set_ylim(-0.5, 1)


if 1:
    ax.plot(a.knots, a.knots * 0, '-o')

    cutY = 0.1
    for iElement in range(g.nElements):
        ax.plot(q.points[iElement], q.weights[iElement],'x')
        for cutX in q.cuts[iElement]:
            ax.plot([cutX, cutX], [-cutY, cutY],'-')

        nX = 51
        xx = np.linspace(g.pos(iElement,-1), g.pos(iElement,1-1e-14), nX)
        yy = np.zeros((nX, a.p+1))
        for i in range(nX):
            yy[i] = a.evaluate(xx[i], 0)[0]
        for i in range(a.p+1):
            ax.plot(xx, yy[:,i])

    plt.show()

