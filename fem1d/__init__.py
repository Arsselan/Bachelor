
from .gll import *
from .lagrange import *
from .bspline import *
from .legendre import *

from .ansatz import *
from .quadrature import *
from .system import *
from .utilities import *
from .studies import *

from . import sources


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

    def getNodes(self):
        return np.linspace(self.left, self.right, self.nElements+1)


class Domain:
    def __init__(self, alphaFunc):
        self.alphaFunc = alphaFunc

    def alpha(self, globalPos):
        return self.alphaFunc(globalPos)

