import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
import bspline
from scipy.interpolate import BSpline


k = 3
n = 12

left = 0
right = 1.2

# create mesh
print("Meshing...", flush=True)

length = right - left
extra = length / n * k
t = np.linspace(left-extra, right+extra, n+1+2*k)
for i in range(k+1):
    t[i] = left
    t[-i-1] = right


#t[5] = t[6]
#t = np.insert(t, 5, t[5])
#t = np.insert(t, 5, t[5])
#t = np.insert(t, 4, t[4])
#t = np.insert(t, 4, t[4])
#n=n+4

# plot
print("Plotting...", flush=True)
fig, (ax1) = plt.subplots()

ax1.set_xlim(left, right)
ax1.set_ylim(-0.3, 1.1)

x = np.linspace(left, right, 500)
shapes = BSpline.design_matrix(x, t, k)
c = np.eye(len(t) - k - 1)
derivative = BSpline(t, c, k).derivative()(x)
for i in range(n+k):
    ax1.plot(x, shapes[:,[i]].toarray())

for i in range(len(t)-2*k-1):
    x1 = t[k+i]
    x2 = t[k+i+1]
    xx = np.linspace(x1, x2, 11)
    yy = np.zeros((11,k+1))
    dy = np.zeros((11,k+1))
    for j in range(len(xx)):
        ders = bspline.evaluateBSplineBases(k+i, xx[j], k, 1, t)
        yy[j] = ders[0]
        dy[j] = ders[1]
    for j in range(k+1):
        ax1.plot(xx,yy[:,j],'x')

ax1.plot(t, np.zeros(t.size), '-o')

                          
plt.show()

