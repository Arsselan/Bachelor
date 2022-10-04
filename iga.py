import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg

from scipy.interpolate import BSpline


k = 4
n = 3

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

gaussPoints = np.polynomial.legendre.leggauss(k+1)
points = np.zeros(n*(k+1))
weights = np.zeros(n*(k+1))
for i in range(n):
    x1 = t[k+i];
    x2 = t[k+i+1];
    d = x2-x1;
    for j in range(k+1):
        points[i*(k+1)+j] = x1 + d * 0.5 * ( gaussPoints[0][j] + 1 )
        weights[i*(k+1)+j] = gaussPoints[1][j]


# evaluate
print("Evaluating M...", flush=True)
N = BSpline.design_matrix(points, t, k)

M = N.transpose() @ N

print("Evaluating K...", flush=True)
c = np.eye(len(t) - k - 1)
B = BSpline(t, c, k).derivative()(points)

K = B.transpose() @ B

w = scipy.linalg.eigvals(K, M.toarray())
w = np.sqrt(np.real(w))
w = np.sort(w)


# plot
print("Plotting...", flush=True)
#figure, ax = plt.subplots()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_xlim(left, right)
ax1.set_ylim(-0.5, 1.5)

if True:
    x = np.linspace(left, right, 50)
    shapes = BSpline.design_matrix(x, t, k)
    derivative = BSpline(t, c, k).derivative()(x)
    for i in range(n+k):
        ax1.plot(x, shapes[:,[i]].toarray())
        ax2.plot(x, derivative[:,[i]])

ax1.plot(t, np.zeros(t.size), '-o')
ax1.plot(t, np.ones(t.size), '-o')

ax1.plot(points, np.ones(points.size), 'x')


soundSpeed = 1#np.sqrt(E/rho);
freq = np.pi*np.linspace(0,n,n+1)*soundSpeed / length;

ax3.plot(w)
ax3.plot(freq,'--')

                          
plt.show()

