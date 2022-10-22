import matplotlib.pyplot as plt
import numpy as np

from waves1d import *

path = "results/time_domain_study_extra"

dataSplineCON = np.genfromtxt(path + "/Spline_CON.txt")
dataSplineRS = np.genfromtxt(path + "/Spline_RS.txt")
#dataSplineHRZ = np.genfromtxt(path + "/Spline_HRZ.txt")

dataLagrangeCON = np.genfromtxt(path + "/Lagrange_spectral_CON.txt")
#dataLagrangeRS = np.genfromtxt(path + "/Lagrange_spectral_RS.txt")
dataLagrangeHRZ = np.genfromtxt(path + "/Lagrange_spectral_HRZ.txt")

dataInterSplineCON = np.genfromtxt(path + "/InterpolatorySpline_CON.txt")
dataInterSplineHRZ = np.genfromtxt(path + "/InterpolatorySpline_HRZ.txt")


plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams["figure.figsize"] = (12, 6)

figure, ax = plt.subplots(1, 2)
figure.tight_layout(pad=2.5)

# ax.set_xlim(grid.left, grid.right)
#ax[0].set_ylim(500, 4000)
#ax[1].set_ylim(1e-6, 1)


ax[0].semilogy(dataSplineCON[:, 0], dataSplineCON[:, 2], '-o', label="Splines CON")
ax[0].semilogy(dataSplineRS[:, 0], dataSplineRS[:, 2], '-o', label="Splines RS")
#ax[0].plot(dataSplineHRZ[:, 0], dataSplineHRZ[:, 2], label="Splines HRZ")

ax[0].semilogy(dataLagrangeCON[:, 0], dataLagrangeCON[:, 2], '-o', label="Lagrange CON")
#ax[0].plot(dataLagrangeRS[:, 0], dataLagrangeRS[:, 2], label="Lagrange RS")
ax[0].semilogy(dataLagrangeHRZ[:, 0], dataLagrangeHRZ[:, 2], '-o', label="Lagrange HRZ")

ax[0].semilogy(dataInterSplineCON[:, 0], dataInterSplineCON[:, 2], '-o', label="Interp. Spline CON")
ax[0].semilogy(dataInterSplineHRZ[:, 0], dataInterSplineHRZ[:, 2], '-o', label="Interp. Spline HRZ")


ax[1].semilogy(dataSplineCON[:, 0], dataSplineCON[:, 1], '-o', label="Splines CON")
ax[1].semilogy(dataSplineRS[:, 0], dataSplineRS[:, 1], '-o', label="Splines RS")
#ax[1].plot(dataSplineHRZ[:, 0], dataSplineHRZ[:, 1], label="Splines HRZ")

ax[1].semilogy(dataLagrangeCON[:, 0], dataLagrangeCON[:, 1], '-o', label="Lagrange CON")
#ax[1].plot(dataLagrangeRS[:, 0], dataLagrangeRS[:, 1], label="Lagrange RS")
ax[1].semilogy(dataLagrangeHRZ[:, 0], dataLagrangeHRZ[:, 1], '-o', label="Lagrange HRZ")

ax[1].semilogy(dataInterSplineCON[:, 0], dataInterSplineCON[:, 1], '-o', label="Interp. Spline CON")
ax[1].semilogy(dataInterSplineHRZ[:, 0], dataInterSplineHRZ[:, 1], '-o', label="Interp. Spline HRZ")

ax[0].set_title('Maximum eigenvalues')
ax[1].set_title('Time domain error')
ax[0].set_xlabel('ficticious domain size')
ax[1].set_xlabel('ficticious domain size')
ax[0].set_ylabel('maximum eigenvalue')
ax[1].set_ylabel('error in time domain')

ax[0].legend()
#ax[1].legend()

plt.savefig(path + '/comparison.pdf')

plt.show()

