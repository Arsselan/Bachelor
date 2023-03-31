import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
from scipy.fftpack import fft

from context import fem1d

#ref = np.loadtxt("results/example_timedomain_impact/Lagrange_n=1000_p=2_RS.dat")
#ref = np.loadtxt("results/example_timedomain_impact_reference/reference3.dat")
#ref = np.loadtxt("results/example_timedomain_impact_reference/lagrange_n100000_p1_rs_reduced2.dat")
ref = np.loadtxt("results/example_timedomain_impact/Spline_n=1000_p=4_CON.dat")
#ref = np.loadtxt("results/example_timedomain_impact/Spline_n=1000_p=3_CON.dat")

ref = np.delete(ref, 0, 0)
ref = np.delete(ref, 0, 0)

dt = 2e-4
nt = 120000

print("Reference first / last time: %e / %e" % (ref[0, 0], ref[-1, 0]))


def computeScipySpectrum(u):
    uf = fft(u)
    ff = np.linspace(0.0, 1.0 / (2.0 * dt), nt // 2)
    plt.semilogy(ff, 2.0 / nt * np.abs(uf[0:nt // 2]))


def computeErrors(ansatzType, p, nValues):
    dofs = []
    errors = []
    errorOverTimeReturned = []
    for n in nValues:
        title = ansatzType + "_n=%d" % n + "_p=%d" % p + "_" + "RS.dat"
        fileName = "results/example_timedomain_impact/" + title
        data = np.loadtxt(fileName)
        data = np.delete(data, 0, 0)
        data = np.delete(data, 0, 0)
        print("Data first / last time: %e / %e" % (data[0, 0], data[-1, 0]))

        errorOverTime = (ref[:, 1] - data[:, 1])
        error = np.linalg.norm(dt*errorOverTime)
        errors.append(error)
        if ansatzType == "Lagrange":
            dof = p*n+1
        elif ansatzType == "Spline":
            dof = n + p
        dofs.append(dof)
        if (n == 200 and p == 3) or (n == 200 and p == 2):
            fem1d.writeColumnFile(fileName + "ot", [ref[:, 0], errorOverTime])
            print("STORING!")
        print(title + "Dof: %d, Error: %e" % (dof, error))
    return dofs, errors, errorOverTimeReturned


def computeConvergence():
    dofsLagrange2, errorLagrange2, errorOverTimeLagrange2 = computeErrors("Lagrange", 2, [12, 25, 50, 100, 200, 400, 800])
    dofsSpline2, errorSpline2, errorOverTimeSpline2 = computeErrors("Spline", 2, [25, 50, 100, 200, 400, 800, 1600])

    dofsLagrange3, errorLagrange3, errorOverTimeLagrange3 = computeErrors("Lagrange", 3, [6, 12, 25, 50, 100, 200, 400])
    dofsSpline3, errorSpline3, errorOverTimeSpline3 = computeErrors("Spline", 3, [25, 50, 100, 200, 400, 800, 1600])

    plt.loglog(dofsLagrange2, errorLagrange2, "-o", label="Lagrange, p=2")
    plt.loglog(dofsSpline2, errorSpline2, "-o", label="Spline, p=2")

    plt.loglog(dofsLagrange3, errorLagrange3, "-o", label="Lagrange, p=3")
    plt.loglog(dofsSpline3, errorSpline3, "-o", label="Spline, p=3")

    fem1d.writeColumnFile("results/example_timedomain_impact/convergence_lagrange_p2.dat", [dofsLagrange2, errorLagrange2])
    fem1d.writeColumnFile("results/example_timedomain_impact/convergence_lagrange_p3.dat", [dofsLagrange3, errorLagrange3])
    fem1d.writeColumnFile("results/example_timedomain_impact/convergence_splines_p2.dat", [dofsSpline2, errorSpline2])
    fem1d.writeColumnFile("results/example_timedomain_impact/convergence_splines_p3.dat", [dofsSpline3, errorSpline3])

    plt.legend()

    plt.show()


def plotComparison(ansatzType1, ansatzType2, p1, p2, n1, n2):
    title1 = ansatzType1 + "_n=%d" % n1 + "_p=%d" % p1 + "_" + "RS.dat"
    fileName1 = "results/example_timedomain_impact/" + title1
    data1 = np.loadtxt(fileName1)
    title2 = ansatzType2 + "_n=%d" % n2 + "_p=%d" % p2 + "_" + "RS.dat"
    fileName1 = "results/example_timedomain_impact/" + title2
    data2 = np.loadtxt(fileName1)
    plt.plot(data1[:, 0], data1[:, 1], label=title1)
    plt.plot(data2[:, 0], data2[:, 1], label=title2)
    plt.legend()
    plt.show()

