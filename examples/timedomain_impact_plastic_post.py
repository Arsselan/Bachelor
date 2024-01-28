import numpy as np
import matplotlib.pyplot as plt


# p=2
def post_p2_n100_dt5():
    dispLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n100_p2_RS_dt5.000000e-06.dat")
    dispLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n100_p2_CON_dt5.000000e-06.dat")
    dispSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n200_p2_RS_dt5.000000e-06.dat")
    dispSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n200_p2_CON_dt5.000000e-06.dat")

    plt.plot(dispLagrangeCON[:, 0], dispLagrangeCON[:, 2], "-", label="Lagrange, consistent, p=2")
    plt.plot(dispLagrangeRS[:, 0], dispLagrangeRS[:, 2], "-", label="Lagrange, lumped, p=2")
    plt.plot(dispSplineCON[:, 0], dispSplineCON[:, 2], "-", label="Splines, consistent, p=2")
    plt.plot(dispSplineRS[:, 0], dispSplineRS[:, 2], "-", label="Splines, lumped, p=2")

    plt.title("displacement of the right end")
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()

    plt.show()

    epsLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n100_p2_RS_dt5.000000e-06_eps.dat")
    epsLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n100_p2_CON_dt5.000000e-06_eps.dat")
    epsSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n200_p2_RS_dt5.000000e-06_eps.dat")
    epsSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n200_p2_CON_dt5.000000e-06_eps.dat")

    plt.plot(epsLagrangeCON[:, 0], epsLagrangeCON[:, 1], ".", label="Lagrange, consistent, p=2")
    plt.plot(epsLagrangeRS[:, 0], epsLagrangeRS[:, 1], ".", label="Lagrange, lumped, p=2")
    plt.plot(epsSplineCON[:, 0], epsSplineCON[:, 1], ".", label="Splines, consistent, p=2")
    plt.plot(epsSplineRS[:, 0], epsSplineRS[:, 1], ".", label="Splines, lumped, p=2")

    plt.title("plastic strain at quadrature points")
    plt.xlabel("position")
    plt.ylabel("plastic strain")
    plt.legend()

    plt.show()


# p=2
def post_p2_n50_dt5():
    dispLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n50_p2_RS_dt5.000000e-06.dat")
    dispLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n50_p2_CON_dt5.000000e-06.dat")
    dispSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n100_p2_RS_dt5.000000e-06.dat")
    dispSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n100_p2_CON_dt5.000000e-06.dat")

    plt.plot(dispLagrangeCON[:, 0], dispLagrangeCON[:, 2], "-", label="Lagrange, consistent, p=2")
    plt.plot(dispLagrangeRS[:, 0], dispLagrangeRS[:, 2], "-", label="Lagrange, lumped, p=2")
    plt.plot(dispSplineCON[:, 0], dispSplineCON[:, 2], "-", label="Splines, consistent, p=2")
    plt.plot(dispSplineRS[:, 0], dispSplineRS[:, 2], "-", label="Splines, lumped, p=2")

    plt.title("displacement of the right end")
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()

    plt.show()

    epsLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n50_p2_RS_dt5.000000e-06_eps.dat")
    epsLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n50_p2_CON_dt5.000000e-06_eps.dat")
    epsSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n100_p2_RS_dt5.000000e-06_eps.dat")
    epsSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n100_p2_CON_dt5.000000e-06_eps.dat")

    plt.plot(epsLagrangeCON[:, 0], epsLagrangeCON[:, 1], ".", label="Lagrange, consistent, p=2")
    plt.plot(epsLagrangeRS[:, 0], epsLagrangeRS[:, 1], ".", label="Lagrange, lumped, p=2")
    plt.plot(epsSplineCON[:, 0], epsSplineCON[:, 1], ".", label="Splines, consistent, p=2")
    plt.plot(epsSplineRS[:, 0], epsSplineRS[:, 1], ".", label="Splines, lumped, p=2")

    plt.title("plastic strain at quadrature points")
    plt.xlabel("position")
    plt.ylabel("plastic strain")
    plt.legend()

    plt.show()


# p=2
def post_p2_dt5():
    dispLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p2_RS_dt5.000000e-06.dat")
    dispLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p2_CON_dt5.000000e-06.dat")
    dispSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n50_p2_RS_dt5.000000e-06.dat")
    dispSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n50_p2_CON_dt5.000000e-06.dat")

    plt.plot(dispLagrangeCON[:, 0], dispLagrangeCON[:, 2], "-", label="Lagrange, consistent, p=2")
    plt.plot(dispLagrangeRS[:, 0], dispLagrangeRS[:, 2], "-", label="Lagrange, lumped, p=2")
    plt.plot(dispSplineCON[:, 0], dispSplineCON[:, 2], "-", label="Splines, consistent, p=2")
    plt.plot(dispSplineRS[:, 0], dispSplineRS[:, 2], "-", label="Splines, lumped, p=2")

    plt.title("displacement of the right end")
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()

    plt.show()

    epsLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p2_RS_dt5.000000e-06_eps.dat")
    epsLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p2_CON_dt5.000000e-06_eps.dat")
    epsSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n50_p2_RS_dt5.000000e-06_eps.dat")
    epsSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n50_p2_CON_dt5.000000e-06_eps.dat")

    plt.plot(epsLagrangeCON[:, 0], epsLagrangeCON[:, 1], ".", label="Lagrange, consistent, p=2")
    plt.plot(epsLagrangeRS[:, 0], epsLagrangeRS[:, 1], ".", label="Lagrange, lumped, p=2")
    plt.plot(epsSplineCON[:, 0], epsSplineCON[:, 1], ".", label="Splines, consistent, p=2")
    plt.plot(epsSplineRS[:, 0], epsSplineRS[:, 1], ".", label="Splines, lumped, p=2")

    plt.title("plastic strain at quadrature points")
    plt.xlabel("position")
    plt.ylabel("plastic strain")
    plt.legend()

    plt.show()


# p=3
def post_p3_dt5():
    dispLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_RS_dt5.000000e-06.dat")
    dispLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_CON_dt5.000000e-06.dat")
    dispSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_RS_dt5.000000e-06.dat")
    dispSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_CON_dt3.493084e-06.dat")

    plt.plot(dispLagrangeCON[:, 0], dispLagrangeCON[:, 2], "-", label="Lagrange, consistent, p=3")
    plt.plot(dispLagrangeRS[:, 0], dispLagrangeRS[:, 2], "-", label="Lagrange, lumped, p=3")
    plt.plot(dispSplineCON[:, 0], dispSplineCON[:, 2], "-", label="Splines, consistent, p=3")
    plt.plot(dispSplineRS[:, 0], dispSplineRS[:, 2], "-", label="Splines, lumped, p=3")

    plt.title("displacement of the right end")
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()

    plt.show()

    epsLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_RS_dt5.000000e-06_eps.dat")
    epsLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_CON_dt5.000000e-06_eps.dat")
    epsSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_RS_dt5.000000e-06_eps.dat")
    epsSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_CON_dt3.493084e-06_eps.dat")

    plt.plot(epsLagrangeCON[:, 0], epsLagrangeCON[:, 1], ".", label="Lagrange, consistent, p=3")
    plt.plot(epsLagrangeRS[:, 0], epsLagrangeRS[:, 1], ".", label="Lagrange, lumped, p=3")
    plt.plot(epsSplineCON[:, 0], epsSplineCON[:, 1], ".", label="Splines, consistent, p=3")
    plt.plot(epsSplineRS[:, 0], epsSplineRS[:, 1], ".", label="Splines, lumped, p=3")

    plt.title("plastic strain at quadrature points")
    plt.xlabel("position")
    plt.ylabel("plastic strain")

    plt.legend()

    plt.show()


# p=3
def post_p3_dt25():
    dispLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_RS_dt2.500000e-06.dat")
    dispLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_CON_dt2.500000e-06.dat")
    dispSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_RS_dt2.500000e-06.dat")
    dispSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_CON_dt2.500000e-06.dat")

    plt.plot(dispLagrangeCON[:, 0], dispLagrangeCON[:, 2], "-", label="Lagrange, consistent, p=3")
    plt.plot(dispLagrangeRS[:, 0], dispLagrangeRS[:, 2], "-", label="Lagrange, lumped, p=3")
    plt.plot(dispSplineCON[:, 0], dispSplineCON[:, 2], "-", label="Splines, consistent, p=3")
    plt.plot(dispSplineRS[:, 0], dispSplineRS[:, 2], "-", label="Splines, lumped, p=3")

    plt.title("displacement of the right end")
    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.legend()

    plt.show()

    epsLagrangeRS = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_RS_dt2.500000e-06_eps.dat")
    epsLagrangeCON = np.loadtxt("results/crash_test_plastic/Lagrange_n25_p3_CON_dt2.500000e-06_eps.dat")
    epsSplineRS = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_RS_dt2.500000e-06_eps.dat")
    epsSplineCON = np.loadtxt("results/crash_test_plastic/Spline_n75_p3_CON_dt2.500000e-06_eps.dat")

    plt.plot(epsLagrangeCON[:, 0], epsLagrangeCON[:, 1], ".", label="Lagrange, consistent, p=3")
    plt.plot(epsLagrangeRS[:, 0], epsLagrangeRS[:, 1], ".", label="Lagrange, lumped, p=3")
    plt.plot(epsSplineCON[:, 0], epsSplineCON[:, 1], ".", label="Splines, consistent, p=3")
    plt.plot(epsSplineRS[:, 0], epsSplineRS[:, 1], ".", label="Splines, lumped, p=3")

    plt.title("plastic strain at quadrature points")
    plt.xlabel("position")
    plt.ylabel("plastic strain")

    plt.legend()

    plt.show()

