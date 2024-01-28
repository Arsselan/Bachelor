import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse
import scipy.sparse.linalg
from scipy.fftpack import fft

from context import fem1d

outputDir = "results/timedomain_impact_plastic_study_xxx/"

config = fem1d.StudyConfig(
    # problem
    left=0,
    right=1.0,
    extra=0,

    # method
    ansatzType='Lagrange',
    #ansatzType='Spline',
    #ansatzType='InterpolatorySpline',
    n=100,
    p=2,

    continuity='p-1',
    mass='RS',

    depth=50,
    spectral=False,
    dual=False,
    stabilize=0,
    smartQuadrature=True,
    source=fem1d.sources.NoSource(),
    fixedDof=[0]
)

compute = True
finalDisp = True

tMax = 0.1
nt = 20000*2
dt = tMax / nt

# load reference
if not compute:
    if finalDisp:
        ref = np.loadtxt(outputDir + "/Spline_n800_p2_CON_dt5.000000e-07_final_disp.dat")
    else:
        ref = np.loadtxt(outputDir + "/Spline_n800_p2_CON_dt5.000000e-07.dat")
        ref = ref[1::5, :]


def computeError(config):
    p = config.p
    config.n = config.n * (eval(config.continuity) + 1)

    if finalDisp:
        title2 = outputDir + "/" + config.ansatzType + "_n%d" % config.n + "_p%d" % config.p + "_" + config.mass + "_dt%e" % dt + "_final_disp.dat"
        disp = np.loadtxt(title2)
        err = ref[:, 1] - disp[:, 1]
        err = np.linalg.norm(err) / np.linalg.norm(ref[:, 1])
    else:
        title2 = outputDir + "/" + config.ansatzType + "_n%d" % config.n + "_p%d" % config.p + "_" + config.mass + "_dt%e" % dt + ".dat"
        disp = np.loadtxt(title2)
        err = ref[:, 2] - disp[1:, 2]
        err = np.linalg.norm(err) / np.linalg.norm(ref[:, 2])

    grid = fem1d.UniformGrid(config.left, config.right, config.n)
    ansatz = fem1d.createAnsatz(config.ansatzType, config.continuity, config.p, grid)
    nDof = ansatz.nDof()
    print("Error: ", err)
    return nDof, err


def computeErrors():
    for ansatzType in ["Spline", "Lagrange"]:
        config.ansatzType = ansatzType
        if ansatzType == "Spline":
            config.continuity = 'p-1'
        else:
            config.continuity = "0"

        figure, ax = plt.subplots()
        masses = ["CON", "RS"]
        if ansatzType == "Lagrange":
            masses = ["CON", "HRZ"]

        for mass in masses:
            for p in [1, 2, 3]:
                print("p=%d" % p)
                data = np.zeros((4, 2))
                i = 0
                for n in [12, 24, 48, 96]:
                    config.n = int(n / p)
                    config.p = p
                    config.mass = mass
                    nDof, err = computeError(config)
                    data[i, 0] = nDof
                    data[i, 1] = err
                    i += 1
                ax.loglog(data[:, 0], data[:, 1], "-*", label=config.ansatzType + " " + mass + " p=" + str(p))
                filename = fem1d.getFileBaseNameAndCreateDir(
                        outputDir + "/", "convergence_" + config.ansatzType + "_" + config.mass + "_p" + str(config.p) + ".dat")
                if finalDisp:
                    filename = fem1d.getFileBaseNameAndCreateDir(
                        outputDir + "/",
                        "convergence_" + config.ansatzType + "_" + config.mass + "_p" + str(config.p) + "_final_disp.dat")
                print("Saving " + filename)
                np.savetxt(filename, data)
        plt.legend()
        plt.show()


if compute:
    for ansatzType in ["Spline", "Lagrange"]:
        config.ansatzType = ansatzType
        if ansatzType == "Spline":
            config.continuity = 'p-1'
        else:
            config.continuity = "0"

        for mass in ["CON", "RS"]:
#            for p in [1, 2, 3]:
            for p in [3]:
#                for n in [12, 24, 48, 96]:
                for n in [24]:
                    print("\n\n", ansatzType, " ", mass, " ", p, " ", n, "\n\n")
                    config.n = int(n / p)
                    config.p = p
                    config.mass = mass
                    exec(open("examples/timedomain_impact_plastic.py").read())
