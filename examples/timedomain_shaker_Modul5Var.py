import numpy as np
import shutil
import os
import sys
from context import fem1d
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

#bester Wert bei nFrequencies = 2 und firstFrequency = 7
dataEx = np.loadtxt("examples/shaker_experiments_2.dat")
#firstFrequency und nFrequencies verändern 
nFrequencies = 11
firstFrequency = 1
lastFrequency = firstFrequency + nFrequencies
#letze Frequenz muss kleiner 12 sein
#Summe = 6
# firstFrequency muss größer als nFrequencies sein
def getOutputDir(params):
    outputDir = "shaker_elas%e_damp" % params[0]
    for i in range(len(params)):
        if i > 0:
            outputDir += "_%e" % params[i]
    return outputDir

def runStudy(params, disablePlots=True):
    outputDir = getOutputDir(params)
    print("\n Dir: %s\n" % outputDir)
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    data = np.ndarray((nFrequencies, 4))
    for i in range(nFrequencies):
        #frequency = 100 + (i + firstFrequency) * 50
        frequency = dataEx[i + firstFrequency, 0]
        print("Frequency: %e" % frequency)
        import examples.timedomain_shaker
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        extraDelta = 0.0
        examples.timedomain_shaker.run(extraDelta, params, frequency, disablePlots)
        sys.stdout = save_stdout

def plotBoth(dataSim):
    figure, ax = plt.subplots()
    ptx = dataEx[:, 0]
    pty = dataEx[:, 1]
    ax.plot(ptx, pty, "-o", label="Speichermodul (Experiment)")
    ptx = dataEx[:, 0]
    pty = dataEx[:, 2]
    ax.plot(ptx, pty, "-o", label="Verlustmodul (Experiment)")
    ptx = dataSim[:, 0]
    pty = dataSim[:, 1]
    ax.plot(ptx, pty, "-o", label="Speichermodul (Simulation)")
    ptx = dataSim[:, 0]
    pty = dataSim[:, 2]
    ax.plot(ptx, pty, "-o", label="Verlustmodul (Simulation)")
    figure.set_size_inches(10, 6)
    plt.legend(fontsize=20)
    plt.xlabel("Frequenz", fontsize=22)
    plt.ylabel("Speichermodul/Verlustmodul", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.savefig('Haupt_Fre_Scipy3.pdf', format='pdf')
    plt.show()

def readData(params, path="."):
    outputDir = path+"/"+getOutputDir(params)
    return np.loadtxt(outputDir + "/shaker_freq_storage_loss_delta_left.dat")

def computeError(dataSim, returnAllError= False):
    errorStorage = np.linalg.norm(dataEx[firstFrequency:lastFrequency, 1] - dataSim[:, 1])
    errorStorage /= np.linalg.norm(dataEx[firstFrequency:lastFrequency, 1])
    errorLoss = np.linalg.norm( dataEx[firstFrequency:lastFrequency, 2] - dataSim[:, 2] )
    errorLoss /= np.linalg.norm(dataEx[firstFrequency:lastFrequency, 2])
    if returnAllError:
        return [errorStorage + errorLoss, errorLoss, errorStorage]
    return errorStorage + errorLoss
    #return errorLoss
    #return errorStorage

def objectiveFunction(params):
    runStudy(params)
    data = readData(params)
    error = computeError(data)
    print("\n ----> Error: %e \n\n" % error)
    #plotBoth(data) # @Arsselan: Comment out for production
    return error

def write_to_file(filename, iteration_count, objective_value, params):
    with open(filename, "a") as file:
        file.write(f"{iteration_count} {objective_value:.6e}")
        for para in params:
            file.write(f" {para:.6e}")
        file.write("\n")


def objective_function_scipy(u):

    # @Arsselan: Change typical parameter factors to end up in good ranges with all 
    # paramters entering this function (u) being of the same 
    umod = list(u)
    umod[0] *= 1e5
    umod[1] *= 1
    umod[2] *= 1e-5
    umod[3] *= 1e-12
    ulist = list(u)
    objective_function_scipy.count += 1
    print("New iteration: %d" % objective_function_scipy.count, "u: ", u)
    objective = objectiveFunction(np.array(umod))
    if objective < objective_function_scipy.best:
        objective_function_scipy.best = objective
#    else:                  # @Arsselan: Comment in for less disc usage
#        removeDir(u)
    write_to_file(objective_function_scipy.filename, objective_function_scipy.count, objective, u)
    
    return objective

objective_function_scipy.count = 0
objective_function_scipy.best = 1e10
objective_function_scipy.filename = ""

def doScipyOptimize():
    from scipy import optimize
    initialGuess = np.array([2.0, 0.01, 0.01, 0.01])
    objective_function_scipy.filename = f"Ziteration_SO_storage_{initialGuess[0]:.2f}_{initialGuess[1]:.2f}_{initialGuess[2]:.2f}_{initialGuess[3]:.2f}_nFrequenz={nFrequencies}_firstFrequenz={firstFrequency}.dat"
    optimize.minimize(objective_function_scipy,
                      initialGuess,
                      #tol=0,
                      method='L-BFGS-B',
                      bounds=[(0.01, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                      options={'eps': 1e-1, 'maxiter': 150})
    #return result.fun
def objective_function_gfo(para):
    u = [para["E"], para["D1"], para["D2"], para["D3"]]
    return -objective_function_scipy(u)
# 4.0, 0.3, 0.2, 0.2
def doParticleSwarm():
    from gradient_free_optimizers import ParticleSwarmOptimizer
    E1, E2, E3 = 2.0, 2.1, 0.01 
    D1_1, D1_2, D1_3 = 0.01, 0.02, 0.001
    D2_1, D2_2, D2_3 = 0.01, 0.02, 0.001 
    D3_1, D3_2, D3_3 = 0.01, 0.02, 0.001 
    search_space = {
        "E": np.arange(E1, E2, E3),
        "D1": np.arange(D1_1, D1_2, D1_3),
        "D2": np.arange(D2_1, D2_2, D2_3),
        "D3": np.arange(D3_1, D3_2, D3_3)
    }
    opt = ParticleSwarmOptimizer(search_space, population=100)
    objective_function_scipy.filename = f"Ziteration_PS_storage_E{E1}_to_{E2}_with_{E3}_D1_{D1_1}_to_{D1_2}_with_{D1_3}_D2_{D2_1}_to_{D2_2}_with_{D2_3}_D3_{D3_1}_to_{D3_2}_with_{D3_3}_nFrequenz={nFrequencies}_firstFrequenz={firstFrequency}_pop=150.dat"
    opt.search(objective_function_gfo, n_iter=150)

def tangent(function, x, eps=1e3):
    f = function(x)
    n = x.shape[0]
    t = np.ndarray(n)
    for i in range(n):
        iEps = eps * abs(x[i])
        xp = x * 1.0
        xp[i] += iEps
        deltaF = function(xp) - f
        while abs(deltaF) < 1e-10:
            print("Warning! Delta too small (deltaF=%e, iEps=%e)." % (deltaF, iEps))
            iEps *= 10.0
            xp = x * 1.0
            xp[i] += iEps
            deltaF = function(xp) - f
        t[i] = deltaF / iEps
    return f, t

def gradientDescent(function, initial):
    x = initial
    for i in range(100):
        alpha = 0.01
        f, t = tangent(function, x, 1e-1)
        print("Objective: %e" % f)
        print("Tangent: ", t)
        dx = - alpha / np.linalg.norm(t) * t
        print("Delta x: ", dx)
        x += dx
        while function(x) > f:
            x -= dx
            alpha *= 0.5
            print("Warning! Objective not reduced. Setting alpha=%e." % alpha)
            dx = - alpha / np.linalg.norm(t) * t
            print("Delta x: ", dx)
            x += dx
    return x

def doGradientDescent():
    #initial = np.array([3.954796, 0.3999992, 0.2338214, 0.2363156])
    initial = np.array([2.0, 0.009999849, 0.01877384, 0.05303151])
    filename = f"iteration_GD_{initial[0]:.2f}_{initial[1]:.2f}_{initial[2]:.2f}_{initial[3]:.2f}.dat"
    objective_function_scipy.filename = filename

    x = gradientDescent(objective_function_scipy, initial)


def Plot_Frequenz():
    filename = "Z_Überprüfung2Neu\iteration_5VAR_SO_4.00_0.30_0.30_0.30_0.30_nFrequenz=11_firstFrequenz=1.dat"
    row = 35
    data = np.loadtxt(filename)
    param = data[row,2:]
    print("Parameters: ", param)
    #param = [3.851959e+00, 9.992388e-03, 3.827240e-01, 2.774879e-01, 0.000000e+00]
    param[0] *= 1e5
    param[1] *= 1
    param[2] *= 1e-5
    param[3] *= 1e-12
    param[4] *= 1e-23
    data = readData(param,"C:/Users/arsse/OneDrive/Desktop/iga-stuff/AlteErgebnisse")
    print(computeError(data, True))
    plotBoth(data)

    