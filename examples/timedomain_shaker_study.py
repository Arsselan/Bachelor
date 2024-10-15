import numpy as np
import shutil
import os
import sys

from context import fem1d

dataEx = np.loadtxt("examples/shaker_experiments_2.dat")

nFrequencies = 12
firstFrequency = 0
lastFrequency = firstFrequency + nFrequencies


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


def plotStudy(data):
    fem1d.plot(data[:, 0], [data[:, 1], data[:, 2], data[:, 3]], ["storage", "loss", "delta"])


def plotReference():
    fem1d.plot(dataEx[:, 0], [
               dataEx[:, 1], dataEx[:, 2]*10,
               dataEx[:, 3], dataEx[:, 4]*10,
               dataEx[:, 5], dataEx[:, 6]*10],
               ["150 storage", "150 loss$\cdot$10","77 storage", "77 loss$\cdot$10","nano storage", "nano loss$\cdot$10"])


def readData(params):
    outputDir = getOutputDir(params)
    return np.loadtxt(outputDir + "/shaker_freq_storage_loss_delta_left.dat")


def plotBoth(dataSim):
    fem1d.plot(dataEx[firstFrequency:lastFrequency, 0], [
               dataEx[firstFrequency:lastFrequency, 1], dataEx[firstFrequency:lastFrequency, 2]*10,
               dataSim[:, 1], dataSim[:, 2]*10 ],
               ["150 storage", "150 loss$\cdot$10",
                "sim storage", "sim loss$\cdot$10"])

def computeError(dataSim):
    errorStorage = np.linalg.norm(dataEx[firstFrequency:lastFrequency, 1] - dataSim[:, 1])
    errorStorage /= np.linalg.norm(dataEx[firstFrequency:lastFrequency, 1])
    errorLoss = np.linalg.norm( dataEx[firstFrequency:lastFrequency, 2] - dataSim[:, 2] )
    errorLoss /= np.linalg.norm(dataEx[firstFrequency:lastFrequency, 2])
    return errorStorage + errorLoss
    #return errorLoss
    #return errorStorage

def objectiveFunction(params):
    runStudy(params)
    data = readData(params)
    error = computeError( data )
    print("\n ----> Error: %e \n\n" % error)
    plotBoth(data)  # @Arsselan: Comment out for production
    return error


def removeDir(params):
    outputDir = getOutputDir(params)
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)


def check(params, run=True, disablePlots=False):
    if run:
        runStudy(params, disablePlots)
    data = readData(params)
    error = computeError(data)
    print("Error: %e" % error)
    plotBoth(data)













def objective_function_scipy(u):


    # @Arsselan: Change typical parameter factors to end up in good ranges with all 
    # paramters entering this function (u) being of the same order of magnitude
    umod = list(u)
    umod[0] *= 1e5
    umod[1] *= 1
    umod[2] *= 1e-5
    umod[3] *= 1e-12
    ulist = list(u)

    objective_function_scipy.count += 1
    print("New iteration: %d" % objective_function_scipy.count, "u: " , u)
    objective = objectiveFunction(np.array(umod))
    if objective < objective_function_scipy.best:
        objective_function_scipy.best = objective
#    else:                  # @Arsselan: Comment in for less disc usage
#        removeDir(u)
    with open("iterations.dat", "a") as file:
        file.write("%d %e" % (objective_function_scipy.count, objective))
        for para in u:
            file.write(" %e" % para)
        file.write("\n")
    return objective


objective_function_scipy.count = 0
objective_function_scipy.best = 1e10


def doScipyOptimize():
    import scipy
    from scipy import optimize
    initialGuess = np.array([2.0, 0.5, 0.1, 0.1]) # @Arsselan: Change initial parameters here
    scipy.optimize.minimize(objective_function_scipy, initialGuess,
                            tol=0, method='L-BFGS-B', bounds=[(0.01, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)], options={'eps': 1e-4, 'maxiter': 1000})
















def objective_function_gfo(para):
    u = [para["E"], para["D1"], para["D2"], para["D3"]]
    return -objective_function_scipy(u)


def doParticleSwarm():
    from gradient_free_optimizers import ParticleSwarmOptimizer
    # @Arsselan: Change initial parameters and range here
    search_space = {
                    "E": np.arange(1e4, 2e4, 1e1),
                    "D1": np.arange(1, 10, 0.01),
                    "D2": np.arange(1, 10, 0.01),
                    "D3": np.arange(1, 10, 0.01)
                    }
    opt = ParticleSwarmOptimizer(search_space, population=5)
    opt.search(objective_function_gfo, n_iter=500)









def tangent( function, x, eps=1e-6 ):
    f = function(x)
    n = x.shape[0]
    t = np.ndarray( n )
    for i in range(n):
        iEps = eps * abs(x[i])
        xp = x * 1.0
        xp[i] += iEps
        deltaF = function( xp ) - f
        while abs(deltaF) < 1e-10:
            print("Waring! Delta too small ( deltaF=%e, iEps=%e ). " % (deltaF, iEps))
            iEps *= 10.0
            xp = x * 1.0
            xp[i] += iEps
            deltaF = function(xp) - f
        t[i] = deltaF / iEps
    return f, t

def gradientDescent( function, initial ):
    x = initial
    for i in range(100):
        alpha = 0.01# * ( 1.0 + np.linalg.norm( x ) )
        f, t = tangent( function, x, 1e-5 )
        print("Objective: %e" % f)
        print("Tangent: ", t)

        dx = - alpha / np.linalg.norm( t ) * t
        print( "Delta x: ", dx)
        x += dx
        while function(x) > f:
            x -= dx
            alpha *= 0.5
            print("Waring! Objective not reduced. Setting alpha=%e." % alpha)
            dx = - alpha / np.linalg.norm(t) * t
            print("Delta x: ", dx)
            x += dx

    return x

def doGradientDescent():

    # @Arsselan: Change initial parameters here
    initial = np.array([4.0, 1.0, 5.0, 1.0])
    x = gradientDescent( objective_function_scipy, initial )

