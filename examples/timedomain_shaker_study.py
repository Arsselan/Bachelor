import numpy as np
import shutil
import os
import sys

from context import fem1d

dataEx = np.loadtxt("examples/shaker_experiments_2.dat")

nFrequencies = 10
firstFrequency = 2
lastFrequency = firstFrequency + nFrequencies

def runStudy(params):
    print("\ndamping: %e, %e,  elasticity: %e\n" % (params[0], params[1], params[2]))
    outputDir = "shaker_damp%e_%e_elas%e" % (params[0], params[1], params[2])
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    data = np.ndarray((nFrequencies, 4))
    for i in range(nFrequencies):
        #frequency = 100 + firstFrequency * 50 + i * 50
        frequency = 50 + i * 50
        print("Frequency: %e" % frequency)
        #exec(open("examples/timedomain_shaker.py").read())
        import examples.timedomain_shaker
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        extraDelta = 0.0
        examples.timedomain_shaker.run(extraDelta, params[0], params[1], params[2], frequency, True)
        sys.stdout = save_stdout


def plotStudy(data):
    fem1d.plot(data[:, 1], [data[:, 3], data[:, 4], data[:, 5]], ["storage", "loss", "delta"])


def plotReference():
    data = np.loadtxt("examples/shaker_experiments.dat")
    fem1d.plot(data[:, 0], [
               data[:, 1], data[:, 2]*10,
               data[:, 3], data[:, 4]*10,
               data[:, 5], data[:, 6]*10],
               ["150 storage", "150 loss$\cdot$10","77 storage", "77 loss$\cdot$10","nano storage", "nano loss$\cdot$10"])


def readData(params):
    return np.loadtxt("shaker_damp%e_%e_elas%e/shaker_damp_freq_storage_loss_delta_left.dat" % (params[0], params[1], params[2]))


def plotBoth(dataSim):
    fem1d.plot(dataEx[firstFrequency:lastFrequency, 0], [
               dataEx[firstFrequency:lastFrequency, 1], dataEx[firstFrequency:lastFrequency, 2]*10,
               dataSim[:, 3], dataSim[:, 4]*10 ],
               ["150 storage", "150 loss$\cdot$10",
#                "77 storage", "77 loss$\cdot$10",
#                "nano storage", "nano loss$\cdot$10",
                      "sim storage", "sim loss$\cdot$10"])

def computeError(dataSim):
    errorStorage = np.linalg.norm(dataEx[firstFrequency:lastFrequency, 1] - dataSim[:, 3])
    errorStorage /= np.linalg.norm(dataEx[firstFrequency:lastFrequency, 1])
    errorLoss = np.linalg.norm( dataEx[firstFrequency:lastFrequency, 2] - dataSim[:, 4] )
    errorLoss /= np.linalg.norm(dataEx[firstFrequency:lastFrequency, 2])
    #return errorStorage + errorLoss
    return errorLoss


def objectiveFunction(params):
    runStudy(params)
    data = readData(params)
    error = computeError( data )
    print("\n\ndamping: %e, %e, elasticity: %e " % (params[0], params[1], params[2]))
    print(" ----> Error: %e \n\n" % error)
    return error


def removeDir(params):
    outputDir = "shaker_damp%e_%e_elas%e" % (params[0], params[1], params[2])
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)


def check(damping, damping2, elasticity, run=True):
    params = [ damping, damping2, elasticity ]
    if run:
        runStudy(params)
    data = readData(params)
    error = computeError(data)
    print("Error: %e" % error )
    plotBoth(data)



def objective_function_scipy(u):
    objective_function_scipy.count += 1
    print("New iteration: %d" % objective_function_scipy.count, "u: " , u)
    objective = objectiveFunction([ u[0], u[1], u[2] ])
    if objective < objective_function_scipy.best:
        objective_function_scipy.best = objective
    else:
        removeDir([ u[0], u[1], u[2] ])
    with open("iterations.dat", "a") as file:
        file.write("%d %e %e %e %e\n" % (
        objective_function_scipy.count, u[0], u[1], u[2], objective))
    return objective
objective_function_scipy.count = 0
objective_function_scipy.best = 1e10


# after night, eps=10: 6.215664e+02, 4.318325e+04
# after
# for storage only, eps=10: 2.582204e+03, 4.127329e+04 error:  5.106623e-02
# for storage only, eps=1: 3.426408e+03, 4.417558e+04

def optimize():
    import scipy
    from scipy import optimize
    #scipy.optimize.minimize(objective_function_scipy, [300, 4e4], method='L-BFGS-B')
#    scipy.optimize.minimize(objective_function_scipy, [4.318325e+04, 0.0],
#                            tol=0, bounds=[(0.0, 1e5), (0.0, 1e5)], options={'eps': 10, 'maxiter': 1000})
    scipy.optimize.minimize(objective_function_scipy, [50, 50, 1.e4],
                            tol=0, method='L-BFGS-B', options={'eps': 1, 'maxiter': 1000})
#    scipy.optimize.minimize(objective_function_scipy, [6.215664e+02, 0.0, 4.318325e+04],
#                            tol=0, method='L-BFGS-B', jac=None, options={'eps': 1.0, 'maxiter': 1000})




def objective_function_gfo(para):
    u = [ para["D1"], para["D2"], para["E"] ]
    return -objective_function_scipy(u)

def particleSwarm():
    from gradient_free_optimizers import ParticleSwarmOptimizer
    search_space = {"D1": np.arange(400, 800, 1),
                    "D2": np.arange(0, 2e2, 1),
                    "E": np.arange(3e4, 5e4, 10)}
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
        if abs(deltaF) < 1e-10:
            print("Waring! Delta too small.")
        t[i] = deltaF / iEps
    return f, t

def gradientDescent( function, initial ):
    x = initial
    for i in range(100):
        alpha = 0.01 * ( 1.0 + np.linalg.norm( x ) )
        f, t = tangent( function, x, 1e-3 )
        print("Objective: %e" % f)
        print("Tangent: %e %e %e" % (t[0], t[1], t[2]))
        dx = alpha / np.linalg.norm( t ) * t
        print( "Delta x: %e %e %e" % (dx[0], dx[1], dx[2]))
        x -= dx
    return x

def doGradientDescent():
    initial = np.array([6.215664e+02, 50.0, 4.318325e+04])
    x = gradientDescent( objective_function_scipy, initial )