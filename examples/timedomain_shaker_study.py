import numpy as np
import shutil
import os
import sys

from context import fem1d

dataEx = np.loadtxt("examples/shaker_experiments.dat")

nFrequencies = 9

def runStudy(params):
    print("\ndamping: %e, %e,  elasticity: %e\n" % (params[0], params[1], params[2]))
    outputDir = "shaker_damp%e_%e_elas%e" % (params[0], params[1], params[2])
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    data = np.ndarray((nFrequencies, 4))
    for i in range(nFrequencies):
        frequency = 100 + i * 50
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
    fem1d.plot(data[:nFrequencies, 0], [
               data[:nFrequencies, 1], data[:nFrequencies, 2]*10,
               data[:nFrequencies, 3], data[:nFrequencies, 4]*10,
               data[:nFrequencies, 5], data[:nFrequencies, 6]*10],
               ["150 storage", "150 loss$\cdot$10","77 storage", "77 loss$\cdot$10","nano storage", "nano loss$\cdot$10"])


def readData(params):
    return np.loadtxt("shaker_damp%e_%e_elas%e/shaker_damp_freq_storage_loss_delta_left.dat" % (params[0], params[1], params[2]))


def plotBoth(dataSim):
    fem1d.plot(dataEx[:nFrequencies, 0], [
               dataEx[:nFrequencies, 1], dataEx[:nFrequencies, 2]*10,
#               dataEx[:nFrequencies, 3], dataEx[:nFrequencies, 4]*10,
#               dataEx[:nFrequencies, 5], dataEx[:nFrequencies, 6]*10,
               dataSim[:nFrequencies, 3], dataSim[:nFrequencies, 4]*10 ],
               ["150 storage", "150 loss$\cdot$10",
#                "77 storage", "77 loss$\cdot$10",
#                "nano storage", "nano loss$\cdot$10",
                      "sim storage", "sim loss$\cdot$10"])

def computeError(dataSim):
    errorStorage = np.linalg.norm(dataEx[:nFrequencies, 1] - dataSim[:nFrequencies, 3])
    errorStorage /= np.linalg.norm(dataEx[:nFrequencies, 1])
    errorLoss = np.linalg.norm( dataEx[:nFrequencies, 2] - dataSim[:nFrequencies, 4] )
    errorLoss /=np.linalg.norm(dataEx[:nFrequencies, 2])
    return errorStorage + errorLoss


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
    #scipy.optimize.minimize(objective_function_scipy, [300, 4e4], method='L-BFGS-B')
#    scipy.optimize.minimize(objective_function_scipy, [4.318325e+04, 0.0],
#                            tol=0, bounds=[(0.0, 1e5), (0.0, 1e5)], options={'eps': 10, 'maxiter': 1000})
    scipy.optimize.minimize(objective_function_scipy, [100, 5, 4.318325e+04],
                            tol=0.01, method='L-BFGS-B', bounds=[(0.0, 1e4), (0.0, 1e4), (0.0, 1e4)], options={'maxls': 5, 'eps': 1, 'maxiter': 1000})
#    scipy.optimize.minimize(objective_function_scipy, [6.215664e+02, 0.0, 4.318325e+04],
#                            tol=0, method='L-BFGS-B', jac=None, options={'eps': 1.0, 'maxiter': 1000})


def check(damping, damping2, elasticity, run=True):
    params = [ damping, damping2, elasticity ]
    if run:
        runStudy(params)
    data = readData(params)
    error = computeError(data)
    print("Error: %e" % error )
    plotBoth(data)