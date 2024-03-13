import numpy as np
import shutil
import os

from context import fem1d

dataEx = np.loadtxt("examples/shaker_experiments.dat")

nFrequencies = 9

def runStudy(damping, elasticity):
    print("\ndamping: %e, elasticity: %e\n" % (damping, elasticity))
    outputDir = "shaker_damp%e_elas%e" % (damping, elasticity)
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    data = np.ndarray((nFrequencies, 4))
    for i in range(nFrequencies):
        frequency = 100 + i * 50
        print("\nFrequency: %e\n" % frequency)
        #exec(open("examples/timedomain_shaker.py").read())
        import examples.timedomain_shaker
        examples.timedomain_shaker.run(damping, elasticity, frequency, True)


def plotStudy(data):
    fem1d.plot(data[:, 1], [data[:, 2], data[:, 3], data[:, 4]], ["storage", "loss", "delta"])


def plotReference():
    data = np.loadtxt("examples/shaker_experiments.dat")
    fem1d.plot(data[:nFrequencies, 0], [
               data[:nFrequencies, 1], data[:nFrequencies, 2]*10,
               data[:nFrequencies, 3], data[:nFrequencies, 4]*10,
               data[:nFrequencies, 5], data[:nFrequencies, 6]*10],
               ["150 storage", "150 loss$\cdot$10","77 storage", "77 loss$\cdot$10","nano storage", "nano loss$\cdot$10"])


def readData(damping, elasticity):
    return np.loadtxt("shaker_damp%e_elas%e/shaker_damp_freq_storage_loss_delta_left.dat" % (damping, elasticity))


def plotBoth(dataSim):
    fem1d.plot(dataSim[:, 1], [dataSim[:, 2], dataSim[:, 3], dataSim[:, 4]], ["storage", "loss", "delta"])
    fem1d.plot(dataEx[:nFrequencies, 0], [
               dataEx[:nFrequencies, 1], dataEx[:nFrequencies, 2]*10,
#               dataEx[:nFrequencies, 3], dataEx[:nFrequencies, 4]*10,
#               dataEx[:nFrequencies, 5], dataEx[:nFrequencies, 6]*10,
               dataSim[:nFrequencies, 2], dataSim[:nFrequencies, 3]*10 ],
               ["150 storage", "150 loss$\cdot$10",
#                "77 storage", "77 loss$\cdot$10",
#                "nano storage", "nano loss$\cdot$10",
                      "sim storage", "sim loss$\cdot$10"])

def computeError(dataSim):
    errorStorage = np.linalg.norm(dataEx[:nFrequencies, 1] - dataSim[:nFrequencies, 2])
    errorStorage /= np.linalg.norm(dataEx[:nFrequencies, 1])
    errorLoss = np.linalg.norm( dataEx[:nFrequencies, 2] - dataSim[:nFrequencies, 3] )
    errorLoss /=np.linalg.norm(dataEx[:nFrequencies, 2])
    return errorStorage + errorLoss


def objectiveFunction(damping, elasticity):
    runStudy(damping, elasticity)
    data = readData(damping, elasticity)
    error = computeError( data )
    print("\n\ndamping: %e, elasticity: %e " % (damping, elasticity))
    print(" ----> Error: %e \n\n" % error)
    return error


def objective_function_scipy(u):
    objective_function_scipy.count += 1
    print("New iteration: %d" % objective_function_scipy.count, "u: " , u)
    return objectiveFunction(u[0], u[1])
objective_function_scipy.count = 0

def optimize():
    import scipy
    #scipy.optimize.minimize(objective_function_scipy, [300, 4e4], method='L-BFGS-B')
    scipy.optimize.minimize(objective_function_scipy, [300, 4e4],
                            tol=0, method='L-BFGS-B', jac=None, options={'eps': 10.0, 'maxiter': 1000})

