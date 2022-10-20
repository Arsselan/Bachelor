import numpy as np

from waves1d import *


factors = []
errors = []
maxws = []
nStudies = 300
for i in range(nStudies+1):
    factor = 3*i/nStudies
    factors.append(factor)
    print("Factor: %e", factor)
    exec(open("example_timedomain.py").read())
    maxws.append(np.abs(w))
    errors.append(error)

plot(factors, [errors])

plot(factors, [maxws])

data = np.ndarray((nStudies+1, 3))
data[:, 0] = np.array(factors)
data[:, 1] = np.array(errors)
data[:, 2] = np.array(maxws)

title = ansatzType + "_RS"
filename = getFileBaseNameAndCreateDir("results/time_domain_study_fine/", title)
np.savetxt(filename + ".txt", data)

