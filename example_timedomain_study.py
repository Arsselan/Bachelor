import numpy as np

from waves1d import *

ne = 11
extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
    0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]

extras = np.array(extras) * 0.1 * 0.5

#extras = [0.0, 0.2]
errors = []
maxws = []
nStudies = len(extras)
for i in range(nStudies):
    print("Extra: %e" % extras[i])
    extra = extras[i]
    exec(open("example_timedomain.py").read())
    maxws.append(np.abs(w))
    errors.append(error)

plot(extras, [errors])

plot(extras, [maxws])

data = np.ndarray((nStudies, 3))
data[:, 0] = np.array(extras)
data[:, 1] = np.array(errors)
data[:, 2] = np.array(maxws)

title = ansatzType
if spectral is True:
    title += "_spectral"

title += "_" + mass
filename = getFileBaseNameAndCreateDir("results/time_domain_study_extra_stabilized_1e-8/", title)
np.savetxt(filename + ".txt", data)
