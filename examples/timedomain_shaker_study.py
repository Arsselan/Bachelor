import numpy as np

from context import fem1d

nFrequencies = 11

for d in range(5):
    damping = 100 + 100 * d
    data = np.ndarray((nFrequencies, 4))
    for i in range(nFrequencies):
        frequency = 100 + i * 50
        print("\n\nFrequency: %e\n" % frequency)
        exec(open("examples/timedomain_shaker.py").read())
        data[i, 0] = frequency
        data[i, 1] = deltaLeft
        data[i, 2] = deltaStorageLeft
        data[i, 3] = deltaLossLeft
    fem1d.plot(data[:, 0], [data[:, 1], data[:, 2], data[:, 3]], ["delta", "delta storage", "delta loss"])


