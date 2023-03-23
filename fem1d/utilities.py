import os
import numpy as np
import matplotlib.pyplot as plt


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot(ptx, pty, labels=[]):
    figure, ax = plt.subplots()
    if len(labels) == 0:
        for i in range(len(pty)):
            ax.plot(ptx, pty[i])
    else:
        for i in range(len(pty)):
            ax.plot(ptx, pty[i], label=labels[i])
        plt.legend()
    plt.show()


def getFileBaseNameAndCreateDir(path, baseName):
    if not os.path.exists(path):
        os.makedirs(path)
    return path + baseName


def writeColumnFile(path, columns):
    nCols = len(columns)
    nRows = len(columns[0])
    data = np.ndarray((nRows, nCols))
    for i in range(nCols):
        data[:, i] = columns[i]
    np.savetxt(path, data)

