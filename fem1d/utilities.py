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


def plot(ptx, pty):
    figure, ax = plt.subplots()
    for i in range(len(pty)):
        ax.plot(ptx, pty[i])
    plt.show()


def getFileBaseNameAndCreateDir(path, baseName):
    if not os.path.exists(path):
        os.makedirs(path)
    return path + baseName

