import matplotlib.pyplot as plt
import numpy as np

from waves1d import *

path = "results/time_domain_study_extra"

ne = 11
extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
    0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]

extras = np.array(extras) * 0.1 * 0.5

#extra = extras[-1]
extra = 0.018
exec(open("example_timedomain.py").read())




plt.rcParams['axes.titleweight'] = 'bold'
#plt.rcParams["figure.figsize"] = (8, 4)

figure, ax = plt.subplots()

title = ansatzType
title += " " + mass
title += " %d dof " % M.shape[0]

plt.title(title)
#figure.tight_layout(pad=2.5)

# ax.set_xlim(grid.left, grid.right)

ax.plot(nodes, evalU[1], '-', label="time = %3.2f" % times[1])
idx = int(nt / 4)
ax.plot(nodes, evalU[idx], '-.', label="time = %3.2f" % times[idx])
idx = int(nt / 2)
ax.plot(nodes, evalU[idx], '-.', label="time = %3.2f" % times[idx])
idx = int(nt)
ax.plot(nodes, evalU[idx], '--', label="time = %3.2f" % times[idx])

ax.legend()

plt.savefig(path + '/snapshots' + title.replace(' ', '_') + ' .pdf')

plt.show()

