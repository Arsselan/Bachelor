import matplotlib.pyplot as plt
import numpy as np

from context import fem1d

path = "results/time_domain_study_extra"

ne = 11
extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
    0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]

extras = np.array(extras) * 0.1 * 0.5

#extra = extras[-1]
extra = 0.15
exec(open("examples/timedomain_wave_propagation.py").read())

title = ansatzType
title += " " + mass
title += " %d dof" % M.shape[0]

# snapshot times
idx1 = 1
idx2 = int(nt / 4)
idx3 = int(nt / 2)
idx4 = int(nt)

# save results
data = np.zeros((nodes.size, 6))
data[:, 0] = nodes
data[:, 1] = evalU[0]
data[:, 2] = evalU[idx1]
data[:, 3] = evalU[idx2]
data[:, 4] = evalU[idx3]
data[:, 5] = evalU[idx4]
np.savetxt(path + "/snapshots_" + title.replace(' ', '_') + ".dat", data)

# plot results
plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams["figure.figsize"] = (8, 4)
figure, ax = plt.subplots()
plt.title(title)
# figure.tight_layout(pad=2.5)
# ax.set_xlim(grid.left, grid.right)
ax.plot(nodes, evalU[0], '-', label="time = %3.2f" % times[0])
ax.plot(nodes, evalU[idx1], '-', label="time = %3.2f" % times[idx1])
ax.plot(nodes, evalU[idx2], '-.', label="time = %3.2f" % times[idx2])
ax.plot(nodes, evalU[idx3], '-.', label="time = %3.2f" % times[idx3])
ax.plot(nodes, evalU[idx4], '--', label="time = %3.2f" % times[idx4])
ax.legend()
plt.savefig(path + '/snapshots' + title.replace(' ', '_') + ' .pdf')
plt.show()

