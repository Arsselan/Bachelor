import matplotlib.pyplot as plt
import numpy as np

#path = "results/time_domain_study_extra"
#path = "results/time_domain_study_extra_not_stabilized"
#path = "results/time_domain_study_extra_stabilized_1e-8"
path = "results/timedomain_wave_reflection_study_0"

names = [
    "Spline_CON",
    "Spline_RS",
    # "Spline_HRZ",
    "Lagrange_CON",
    # "Lagrange_RS",
    "Lagrange_HRZ",
    "InterpolatorySpline_CON",
    # "InterpolatorySpline_RS",
    "InterpolatorySpline_HRZ",
    "Lagrange_spectral_CON",
    "Lagrange_spectral_HRZ",
]

# load data
nNames = len(names)
data = [None] * nNames
for iName in range(nNames):
    data[iName] = np.genfromtxt(path + "/" + names[iName] + ".txt")


# prepare figure
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams["figure.figsize"] = (12, 6)

figure, ax = plt.subplots(1, 2)
figure.tight_layout(pad=2.5)

# ax.set_xlim(grid.left, grid.right)
# ax[0].set_ylim(500, 4000)
# ax[1].set_ylim(1e-6, 1)

# plot
for iName in range(nNames):
    cData = data[iName]
    style = '-o'
    if iName > 3:
        style = '--.'
    ax[0].semilogy(cData[:, 0], cData[:, 2], style, label="wmax " + names[iName])
    ax[1].semilogy(cData[:, 0], cData[:, 1], style, label="err " + names[iName])

ax[0].set_title('Maximum eigenvalues')
ax[1].set_title('Time domain error')
ax[0].set_xlabel('ficticious domain size')
ax[1].set_xlabel('ficticious domain size')
ax[0].set_ylabel('maximum eigenvalue')
ax[1].set_ylabel('error in time domain')

ax[0].legend()
ax[1].legend()

plt.savefig(path + '/comparison.pdf')

plt.show()
