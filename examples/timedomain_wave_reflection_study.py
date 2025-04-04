
import numpy as np
from context import fem1d

ne = 11
extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
    0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]
extras = np.array(extras) * 0.1 * 0.5

#extras = [1.495e-2]

#extras = [5e-3]

#extras = np.linspace(1.4455e-2, 1.495e-02, 11)

#extras = np.logspace(np.log10(1.4455e-2), np.log10(1.495e-02), 11)

#extras = np.linspace(1.4455e-2, 1.495e-02 + (1.495e-02 - 1.4455e-2)*0.1, 12)

#extras = [1.495e-02 + (1.495e-02 - 1.4455e-2)*0.04]

#extras = [1.495e-02]

#extras = np.linspace(1.492e-2, 1.4955e-02, 36)
#extras = np.linspace(1.492e-2, 1.493e-2, 11)

# extras = [0.014955]

n = 240
p = 3

config = fem1d.StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0,

    # method
    ansatzType='Lagrange',
    #ansatzType='InterpolatorySpline',
    # ansatzType='Spline',
    n=n,
    p=p,
    continuity='p-1',
    mass='CON',

    depth=35,
    spectral=False,
    dual=False,
    stabilize=0.0,
    smartQuadrature=True,

    source=fem1d.sources.NoSource(),

    eigenvalueStabilizationM=1e-4
)

masses = ['CON', 'HRZ']
# masses = ['RS']
#masses = ['CON']

for mass in masses:
    print("\n\n mass: " + mass)

    config.mass = mass

    errors = []
    maxws = []
    tMaxs = []
    dts = []
    nts = []
    nStudies = len(extras)
    for i in range(nStudies):
        print("\nExtra: %e (%d/%d)" % (extras[i], i, len(extras)))

        config.extra = extras[i]
        k = eval(config.continuity)
        config.n = int(n / (config.p - k))

        try:
            exec(open("examples/timedomain_wave_reflection.py").read())
            w, error, tMax, dt, nt = getResults()
        except:
            print("An exception occurred")
            w = 0
            error = 0
            tMax = 0
            dt = 0
            nt = 0

        print("Error: %e" % error)
        maxws.append(np.abs(w))
        errors.append(error)
        tMaxs.append(tMax)
        dts.append(dt)
        nts.append(nt)

    fem1d.plot(extras, [errors])
    fem1d.plot(extras, [maxws])

    data = np.ndarray((nStudies, 6))
    data[:, 0] = np.array(extras)
    data[:, 1] = np.array(errors)
    data[:, 2] = np.array(maxws)
    data[:, 3] = np.array(tMaxs)
    data[:, 4] = np.array(dts)
    data[:, 5] = np.array(nts)

    title = config.ansatzType
    if config.spectral is True:
        title += "_spectral"

    title += "_" + config.mass
    title += "_p%d" % config.p
    title += "_n%d" % config.n
    title += "_dof%d" % study.ansatz.nDof()
    title += "_alp%e" % config.stabilize
    title += "_eps%e" % config.eigenvalueStabilizationM
    filename = fem1d.getFileBaseNameAndCreateDir("results/timedomain_wave_reflection_study"
                                                 + "/const_dt/", title)
    np.savetxt(filename + ".dat", data)
