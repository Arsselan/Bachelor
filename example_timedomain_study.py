from fem1d.studies import *
import sources

ne = 11
extras = list(np.linspace(0, 0.099, ne)) + list(np.linspace(0.1, 0.199, ne)) + list(np.linspace(0.2, 0.299, ne)) + [
    0.3] + list(np.linspace(0.3, 0.399, ne)) + [0.4]
extras = np.array(extras) * 0.1 * 0.5

#extras = [5e-3]

n = 240
p = 2

config = StudyConfig(
    # problem
    left=0,
    right=1.2,
    extra=0,

    # method
    # ansatzType='Lagrange',
    ansatzType='InterpolatorySpline',
    # ansatzType='Spline',
    n=n,
    p=p,
    continuity='p-1',
    mass='CON',

    depth=35,
    spectral=False,
    dual=False,
    stabilize=0,
    smartQuadrature=False,
    source=sources.NoSource()
)

#masses = ['CON', 'HRZ', 'RS']
masses = ['CON']

for mass in masses:
    print(mass)

    config.mass = mass

    errors = []
    maxws = []
    nStudies = len(extras)
    for i in range(nStudies):
        print("Extra: %e (%d/%d)" % (extras[i], i, len(extras)))

        config.extra = extras[i]
        k = eval(config.continuity)
        config.n = int(n / (config.p - k))

        try:
            exec(open("example_timedomain_reflection.py").read())
            w, error = getResults()
        except:
            print("An exception occurred")
            w = 0
            error = 0


        maxws.append(np.abs(w))
        errors.append(error)

    plot(extras, [errors])
    plot(extras, [maxws])

    data = np.ndarray((nStudies, 3))
    data[:, 0] = np.array(extras)
    data[:, 1] = np.array(errors)
    data[:, 2] = np.array(maxws)

    title = config.ansatzType
    if config.spectral is True:
        title += "_spectral"

    title += "_" + config.mass
    filename = getFileBaseNameAndCreateDir("results/time_domain_study_extra_" + str(config.stabilize) + "/", title)
    np.savetxt(filename + ".txt", data)
