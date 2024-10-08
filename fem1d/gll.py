import numpy as np

def lgP(n, xi):
    if n == 0:
        return 1.0
    if n == 1:
        return xi

    fP = 1.0
    sP = xi
    nP = 1.0
    for i in range(2, n + 1):
        nP = ((2.0 * i - 1.0) * xi * sP - (i - 1.0) * fP) / i
        fP = sP
        sP = nP

    return nP


def dLgP(n, xi):
    return n * (lgP(n - 1, xi) - xi * lgP(n, xi)) / (1.0 - xi * xi)


def d2LgP(n, xi):
    return (2.0 * xi * dLgP(n, xi) - n * (n + 1.0) * lgP(n, xi)) / (1.0 - xi * xi)


def d3LgP(n, xi):
    return (4.0 * xi * d2LgP(n, xi) - (n * (n + 1.0) - 2.0) * dLgP(n, xi)) / (1.0 - xi * xi)


def computeGllPoints(n, epsilon=1e-15):
    if n < 2:
        return np.array([[-1, 1], [1, 1]])

    points_and_weights = np.zeros((2, n))

    x = points_and_weights[0]
    w = points_and_weights[1]

    x[0] = -1
    x[n - 1] = 1
    w[0] = 2.0 / (n * (n - 1.0))
    w[n - 1] = w[0]

    n_2 = n // 2

    for i in range(1, n_2):
        xi = (1.0 - (3.0 * (n - 2.0)) / (8.0 * (n - 1.0) ** 3)) * np.cos((4.0 * i + 1.0) * np.pi / (4.0 * (n - 1.0) + 1.0))

        error = 1.0

        while error > epsilon:
            y = dLgP(n - 1, xi)
            y1 = d2LgP(n - 1, xi)
            y2 = d3LgP(n - 1, xi)

            dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)

            xi -= dx
            error = abs(dx)

        x[i] = xi
        x[n - i - 1] = -xi
        w[i] = 2.0 / (1.0 - xi ** 2) / (y * y1) * (2.0 * (n - 1.0)) ** 2
        w[n - i - 1] = w[i]

    return points_and_weights


if __name__ == "__main__":
    n = 4  # Number of Gauss-Lobatto points
    points_and_weights = computeGllPoints(n)

    print("GLL Points and Weights:\n", points_and_weights)
