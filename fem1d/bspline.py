import numpy as np

def compute_bspline_basis(n, p, u):
    """
    Computes the B-spline basis functions and their derivatives.
    :param n: Number of control points
    :param p: Degree of the B-spline
    :param u: Knot vector
    :return: The B-spline basis functions and their derivatives
    """
    N = len(u) - p - 1  # Number of basis functions
    b_spline_basis = np.zeros((N, p + 1))

    # Initialize the first degree B-spline basis functions
    for i in range(N):
        if u[i] <= u[i + 1] and u[i] <= u[i + 2]:
            b_spline_basis[i][0] = 1.0

    # Compute higher degree B-spline basis functions
    for k in range(1, p + 1):
        for i in range(N - k):
            denom1 = u[i + k] - u[i]
            denom2 = u[i + k + 1] - u[i + 1]

            if denom1 != 0:
                b_spline_basis[i][k] += (b_spline_basis[i][k - 1] * (u[i + k] - u[i])) / denom1
            if denom2 != 0:
                b_spline_basis[i + 1][k] += (b_spline_basis[i + 1][k - 1] * (u[i + k + 1] - u[i + 1])) / denom2

    return b_spline_basis

if __name__ == "__main__":
    # Example usage
    n = 4  # Number of control points
    p = 2  # Degree of B-spline
    u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # Knot vector

    b_spline_basis = compute_bspline_basis(n, p, u)
    print("B-Spline Basis Functions:\n", b_spline_basis)
