import numpy as np

def evaluateLagrangeBases(i, point, points, n, U):
    n_points = len(points)
    
    # Transform points based on U
    points = [U[i] + (U[i + 1] - U[i]) * (p + 1.0) / 2.0 for p in points]
    
    # Initialize the ders matrix
    ders = np.zeros((n + 1, n_points))

    # Evaluate the zeroth order Lagrange basis
    basis = ders[0]
    for k in range(n_points):
        basis[k] = 1.0
        for j in range(n_points):
            if j != k:
                basis[k] *= (point - points[j]) / (points[k] - points[j])

    # Evaluate the first order Lagrange basis
    if n >= 1:
        basis = ders[1]
        for k in range(n_points):
            basis[k] = 0.0
            for j in range(n_points):
                if k == j:
                    continue

                product = 1.0
                for m in range(n_points):
                    if k == m or j == m:
                        continue
                    product *= (point - points[m]) / (points[k] - points[m])
                
                basis[k] += product / (points[k] - points[j])

    # Evaluate the second order Lagrange basis
    if n >= 2:
        basis = ders[2]
        for l in range(n_points):
            basis[l] = 0.0
            for k in range(n_points):
                if l == k:
                    continue

                temp = 0.0
                for j in range(n_points):
                    if l == j or k == j:
                        continue

                    product = 1.0
                    for m in range(n_points):
                        if l == m or k == m or j == m:
                            continue
                        product *= (point - points[m]) / (points[k] - points[m])

                    temp += product / (points[k] - points[j])

                basis[l] += temp / (points[l] - points[k])

    return ders

if __name__ == "__main__":
    # Example usage
    U = [0.0, 1.0]  # Example values for U
    points = [-1.0, 0.0, 1.0]  # Example points
    i = 0  # Example index
    point = 0.5  # Example point to evaluate

    results = evaluate_lagrange_bases(i, point, points, n=2, U=U)
    print("Lagrange Basis Evaluation:\n", results)
