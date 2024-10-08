import numpy as np

def evaluate_legendre_bases(i, point, p, n, U):
    r = -1.0 + 2.0 * (point - U[i]) / (U[i + 1] - U[i])
    J = (U[i + 1] - U[i]) / 2.0
    order = p

    ders = np.zeros((n + 1, order + 1))

    # First row corresponds to the function values
    function_values = ders[0]

    # Temporary arrays
    L = np.zeros(order + 1)
    LD = np.zeros(order + 1)

    # Order 0
    L[0] = 1
    LD[0] = 0

    if n == 0:
        # Integral
        function_values[0] = 0.5 * (1.0 - r)

        if order > 0:
            L[1] = r
            LD[1] = 1
            function_values[1] = 0.5 * (1.0 + r)

            i = 1
            while i < order:
                i += 1
                L[i] = ((2 * i - 1) * r * L[i - 1] - (i - 1) * L[i - 2]) / i
                LD[i] = ((2 * i - 1) * (L[i - 1] + r * LD[i - 1]) - (i - 1) * LD[i - 2]) / i

                function_values[i] = (L[i] - L[i - 2]) / np.sqrt(4 * i - 2)
    
    elif n >= 1:
        derivative_values = ders[1]

        # Integral
        function_values[0] = 0.5 * (1.0 - r)
        derivative_values[0] = -0.5 / J

        if order > 0:
            L[1] = r
            LD[1] = 1
            function_values[1] = 0.5 * (1.0 + r)
            derivative_values[1] = 0.5 / J

            i = 1
            while i < order:
                i += 1
                L[i] = ((2 * i - 1) * r * L[i - 1] - (i - 1) * L[i - 2]) / i
                LD[i] = ((2 * i - 1) * (L[i - 1] + r * LD[i - 1]) - (i - 1) * LD[i - 2]) / i

                function_values[i] = (L[i] - L[i - 2]) / np.sqrt(4 * i - 2)
                derivative_values[i] = (LD[i] - LD[i - 2]) / np.sqrt(4 * i - 2) / J

    return ders

if __name__ == "__main__":
    # Beispielverwendung
    U = [0.0, 1.0]  # Beispielwerte für U
    point = 0.5  # Beispielpunkt
    i = 0  # Beispielindex
    p = 3  # Ordnung
    n = 1  # Beispiel n-Wert

    results = evaluate_legendre_bases(i, point, p, n, U)
    print("Integrierte Legendre-Basis Auswertung:\n", results)
