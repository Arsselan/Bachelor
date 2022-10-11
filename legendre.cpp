#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

std::vector<std::vector<double>> evaluateLegendreBases(const int i,
                          const double point,
                          const int p,
                          const int n,
                          const std::vector<double> &U)
{
    
    
    double r = -1.0 + 2.0 * (point - U[i]) / (U[i+1] - U[i]);
    double J = (U[i+1] - U[i]) / 2.0;
    int order = p;
    
    std::vector<std::vector<double>> ders(n+1, std::vector<double>(p+1, 0.0));
        
    std::vector<double> &functionValues = ders[0];

    // temporary
    std::vector<double> L(order + 1);
    std::vector<double> LD(order + 1);

    // order 0
    L[0] = 1;
    LD[0] = 0;


    if(n == 0) {
        
        // integral
        functionValues[0] = 0.5 * (1.0 - r);

        if (order > 0) {
            L[1] = r;
            LD[1] = 1;
            functionValues[1] = 0.5 * (1.0 + r);

            int i = 1;
            while (i < order)
            {
                i++;
                L[i] = ( (2 * i - 1) * r * L[i - 1] - (i - 1) * L[i - 2] ) / i;
                LD[i] = ( (2 * i - 1) * (L[i - 1] + r * LD[i-1]) - (i - 1) * LD[i - 2] ) / i;

                functionValues[i] = (L[i] - L[i - 2]) / std::sqrt(4 * i - 2);
            }
        }
        
    } else if(n>=1) {
    
        std::vector<double> &derivativeValues = ders[1];
        
        // integral
        functionValues[0] = 0.5 * (1.0 - r);
        derivativeValues[0] = -0.5 / J;

        if (order > 0) {
            L[1] = r;
            LD[1] = 1;
            functionValues[1] = 0.5 * (1.0 + r);
            derivativeValues[1] = 0.5 / J;

            int i = 1;
            while (i < order)
            {
                i++;
                L[i] = ( (2 * i - 1) * r * L[i - 1] - (i - 1) * L[i - 2] ) / i;
                LD[i] = ( (2 * i - 1) * (L[i - 1] + r * LD[i-1]) - (i - 1) * LD[i - 2] ) / i;

                functionValues[i] = (L[i] - L[i - 2]) / std::sqrt(4 * i - 2);
                derivativeValues[i] = (LD[i] - LD[i - 2]) / std::sqrt(4 * i - 2) / J;
            }
        }
        
    }
    
    return ders;
}

PYBIND11_MODULE(legendre, m) {
    m.doc() = "Integrated legendre basis evaluation";

    m.def("evaluateLegendreBases", &evaluateLegendreBases, "Evaluate an integrated Legendre basis");
}

// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include legendre.cpp -o legendre$(python3-config --extension-suffix)

