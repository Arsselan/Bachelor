#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

struct TripletSystem {
    std::vector<double> values;
    std::vector<int> rows;
    std::vector<int> cols;
}

struct WaveSystem {
    TripletSystem M;
    TripletSystem K;
}

WaveSystem assembleTripletSystems(ansatz, quadrature)
{
    WaveSystem system(ansatz.nDof);
    
    for(int iElement = 0; iElement<nElements; iElement++) {

        std::vector<std::vector<double> shapes;

        points = quadrature.points(iElement);

        for(int iPoint = 0; iPoint<nPoints; iPoint++) {
            
            shapes = ansatz.evaluate(iElement, points[iPoint], 1)
            
            Me += computeMassMatrix(shapes[0], rho);
            Ke += computeStiffnessMatrix(shapes[1], E*A);

        }

        lumpMatrix(lumpingMethod, Me);
        
        assemble(Me, lm, system.M);
        assemble(Ke, lm, system.K);
        
    }
    
    return system;
}


PYBIND11_MODULE(lagrange, m) {
    m.doc() = "Lagrange basis evaluation";

    m.def("evaluateLagrangeBases", &evaluateLagrangeBases, "Evaluate a lagrange basis");
}

// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include lagrange.cpp -o lagrange$(python3-config --extension-suffix)
