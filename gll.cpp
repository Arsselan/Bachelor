#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

double lgP(int n, double xi) {

    if(n == 0) { 
        return 1.0;
    }

    if(n == 1) {
        return xi;
    }

    double fP = 1.0;
    double sP = xi;
    double nP = 1.0;
    for(int i=2; i<n+1; i++) {
      nP = ((2.0 * i - 1.0) * xi * sP - (i - 1.0) * fP) / i;
      fP = sP; 
      sP = nP;
    }
    
    return nP;
}
    
double dLgP(int n, double xi) {
    return n * ( lgP(n - 1, xi) - xi * lgP(n, xi)) / (1.0 - xi*xi);
}

double d2LgP(int n, double xi) {
    return (2.0 * xi * dLgP(n, xi) - n * (n + 1.0) * lgP(n, xi)) / (1.0 - xi*xi);
}

double d3LgP(int n, double xi) {
    return (4.0 * xi * d2LgP(n, xi) - (n * (n + 1.0) - 2.0) * dLgP(n, xi)) / (1.0 - xi*xi);                             
}
    
std::vector<std::vector<double>> computeGllPoints(const int n, const double epsilon = 1e-15) {

    if(n < 2) {
        return { {-1, 1}, {1, 1} };
    }

    std::vector<std::vector<double>> pointsAndWeights(2, std::vector<double>(n, 0.0));

    auto &x = pointsAndWeights[0];
    auto &w = pointsAndWeights[1];
    
    x[0] = -1;
    x[n - 1] = 1;
    w[0] = 2.0 / ((n * (n - 1.0)));
    w[n - 1] = w[0];
    
    int n_2 = n / 2;
    
    double temp;                              
    for(int i=1; i<n_2; i++) {
        double xi = (1.0 - (3.0 * (n - 2.0)) / (8.0 * (n - 1.0)*(n - 1.0)*(n - 1.0))) * std::cos((4.0 * i + 1.0) * M_PI / (4.0 * (n - 1.0) + 1.0));

        double error = 1.0;

        while(error > epsilon) {
            double y  =  dLgP (n - 1, xi);
            double y1 = d2LgP (n - 1, xi);
            double y2 = d3LgP (n - 1, xi);

            double dx = 2 * y * y1 / (2 * y1*y1 - y * y2);

            xi -= dx;
            error = std::abs(dx);
        }
        
        x[i] = -xi;
        x[n - i - 1] =  xi;

        temp = lgP(n - 1.0, x[i]);
        w[i] = 2.0 / (n * (n - 1.0) * temp*temp);
        w[n - i - 1] = w[i];
    }
    
    if(n % 2 != 0) {
      x[n_2] = 0;
      temp = lgP(n - 1.0, x[n_2]);
      w[n_2] = 2.0 / ((n * (n - 1.0)) * temp*temp);
    }
    
    return pointsAndWeights;
}


PYBIND11_MODULE(gll, m) {
    m.doc() = "Gauss-Lobatto-Legendre quadrature point computation";

    m.def("computeGllPoints", &computeGllPoints, "Compute the Gauss-Lobatto-Legendre quadrature point", pybind11::arg("n"), pybind11::arg("epsilon") = 1e-15);
}

// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include gll.cpp -o gll$(python3-config --extension-suffix)


    
