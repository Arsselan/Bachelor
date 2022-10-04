#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

std::vector<std::vector<double>> evaluateLagrangeBases(const int i,
                          const double point,
                          std::vector<double> points,
                          const int n,
                          const std::vector<double> &U)
{
    
    const int nPoints = points.size();
    for(double &p : points) {
        p = U[i] + (U[i+1] - U[i]) * (p + 1.0) / 2.0;
    }
    
    std::vector<std::vector<double>> ders(n+1, std::vector<double>(nPoints, 0.0));

    
    {
        std::vector<double> &basis = ders[0];
        for(int i=0; i<nPoints; i++)
        {
            basis[i] = 1;
            for(int j=0; j<nPoints; j++)
            {
                if(j!=i)
                    basis[i] *= ( (point-points[j]) / (points[i] - points[j]) );
            }
        }
    }

    
   if(n>=1) {

        std::vector<double> &basis = ders[1];
        

        for(int i=0; i<nPoints; i++)
        {
            basis[i] = 0;
            for(int j=0; j<nPoints; j++)
            {
                if(i==j)
                    continue;

                double product=1;
                for(int k=0; k<nPoints; k++)
                {
                    if(i==k || j==k)
                        continue;

                    product *= (point-points[k])/(points[i]-points[k]);
                }

                basis[i] += product/(points[i]-points[j]);
            }
        }
    }
    
    
    if(n>=2) {
    
        std::vector<double> &basis = ders[2];
            
        for(int l=0; l<nPoints; l++)
        {
		    basis[l] = 0;

		    for(int i=0; i<nPoints; i++)
		    {
			    if(l==i)
				    continue;

			    double temp = 0;
			    for(int j=0; j<nPoints; j++)
			    {
				    if(l==j || i==j)
					    continue;

				    double product=1;
				    for(int k=0; k<nPoints; k++)
				    {
					    if(l==k || i==k || j==k)
						    continue;

					    product *= (point-points[k])/(points[i]-points[k]);
				    }

				    temp += product/(points[i]-points[j]);
			    }

			    basis[l] += temp/(points[l]-points[i]);
		    }

        }

    }

    return ders;
}

PYBIND11_MODULE(lagrange, m) {
    m.doc() = "Lagrange basis evaluation";

    m.def("evaluateLagrangeBases", &evaluateLagrangeBases, "Evaluate a lagrange basis");
}

// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include lagrange.cpp -o lagrange$(python3-config --extension-suffix)
