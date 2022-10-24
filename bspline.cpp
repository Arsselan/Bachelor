#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

std::vector<std::vector<double>> evaluateBSplineBases(const int i,
                          const double u,
                          const int p,
                          const int n,
                          const std::vector<double> &U)
{
/* Compute nonzero basis functions and their */
/* derivatives. First section is A2.2 modified */
/* to store functions and knot differences. */
/* Input: 
    i: knot span index
    u: coordinate
    p: degree
    n: max derivative order
    U: knot vector 
*/
/* Output: ders */

    std::vector<std::vector<double>> ders(n+1, std::vector<double>(p+1)); 

    std::vector<std::vector<double>> ndu(p+1, std::vector<double>(p+1));
    std::vector<double> left(p+1);
    std::vector<double> right(p+1);
    
    ndu[0][0] = 1.0;
    for(int j=1; j<=p; j++)
    {
        left[j] = u - U[i+1-j];
        right[j] = U[i+j] - u;
        double saved = 0.0;
        for(int r=0; r<j; r++)
        { 
            /* Lower triangle */
            ndu[j][r] = right[r+1] + left[j-r];
            double temp = ndu[r][j-1] / ndu[j][r];

            /* Upper triangle */
            ndu[r][j] = saved + right[r+1]*temp;
            saved = left[j-r] * temp;
        }
        ndu[j][j] = saved;
    }
    
    /* Load the basis functions */
    for (int j=0; j<=p; j++) {
        ders[0][j] = ndu[j][p];
    }
    
    /* This section computes the derivatives (Eq. [2.9]) */
    std::vector<std::vector<double>> a(2, std::vector<double>(p+1));
        
    /* Loop over function index */
    for(int r=0; r<=p; r++) 
    {
        /* Alternate rows in array a */
        int sl = 0;
        int s2 = 1;
        a[0][0] = 1.0;
        
        /* Loop to compute kth derivative */
        for(int k=1; k<=n; k++)
        {
            double d = 0.0;
            int rk = r - k;
            int pk = p - k;
            
            if (r >= k)
            {
                a[s2][0] = a[sl][0] / ndu[pk+1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            
            int j1;
            if(rk >= -1) {
                j1 = 1;
            } else { 
                j1 = -rk;
            }
            
            int j2;
            if(r-1 <= pk) { 
                j2 = k - 1;
            } else {
                j2 = p-r;
            }
               
            for(int j=j1; j<=j2; j++)
            {
                a[s2][j] = (a[sl][j] - a[sl][j-1]) / ndu[pk+1][rk+j];
                d += a[s2][j] * ndu[rk+j][pk];
            }
            
            if(r <= pk)
            {
                a[s2][k] = -a[sl][k-1] / ndu[pk+1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            
            ders[k][r] = d;
            
            /* Switch rows */
            int j = sl;
            sl = s2;
            s2 = j;
        }
    }
        
    /* Multiply through by the correct factors */
    /* (Eq. [2.9]) */
    int r = p;
    for (int k=1; k<=n; k++)
    {
        for(int j=0; j<=p; j++) {
            ders[k][j] *= r;
        }
        r *= (p-k);
    }
    
    return ders;
}

PYBIND11_MODULE(bspline, m) {
    m.doc() = "bspline basis evaluation from 'The Nurbs Book'"; // optional module docstring

    m.def("evaluateBSplineBases", &evaluateBSplineBases, "Evaluate a bspline basis");
}

// g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include bspline.cpp -o bspline$(python3-config --extension-suffix)
