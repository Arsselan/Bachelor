#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <sciplot/sciplot.hpp>

#include "BSpline.h"


int main() {

    BSpline bspline{3, 1, {0,0,0,0,1,2,3,4,4,4,4}};
    std::size_t nShapes = bspline.knots.size() - bspline.p - 1;
    
    std::size_t nCoords = 401;
    std::vector<std::vector<double>> result(nCoords);


    std::ofstream file("shapes.dat");
    std::vector<double> coords(nCoords);
    std::vector<std::vector<double>> shapes(nShapes, std::vector<double>(nCoords));
    
    
    for(std::size_t i=0; i<nCoords; i++) {
        double coord = 4.0 * i / (nCoords-1);
        coords[i] = coord;
        result[i] = evaluate(bspline, coord);
        
        std::cout << std::setw(5) << coord << ": ";
        file << coord << " ";
        for(std::size_t j = 0; j<result[i].size(); j++) {
            double val = result[i][j];
            shapes[j][i] = val;
            std::cout << val << " ";
            file << val << " ";
        }
        std::cout << std::endl;
        file << "\n";
    }
    file.close();
    

    sciplot::Plot2D plot;

    // Set the x and y labels
    plot.xlabel("x");
    plot.ylabel("y");

    // Set the x and y ranges
    plot.xrange(0.0, 4.0);
    plot.yrange(0.0, 1.0);

    // Set the legend to be on the bottom along the horizontal
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
        
    //sciplot::Vec x = sciplot::linspace(0.0, 4.0, 200);
    
    for(std::size_t i=0; i<nShapes; i++) {
        plot.drawCurve(coords, shapes[i]).label("N" + std::to_string(i));
    }
    
    // Create figure to hold plot
    sciplot::Figure fig = {{plot}};
    // Create canvas to hold figure
    sciplot::Canvas canvas = {{fig}};
    canvas.size(1000,500);

    // Show the plot in a pop-up window
    canvas.show();
    
    
 
    std::vector<std::vector<double>> shapes2(nShapes, std::vector<double>(nCoords, 0.0));
    std::vector<std::vector<double>> ders(1, std::vector<double>(bspline.p+1));
    
    for(std::size_t i=0; i<nCoords; i++) {
    
        int knotSpanIndex = bspline.p+coords[i];
        if(knotSpanIndex>nShapes-1) knotSpanIndex = nShapes-1;
        
        evaluateBSplineBases(knotSpanIndex, coords[i], bspline.p, 0, bspline.knots, ders);

        for(std::size_t j = 0; j<ders[0].size(); j++) {
            double val = ders[0][j];

            int shapeIndex = j+coords[i];
            if(shapeIndex>nShapes-1) shapeIndex = nShapes-1;
            shapes2[shapeIndex][i] = val;
        }
        
        
    }
    
    sciplot::Plot2D plot2;

    // Set the x and y labels
    plot2.xlabel("x");
    plot2.ylabel("y");

    // Set the x and y ranges
    plot2.xrange(0.0, 4.0);
    plot2.yrange(0.0, 1.0);

    // Set the legend to be on the bottom along the horizontal
    plot2.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
        
    //sciplot::Vec x = sciplot::linspace(0.0, 4.0, 200);
    
    for(std::size_t i=0; i<shapes2.size(); i++) {
        plot2.drawCurve(coords, shapes2[i]).label("N" + std::to_string(i));
    }
    
    // Create figure to hold plot
    sciplot::Figure fig2 = {{plot2}};
    // Create canvas to hold figure
    sciplot::Canvas canvas2 = {{fig2}};
    canvas2.size(1000,500);

    // Show the plot in a pop-up window
    canvas2.show();
    
    
    auto errors = shapes;
    std::vector<double> error(shapes2.size(), 0.0);
    for(std::size_t i=0; i<shapes2.size(); i++) {
        for(std::size_t j=0; j<nCoords; j++) {
            errors[i][j] = shapes[i][j] - shapes2[i][j];
            error[i] += errors[i][j]*errors[i][j];
        }
        std::cout << "error: " << error[i] << std::endl;
    }
    

    
}
