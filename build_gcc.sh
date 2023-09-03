#!/bin/sh

# bspline
echo "Building fem1d.bspline..."
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include fem1d/bspline.cpp -o fem1d/bspline$(python3-config --extension-suffix)

# lagrange
echo "Building fem1d.lagrange..."
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include fem1d/lagrange.cpp -o fem1d/lagrange$(python3-config --extension-suffix)

# legendre
echo "Building fem1d.legendre..."
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include fem1d/legendre.cpp -o fem1d/legendre$(python3-config --extension-suffix)

# gll
echo "Building fem1d.gll..."
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include fem1d/gll.cpp -o fem1d/gll$(python3-config --extension-suffix)
