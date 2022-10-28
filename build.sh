#!/bin/sh

# bspline
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include bspline.cpp -o bspline$(python3-config --extension-suffix)

# gll
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include gll.cpp -o gll$(python3-config --extension-suffix)

# lagrange
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include lagrange.cpp -o lagrange$(python3-config --extension-suffix)

# legendre
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -Ipybind11/include legendre.cpp -o legendre$(python3-config --extension-suffix)

