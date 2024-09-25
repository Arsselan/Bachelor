# fem1d

This is a simple finite element code tailored to solve one-dimensional problems with a variety of different shape functions and quadrature rules.
Low level functions are implemented in C++ and compiled to Python modules using pybind11, which is included as a git submodule. 
Therefore, when cloning the repository, use
```
git clone SSH_OR_HTTP_ADRESS_OF_THE_REPOSITORY --recurse-submodules
```
Without `--recurse-submodules` the pybind11 directory will be empty.

In order to use fem1d you also need the following Python packages
* numpy
* scipy
* matplotlib


## Getting started

The first thing to do in order to use fem1d is to build the shared libraries to be used as python modules. 
If you are on a Linux system, this is done by running
```
build_gcc.sh
```
If you are on another system or want to compile using another compiler that gcc, use on of the other build scripts or provide a new one for your system.
For the compilation to work, pybind11 must be inside the respective directory. In case you cloned the fem1d repository without `--recurse-submodules`, you can do it explicitely using
```
git submodule update --init
```

### Running tests
You should run the test suite using
```
py.test-3 tests
```

### Running examples
You may run examples using
```
python3 -i -m examples/NAME_OF_THE_EXAMPLE_FILE.py
```

## Physics

### 1d wave equation
$$
\rho \, \ddot{u} - E \, u'' = f
$$

### 1d vibro-acoustics


## Shape functions

### Splines

### Lagrange

### Legendre

TODO


## Quadratures

### Binary tree

### Moment fitting


## Quadrature points

### Gauss-Legendre

### Gauss-Lobatto-Legendre

