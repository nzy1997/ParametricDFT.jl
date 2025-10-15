# ParametricDFT

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nzy1997.github.io/ParametricDFT.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nzy1997.github.io/ParametricDFT.jl/dev/)
[![Build Status](https://github.com/nzy1997/ParametricDFT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nzy1997/ParametricDFT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nzy1997/ParametricDFT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nzy1997/ParametricDFT.jl)

A Julia package for learning parametric quantum Fourier transforms using manifold optimization. This package implements a variational approach to approximate the Discrete Fourier Transform (DFT) using parameterized quantum circuits.

## Quick Start
We provide a [Makefile](Makefile) to help you get started. Please make sure you have `make` installed in your terminal environment.

1. Clone the repository
```bash
$ git clone https://github.com/nzy1997/ParametricDFT.jl.git
```

2. Install the dependencies

```bash
$ make init   # or `make update` if you want to update the dependencies
```

To verify the installation, you can run the tests
```bash
$ make test
```

3. Run the example
```bash
$ make example
```

Congratulations! You have successfully run the example. If you want to understand the theory underlying the code, please check the prerequisite in [note/stepbystep.pdf](note/stepbystep.pdf) and the note in [note/main.pdf](note/main.pdf).