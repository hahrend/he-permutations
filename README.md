# A Compiler for Homomorphically Encrypted Permutations

This repository contains the source code for my [bachelor's thesis](https://cn.dmi.unibas.ch/fileadmin/user_upload/redesign-cn-dmi/pubs/theses/bachelor/Ahrend-Homomorphic-Permutation-Compiler.pdf) at the University of Basel (2026).
The compiler maps programs from a permutation-oriented DSL to optimized routines that work on homomorphically encrypted data.

The project is intended for academic purposes only and is not production-ready.
Use it at your own risk.

## Prerequisites

* Python 3.14
* Core Dependencies: [OpenFHE-Python](https://github.com/openfheorg/openfhe-python) 1.5.0, `networkx` 3.6
* Testing & Visualization: `numpy` 2.4, `matplotlib` 3.10

## Usage

Source programs can be directly executed using the Python interpreter by importing the `interpreter` module (see [example.py](tests/example.py)):

```python
from interpreter import *
...
```

Alternatively, the [`compile()`](src/compiler.py#L105) method will generate a `build` directory for a given source program.
Its `computation` subdirectory can be sent to an untrusted party for delegated execution.
The `computation.py` program contained within will compute the encrypted results and save them to `results`.
To retrieve the results, one must copy this directory to `display` and execute the `display.py` program.  

You can also use the [`run_test()`](src/testing.py#L37) function to automate this process and measure runtimes.
