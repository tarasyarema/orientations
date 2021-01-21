# Enumerating k-connected orientations

![Tests](https://github.com/tarasyarema/orientations/workflows/Test%20with%20Docker/badge.svg?branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tarasyarema/orientations/HEAD?filepath=src%2Ftests.ipynb)

> This project presents an enumeration algorithm for the k-connected orientations 
    of a given multi-graph.

Based on the following researches.
- Enumerating k-arc-connected orientations by Sarah Blind, Kolja Knauer and Petru Valicov [arXiv](https://arxiv.org/abs/1908.02050).
- Enumerating k-connected orientations by Taras Yarema [Diposit UB](#).

## Structure of the project

- [`/src/lib.sage`](./src/lib.sage) contains all the source code for the orientations library.
- [`/src/tests.ipynb`](./src/tests.ipynb) is the notebook that has all the tests for the library.
    It's a good start to get documentation about the library.
- [`/src/tests.sage`](./src/tests.sage) is the exporte Python file for the `src/test.ipynb` notebook,
    which is later copied into a Docker image for testing and ci purposes.
    This file is auto generated with a custom pre-commit hook which depends on
    [`jupytext`](https://github.com/mwouts/jupytext).

## Run locally

### Via `sage`

If you have SageMath installed and the `sage` binary in your path to run the tests you can just
run

```bash
cd src
sage tests.sage
```

After that you can use the lib copying the `src/lib.sage` and loading it from your
Sage scripts with `load('path/to/lib.sage')`.

If you have Jupyter with the Sage backend installed you can run 
the `src/tests.ipynb`.

### Using docker

This method may be handy if you do not have `sage` installed but Docker yes.
In that case, you may just run `make docker` to run the tests via the
latest SageMath image.
Have a look at the [`Dockerfile`](./Dockerfile) 
and the [`Makefile`](./Makefile) to how to use it.

## License

[MIT](./LICENSE)
