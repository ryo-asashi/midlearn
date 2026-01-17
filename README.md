# midlearn

[![PyPI version](https://badge.fury.io/py/midlearn.svg)](https://badge.fury.io/py/midlearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**midlearn** is a **rpy2**-based Python wrapper for the **midr** R package. It provides a model-agnostic framework for interpreting black-box models using a **scikit-learn** compatible API.

The core objective of midlearn is to create a globally interpretable surrogate model through **Maximum Interpretation Decomposition (MID)**. This technique finds an optimal additive approximation of any black-box model (e.g., GBMs, Neural Networks) by minimizing the squared error between the original predictions and the surrogate's components.

## Main Features

-   **Scikit-learn Compatible API**: Fits seamlessly into existing workflows with familiar `.fit()` and `.predict()` methods.
-   **Functional Decomposition**: Deconstructs model predictions into an intercept, main effects $g_j(X_j)$, and interaction effects $g_{jk}(X_j, X_k)$.
-   **Model Fidelity**: Quantifies the quality of the explanation using the Uninterpreted Variation Ratio.
-   **Seamless Visualization**: Built-in support for plotnine-based interfaces to generate feature importance, dependence plots, and additive breakdowns.

## Installation

**midlearn** requires an R installation on your system with the **midr** package.

### 1. Install R Package

From CRAN:
``` r
install.packages('midr')
```

Or from GitHub:
``` r
pak::pak("ryo-asashi/midr")
```

### 2. Install Python package

From PyPI:
``` bash
pip install midlearn
```

Or from GitHub:
``` bash
pip install git+https://github.com/ryo-asashi/midlearn.git
```

## Theoretical Foundation

MID approximates a complex prediction function $f(\mathbf{x})$ as a sum of interpretable functions:
$$
f(\mathbf{x}) \approx g_\emptyset + \sum_{j} g_j(X_{j}) + \sum_{j < k} g_{jk}(X_{j}, X_{k}) + \dots + g_D(\mathbf{x})
$$

The theoretical foundations of MID are described in Iwasawa & Matsumori (2026) [Forthcoming], and the software implementation is detailed in [Asashiba et al. (2025)](https://arxiv.org/abs/2506.08338).

## License

**midlearn** is licensed under the MIT License.
