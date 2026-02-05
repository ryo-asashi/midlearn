# midlearn

[![PyPI version](https://badge.fury.io/py/midlearn.svg)](https://badge.fury.io/py/midlearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**midlearn** is a **rpy2**-based Python wrapper for the **midr** R package. It provides a model-agnostic framework for interpreting black-box models using a **scikit-learn** compatible API.

The core objective of midlearn is to create a globally interpretable surrogate model through **Maximum Interpretation Decomposition (MID)**. This technique finds an optimal additive approximation of any black-box model (e.g., GBMs, Neural Networks) by minimizing the squared error between the original predictions and the surrogate's components.

## Main Features

-   **Scikit-learn Compatible API**: Fits seamlessly into existing workflows with familiar `.fit()` and `.predict()` methods.
-   **Functional Decomposition**: Deconstructs model predictions into an intercept, main effects, and second-order interaction effects, minimizing the squared residuals.
-   **Model Fidelity**: Quantifies the quality of the explanation and the complexity of the model using the Uninterpreted Variation Ratio.
-   **Seamless Visualization**: Built-in support for **plotnine**-based interfaces to generate feature importance, dependence plots, and additive breakdowns.

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

MID is a functional decomposition method.
It deconstructs a black-box prediction function $f(\mathbf{x})$ into several interpretable components: intercept $g_\emptyset$, main effects $g_j(x_j)$, and interactions $g_{jk}(x_j, x_k)$, minimizing the expected squared residual $\mathbf{E}\left[g_D(\mathbf{x})^2\right]$:

$$
f(\mathbf{x}) = g_\emptyset + \sum_{j} g_j(x_{j}) + \sum_{j < k} g_{jk}(x_{j}, x_{k}) + g_D(\mathbf{x})
$$

To ensure the uniqueness and interpretability of each component, MID imposes centering and probability weighted minimum-norm constraints on the decomposition.

By replicating a black-box model with this structured surrogate, we can quantify the "uninterpreted" variance and derive a representation that captures the superior predictive power of machine learning without sacrificing actuarial clarity, as well as measure the complexity of the black-box model that can't be captured.

The theoretical foundations of MID are described in Iwasawa & Matsumori (2026) [Forthcoming], and the software implementation is detailed in [Asashiba et al. (2025)](https://arxiv.org/abs/2506.08338).

## License

**midlearn** is licensed under the MIT License.
