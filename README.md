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

MID is a functional decomposition method that deconstructs a black-box prediction function $f(\mathbf{X})$ into several interpretable components: an intercept $g_\emptyset$, main effects $g_j(X_j)$, and second-order interactions $g_{jk}(X_j, X_k)$, minimizing the squared residuals $\mathbf{E}\left[g_D(\mathbf{X})^2\right]$:

$$
f(\mathbf{X}) = g_\emptyset + \sum_{j} g_j(X_{j}) + \sum_{j < k} g_{jk}(X_{j},\;X_{k}) + g_D(\mathbf{X})
$$

To ensure the uniqueness and identifiability of each component, MID imposes centering and probability-weighted minimum-norm constraints on the decomposition.

By approximating a black-box model with this surrogate structure, we can derive a representation that retains the superior predictive power of machine learning models without sacrificing actuarial transparency.
Furthermore, it allows us to quantify the "uninterpreted" variance, i.e., the portion of the model's logic that can't be captured by low-order effects, via the residual term $g_D(\mathbf{X})$.

The theoretical foundations of MID are described in Iwasawa & Matsumori (2026) [Forthcoming], and the software implementation is detailed in [Asashiba et al. (2025)](https://arxiv.org/abs/2506.08338).

## License

**midlearn** is licensed under the MIT License.
