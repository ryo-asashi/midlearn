# src/midlearn/shapley.py

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any

if TYPE_CHECKING:
    from .api import MIDRegressor, MIDExplainer
    import shap

import numpy as np
import pandas as pd

from . import _r_interface
from . import utils

try:
    import shap
except ImportError:
    shap = None

def _require_shap():
    if shap is None:
        raise ImportError(
            "The 'shap' library is required to use this feature; please install it using `pip install shap`"
        )

class MIDShapley(object):
    """MID-derived Shapley values.

    This object is returned by the `MIDRegressor.shapley()` method and holds a `shap.Explanation` object internally.
    """
    def __init__(
        self,
        estimator: MIDRegressor | MIDExplainer,
        data: pd.DataFrame
    ):
        """Initialize the MIDShapley object.

        Parameters
        ----------
        estimator : MIDRegressor or MIDExplainer
            The fitted MID model instance from which to calculate MID-derived Shapley values.
        data: pd.DataFrame
            Data used to derive predictions.
        """
        _require_shap()
        terms = estimator.terms()
        preds = pd.DataFrame(estimator.r_predict(X=data, output_type='terms', terms=terms))
        preds.columns = terms
        xvars = list(dict.fromkeys(tag for term in terms for tag in term.split(":")))
        shaps = pd.DataFrame(0.0, index=preds.index, columns=xvars)
        for term in preds.columns:
            tags = term.split(":")
            if len(tags) == 1:
                shaps[term] += preds[term]
            else:
                for tag in tags:
                    shaps[tag] += preds[term] / len(tags)
        baseline = getattr(estimator, "intercept", 0.0)
        data_reframed = _r_interface._call_r_model_reframe(
            r_object=estimator.mid_,
            data=data
        )
        data_vis = pd.DataFrame(index=data_reframed.index, columns=xvars)
        for col in xvars:
            if col in data_reframed.columns:
                s = data_reframed[col]
                if pd.api.types.is_bool_dtype(s):
                    data_vis[col] = s.astype(int)
                elif isinstance(s.dtype, pd.CategoricalDtype):
                    data_vis[col] = s.cat.codes
                else:
                    try:
                        data_vis[col] = pd.to_numeric(s)
                    except (ValueError, TypeError):
                        data_vis[col] = s
        self.explanation_: shap.Explanation = shap.Explanation(
            values=shaps.values,
            base_values=np.full(preds.shape[0], baseline),
            data=data_vis.values,
            feature_names=list(shaps.columns)
        )

def plot_shapley(
    shapley: MIDShapley,
    style: Literal['beeswarm', 'violinplot', 'waterfall', 'barplot', 'scatter'] = 'beeswarm',
    instances: Any | None = None,
    variables: Any | None = None,
    **kwargs
):
    """Visualize the calculated SHAP Explanation object with the shap library.
    This function provides a unified plotting interface for MID-derived Shapley values, wrapping various `shap.plots` functions.

    Parameters
    ----------
    shapley : MIDShapley
        A calculated SHAP Explanation wrapper object containing the term contributions.
    style : {'beeswarm', 'violinplot', 'waterfall', 'barplot', 'scatter'}, default 'beeswarm'
        The plotting style.
        'beeswarm' displays a summary plot showing the distribution of SHAP values for each feature across all samples.
        'violinplot' displays a summary plot using violin plots instead of beeswarm scatter points.
        'waterfall' displays contributions for a single prediction as a cascading plot, starting from the expected value.
        'barplot' displays the global mean absolute SHAP values (for multiple samples) or local SHAP values (for a single sample) as simple horizontal bars.
        'scatter' displays a scatter plot of SHAP values for a specific feature to show its dependence and interaction effects.
    instances : int, list of int, or slice, default None
        Specific instances (row indices) to plot. If None, all instances are used.
        For 'waterfall', it is highly recommended to specify a single row (e.g., instances=0).
    variables : str, list of str, int, or list of int, default None
        Specific variables (column names) to plot. If None, all variables are used.
    **kwargs : dict
        Additional keyword arguments forwarded to the underlying `shap.plots` functions.
    """
    _require_shap()
    style = utils.match_arg(style, ['beeswarm', 'violinplot', 'waterfall', 'barplot', 'scatter'])
    row_slice = instances if instances is not None else slice(None)
    col_slice = variables if variables is not None else slice(None)
    explanation = shapley.explanation_[row_slice, col_slice]
    if style == 'beeswarm':
        shap.plots.beeswarm(explanation, **kwargs)
    elif style == 'violinplot':
        shap.plots.violin(explanation, **kwargs)
    elif style == 'waterfall':
        if len(explanation.shape) > 1 and explanation.shape[0] > 1:
            print("Explanation object has more than one observations: the first observation is used")
            shap.plots.waterfall(explanation[0], **kwargs)
        else:
            shap.plots.waterfall(explanation, **kwargs)
    elif style == 'barplot':
        shap.plots.bar(explanation, **kwargs)
    elif style == 'scatter':
        shap.plots.scatter(explanation, **kwargs)
    else:
        raise ValueError(f"The style '{style}' is not supported")

MIDShapley.plot = plot_shapley  # type: ignore
