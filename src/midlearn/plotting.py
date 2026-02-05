# src/midlearn/plotting.py

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Any
if TYPE_CHECKING:
    from .api import (
        MIDRegressor, 
        MIDExplainer, 
        MIDImportance, 
        MIDBreakdown, 
        MIDConditional
    )

import numpy as np
import pandas as pd
import plotnine as p9

from . import _r_interface
from . import plotting_theme as pt
from . import utils

def plot_effect(
    estimator: MIDRegressor | MIDExplainer,
    term: str,
    style: Literal['effect', 'data'] = 'effect',
    theme: str | pt.color_theme | None = None,
    intercept: bool = False,
    main_effects: bool = False,
    data: pd.DataFrame | None = None,
    jitter: float | list[float] = 0.3,
    resolution: int | tuple[int, int] = 100,
    lumped: bool = True,
    **kwargs
):
    """Visualize the estimated main or interaction effect of a fitted MID model with plotnine.
    This is a porting function for the R function `midr::ggmid.mid()`.

    Parameters
    ----------
    estimator : MIDRegressor or MIDExplainer
        A fitted MIDRegressor or MIDExplainer object containing the model components.
    term : str
        The name of the component function (main effect or interaction term) to plot.
    style : {'effect', 'data'}, default 'effect'
        The plotting style. 
        'effect' plots the estimated component function as a line or a surface. 
        'data' plots the specified data points (jittered for factor variables) with MID values represented by color.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the plot.
    intercept : bool, default False
        If True, the global intercept term is added to the component function values.
    main_effects : bool, default False
        If True, main effects are included when plotting two-way interaction terms.
        Ignored for single-term plots.
    data : pandas.DataFrame or None, default None
        The data frame to plot. Required only if `style='data'`.
    jitter : float or list of float, default 0.3
        The amount of jitter to apply to factor variables when `style='data'` is used.
    resolution : int or tuple[int, int], default 100
        The resolution (number of grid points) for calculating the effect. 
        If a single integer, it is used for both axes of a 2D interaction plot. 
        If a tuple (int, int), it specifies the resolution for the first and 
        second predictor in an interaction, respectively.
    lumped: bool, default True
        Controls whether to use the encoded (reduced) levels or the original raw levels when plotting factor variables.
        Automatically set to False when `main_effects=True` to ensure accurate summation of effects across detailed levels.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.
    
    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the visualization of the component function.
    """
    style = utils.match_arg(style, ['effect', 'data'])
    tags = term.split(':')
    lumped = lumped and not main_effects
    if style == 'data':
        if not isinstance(jitter, list):
            jitter = [jitter] * len(tags)
        if data is None:
            raise ValueError("The 'data' argument is required when style='data'. Please provide the pandas.DataFrame to use for plotting.")
        data = data.copy()
        terms = [term]
        if main_effects and len(tags) == 2:
            terms.extend(tags)
        data['mid'] = estimator.r_predict(X=data, output_type='link', terms=terms)
        if intercept:
            data['mid'] += estimator.intercept
    if len(tags) == 1:
        enc = estimator._encoder(tag=term, order=1)
        ety = _r_interface._extract_and_convert(enc, 'type')[0]
        if ety != 'factor' or lumped:
            df = estimator.main_effects(term).copy()
        else:
            lvs = _r_interface._extract_and_convert(enc, 'envir')['olvs']
            df = pd.DataFrame({
                term: pd.Categorical(lvs, categories=lvs),
                f'{term}_levels': list(range(1, len(lvs) + 1))
            })
            df['mid'] = estimator.effect(term=term, x=df)
        if intercept:
            df['mid'] += estimator.intercept
        p = p9.ggplot(data=df, mapping=p9.aes(x=term, y='mid'))
        if style == 'effect':
            if ety == 'linear':
                p = p + p9.geom_line(**kwargs)
                if theme is not None:
                    p = p + p9.aes(color='mid') + pt.scale_color_theme(theme)
            elif ety == 'constant':
                xval = df[[f'{term}_min', f'{term}_max']].to_numpy().ravel('C')
                yval = np.repeat(df['mid'].to_numpy(), 2)
                path_df = pd.DataFrame({term: xval, 'mid': yval})
                p += p9.geom_path(data=path_df, **kwargs)
                if theme is not None:
                    p = p + p9.aes(color='mid') + pt.scale_color_theme(theme)
            else:
                p += p9.geom_col(**kwargs)
                if theme is not None:
                    p = p + p9.aes(fill='mid') + pt.scale_fill_theme(theme)
        if style == 'data':
            jit = 0
            if ety == 'factor':
                jit = jitter[0]
                env = _r_interface._extract_and_convert(enc, 'envir')
                data[term] = env['transform'](data[term], lumped = lumped)
            p += p9.geom_jitter(p9.aes(y = "mid"), data=data, width=jit, height=0, **kwargs)
            if theme is not None:
                p = p + p9.aes(color='mid') + pt.scale_color_theme(theme)
    elif len(tags) == 2:
        xtag, ytag = tags[0], tags[1]
        xenc = estimator._encoder(tag=xtag, order=2)
        yenc = estimator._encoder(tag=ytag, order=2)
        xety = _r_interface._extract_and_convert(xenc, 'type')[0]
        yety = _r_interface._extract_and_convert(yenc, 'type')[0]
        if xety != 'factor' or lumped:
            xfrm = _r_interface._extract_and_convert(xenc, 'frame')
        else:
            xlvs = _r_interface._extract_and_convert(xenc, 'envir')['olvs']
            xfrm = pd.DataFrame({
                xtag: pd.Categorical(xlvs, categories=xlvs),
                f'{xtag}_levels': list(range(1, len(xlvs) + 1))
            })
        if yety != 'factor' or lumped:
            yfrm = _r_interface._extract_and_convert(yenc, 'frame')
        else:
            ylvs = _r_interface._extract_and_convert(yenc, 'envir')['olvs']
            yfrm = pd.DataFrame({
                ytag: pd.Categorical(ylvs, categories=ylvs),
                f'{ytag}_levels': list(range(1, len(ylvs) + 1))
            })
        xidx = np.tile(np.arange(len(xfrm)), len(yfrm))
        yidx = np.repeat(np.arange(len(yfrm)), len(xfrm))
        df = pd.concat(
            [xfrm.iloc[xidx].reset_index(drop=True),
             yfrm.iloc[yidx].reset_index(drop=True)], axis=1
        )
        df['mid'] = estimator.effect(term=term, x=df)
        if intercept:
            df['mid'] += estimator.intercept
        if main_effects:
            df['mid'] += estimator.effect(term=xtag, x=df) + estimator.effect(term=ytag, x=df)
        p = p9.ggplot(df, p9.aes(x=xtag, y=ytag))
        if style == 'effect':
            xres, yres = (resolution, resolution) if isinstance(resolution, int) else (resolution, resolution)
            if xety == 'factor':
                xval = xfrm[xtag].unique()
            else:
                xmin, xmax = df[f'{xtag}_min'].min(), df[f'{xtag}_max'].max()
                xval = np.linspace(xmin, xmax, xres)
            if yety == 'factor':
                yval = yfrm[ytag].unique()
            else:
                ymin, ymax = df[f'{ytag}_min'].min(), df[f'{ytag}_max'].max()
                yval = np.linspace(ymin, ymax, yres)
            grid_df = pd.DataFrame({
                xtag: np.repeat(xval, len(yval)),
                ytag: np.tile(yval, len(xval))
            })
            grid_df['mid'] = estimator.effect(term=term, x=grid_df)
            if intercept:
                grid_df['mid'] += estimator.intercept
            if main_effects:
                grid_df['mid'] += estimator.effect(term=xtag, x=grid_df) + estimator.effect(term=ytag, x=grid_df)
            p += p9.geom_raster(p9.aes(x=xtag, y=ytag, fill='mid'), data=grid_df)
            p += pt.scale_fill_theme(theme if theme is not None else 'midr')
        if style == 'data':
            xjit, yjit = 0, 0
            if xety == 'factor':
                xjit = jitter[0]
                env = _r_interface._extract_and_convert(xenc, 'envir')
                data[xtag] = env['transform'](data[xtag], lumped = lumped)
            if yety == 'factor':
                yjit = jitter[1]
                env = _r_interface._extract_and_convert(yenc, 'envir')
                data[ytag] = env['transform'](data[ytag], lumped = lumped)
            p += p9.geom_jitter(
                mapping=p9.aes(color='mid'), data=data, width=xjit, height=yjit, **kwargs
            )
            if theme is not None:
                p += pt.scale_color_theme(theme)
            else:
                p += p9.scale_color_continuous()
    return p


def plot_importance(
    importance: MIDImportance,
    style: Literal['barplot', 'heatmap'] = 'barplot',
    theme: str | pt.color_theme | None = None,
    terms: list[str] | None = None,
    max_nterms: int | None = 30,
    **kwargs
):
    """Visualize the importance scores of the component functions from a fitted MID model with plotnine.
    This is a porting function for the R function `midr::ggmid.mid.importance()`.

    Parameters
    ----------
    importance : MIDImportance
        A fitted :class:`MIDImportance` object containing the component importance scores.
    style : {'barplot', 'heatmap'}, default 'barplot'
        The plotting style.
        'barplot' displays importance as horizontal bars, suitable for a large number of terms.
        'heatmap' displays importance in a matrix format, suitable for visualizing main effects and two-way interactions simultaneously.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the plot.
    terms : list[str] or None, default None
        An explicit list of term names to display.
        If provided, only the terms in this list are plotted.
    max_nterms : int or None, default 30
        The maximum number of terms to display when `style!='heatmap'`. 
        Terms are sorted by importance before truncation. If None, all terms are displayed.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.

    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the visualization of component importance.
    """
    style = utils.match_arg(style, ['barplot', 'heatmap', 'boxplot', 'violinplot', 'sinaplot'])
    imp_df = importance.importance.copy()
    if terms is not None:
        in_terms = imp_df['term'].isin(terms)
        imp_df = imp_df[in_terms]
    if style != 'heatmap' and max_nterms is not None:
        imp_df = imp_df.head(max_nterms)
    if style == 'barplot':
        p = (
            p9.ggplot(imp_df, p9.aes(x='term', y='importance'))
            + p9.geom_col(**kwargs)
            + p9.coord_flip()
            + p9.labs(x="")
        )
        if theme is not None:
            theme = pt.color_theme(theme)
            var_fill = 'order' if theme.theme_type == 'qualitative' else 'importance'
            p = p + p9.aes(fill=var_fill) + pt.scale_fill_theme(theme)
    elif style == 'heatmap':
        terms = imp_df['term'].str.split(':', expand=True)
        if terms.shape[1] == 1:
            terms.loc[:, 1] = None
        terms[1] = terms[1].fillna(terms[0])
        df1 = pd.DataFrame({
            'x': terms[0], 'y':terms[1], 'importance': imp_df['importance']
        })
        df2 = pd.DataFrame({
            'x': terms[1], 'y':terms[0], 'importance': imp_df['importance']
        })
        df = pd.concat([df1, df2]).drop_duplicates(ignore_index=True)
        all_vars = pd.unique(np.concatenate([terms[0], terms[1]]))
        df['x'] = pd.Categorical(df['x'], categories=all_vars)
        df['y'] = pd.Categorical(df['y'], categories=all_vars)
        p = (
            p9.ggplot(df, p9.aes(x='x', y='y', fill='importance'))
            + p9.geom_tile(**kwargs)
            + p9.labs(x="", y="")
        )
        p += pt.scale_fill_theme(theme if theme is not None else 'grayscale')
    elif style in ['boxplot', 'violinplot', 'sinaplot']:
        terms = imp_df['term'].tolist()
        dist_df = importance.predictions[terms].melt(
            var_name='term', value_name='mid'
        )
        dist_df = dist_df.merge(
            imp_df[['term', 'order', 'importance']], on='term'
        )
        dist_df['term'] = pd.Categorical(
            dist_df['term'],
            categories=terms[::-1],
            ordered=True
        )
        p = p9.ggplot(dist_df, p9.aes(x='term', y='mid'))
        if style == 'boxplot':
            p += p9.geom_boxplot(**kwargs)
        elif style == 'violinplot':
            kwargs.setdefault('scale', 'width')
            p += p9.geom_violin(**kwargs)
        elif style == 'sinaplot':
            kwargs.setdefault('scale', 'width')
            kwargs.setdefault('method', 'density')
            p += p9.geom_sina(**kwargs)
        p = p + p9.coord_flip() + p9.labs(x = '')
        if theme is not None:
            theme = pt.color_theme(theme)
            if style != 'sinaplot':
                var_fill = 'order' if theme.theme_type == 'qualitative' else 'importance'
                p = p + p9.aes(fill=var_fill, group='term') + pt.scale_fill_theme(theme)
            else:
                if theme.theme_type == 'qualitative':
                    var_color = 'order'
                elif theme.theme_type == 'sequential':
                    var_color = 'importance'
                else:
                    var_color = 'mid'
                p = p + p9.aes(color=var_color, group='term') + pt.scale_color_theme(theme)
    return p


def plot_breakdown(
    breakdown: MIDBreakdown,
    style: Literal['waterfall', 'barplot'] = 'waterfall',
    theme: str | pt.color_theme | None = None,
    terms: list[str] | None = None,
    max_nterms: int | None = 15,
    catchall: str = '(others)',
    label_pattern: list[str] | None = ['%t=%v', '%t:%t'],
    format_args: dict[str, Any] = dict(),
    **kwargs
):
    """Visualize the decomposition of a single prediction into contributions from each component term with plotnine.
    This is a porting function for the R function `midr::ggmid.mid.breakdown()`.

    Parameters
    ----------
    breakdown : MIDBreakdown
        A fitted :class:`MIDBreakdown` object containing the term contributions for a specific data point.
    style : {'waterfall', 'barplot'}, default 'waterfall'
        The plotting style.
        'waterfall' displays contributions as a cascading plot, showing how each term adds to the final prediction, starting from the intercept.
        'barplot' displays contributions as simple horizontal bars, relative to zero.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the plot.
    terms : list[str] or None, default None
        An explicit list of term names to display.
        If provided, only the terms in this list are plotted individually and
        all other contributions are aggregated into a single category defined by `catchall`.
    max_nterms : int or None, default 15
        The maximum number of terms to display. Terms beyond this limit are 
        grouped into a single 'catchall' category. If None, all terms are displayed.
    catchall : str, default '(others)'
        The label used for the grouped category when the number of terms exceeds `max_nterms`.
    label_pattern : list of str or None, default None
        A list of one or two format strings for axis labels.
        The first element is used for main effects (default: "%t=%v").
        The second element is used for interactions (default: "%t:%t").
        Use "%t" for the term name and "%v" for its formatted value.
    format_args : dict or None, default None
        A dictionary of additional arguments for formatting values (e.g., {'digits': 3}).
        Common-and currently possible-keys include 'digits' for decimal precision.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.

    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the breakdown visualization.
    """
    style = utils.match_arg(style, ['waterfall', 'barplot'])
    brk_df = breakdown.breakdown.copy()
    catchall_value = 0
    use_catchall = False
    if terms is not None:
        in_terms = brk_df['term'].isin(terms)
        resid = brk_df[~in_terms]['mid'].sum()
        brk_df = brk_df[in_terms].copy()
        if resid != 0:
            catchall_value += resid
            use_catchall = True
    nmax = min(max_nterms, len(brk_df))
    if nmax < len(brk_df):
        resid = brk_df.iloc[max_nterms - 1:]['mid'].sum()
        brk_df = brk_df.head(max_nterms - 1).copy()
        catchall_value += resid
        use_catchall = True
    inputs = pd.DataFrame(breakdown.data).iloc[0]
    values = dict()
    for col, val in inputs.items():
        if isinstance(val, (float, np.number)):
            values[col] = f"{{:.{format_args.get('digits', 4)}g}}".format(val)
        else:
            values[col] = str(val)
    if label_pattern is None or len(label_pattern) < 1:
        label_pattern = ['%t=%v', '%t:%t']
    if len(label_pattern) < 2:
        label_pattern.append('%t:%t')
    labels = []
    for i in range(len(brk_df)):
        term = brk_df.iloc[i]['term']
        tags = str(term).split(':')
        if len(tags) == 1:
            label = label_pattern[0]
            t = tags[0]
            v = values.get(t, "")
            label = label.replace('%t', t).replace('%v', v)
        else:
            label = label_pattern[1]
            for j in range(min(len(tags), 2)):
                t = tags[j]
                v = values.get(t, "")
                label = label.replace('%t', t, 1).replace('%v', v, 1)
        labels.append(label)
    brk_df['term'] = labels
    if use_catchall:
        catchall_row = pd.DataFrame({'term': [catchall], 'mid': [catchall_value]})
        brk_df = pd.concat([brk_df, catchall_row], ignore_index=True)
    brk_df['term'] = pd.Categorical(
        brk_df['term'], categories=brk_df['term'].iloc[::-1]
    )
    if style == 'waterfall':
        intercept = breakdown.intercept
        cs = np.cumsum(np.r_[intercept, brk_df['mid']])
        brk_df['xmin'], brk_df['xmax'] = cs[:-1], cs[1:]
        brk_df['ymin'], brk_df['ymax'] = brk_df['term'].cat.codes + 1 - 0.4, brk_df['term'].cat.codes + 1 + 0.4
        brk_df['ymin2'] = (brk_df['ymin'] - 1).clip(lower=brk_df['ymin'].min())
        p = (
            p9.ggplot(brk_df, p9.aes(y='term'))
            + p9.geom_vline(xintercept=intercept, size=0.5)
            + p9.geom_rect(p9.aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax'), **kwargs)
            + p9.geom_linerange(p9.aes(x='xmax', ymax='ymax', ymin='ymin2'), size=0.5)
            + p9.labs(x='yhat')
            + p9.scale_y_discrete(name="")
        )
    elif style == 'barplot':
        p = (
            p9.ggplot(brk_df, p9.aes(x='term', y='mid'))
            + p9.geom_col(**kwargs)
            + p9.geom_hline(yintercept=0, linetype='dashed', color='#808080')
            + p9.coord_flip()
            + p9.labs(x="")
        )
    if theme is not None:
        theme = pt.color_theme(theme)
        if theme.theme_type == 'qualitative':
            mid_sign = np.where(brk_df['mid'] > 0, '> 0', '< 0')
            p = p + p9.aes(fill=mid_sign) + pt.scale_fill_theme(theme) + p9.labs(fill='mid')
        else:
            p = p + p9.aes(fill='mid') + pt.scale_fill_theme(theme)
    return p


def plot_conditional(
    conditional: MIDConditional,
    style: Literal['ice', 'centered'] = 'ice',
    theme: str | pt.color_theme | None = None,
    var_color: str | None = None,
    dots: bool = True,
    reference: int = 0,
    **kwargs
):
    """Visualize Individual Conditional Expectation (ICE) plots or Centered ICE (c-ICE) plots with plotnine.
    This is a porting function for the R function `midr::ggmid.mid.conditional()`.

    Parameters
    ----------
    conditional : MIDConditional
        A fitted :class:`MIDConditional` object containing the ICE data.
    style : {'ice', 'centered'}, default 'ice'
        The plotting style.
        'ice' plots raw predicted values against the predictor variable.
        'centered' displays the **change in prediction** relative to a `reference` point,
        by subtracting the prediction at the `reference` point for each individual observation.
    theme : str or pt.color_theme or None, default None
        The color theme to use for the line colors.
    var_color : str or None, default None
        The name of a column (from the original data) to map to the color aesthetic of the ICE lines. This helps visualize heterogeneity.
    dots : bool, default True
        If True, plots points for the observed (original) predictions for each sample.
    reference : int, default 0
        The 0-indexed sample point used as the reference prediction for centering when `style='centered'` is used.
    **kwargs : dict
        Additional keyword arguments passed to the main layer of the plot.

    Returns
    -------
    plotnine.ggplot.ggplot
        A plotnine object representing the conditional expectation visualization.
    """
    style = utils.match_arg(style, ['ice', 'centered'])
    variable = conditional.variable
    obs_df = conditional.observed.copy()
    con_df = conditional.conditional.copy()
    if style == 'centered':
        values = conditional.values
        ref = values[min(len(values) - 1, max(0, reference))]
        ref_df = con_df.loc[con_df[variable] == ref, ['.id', 'yhat']].rename(columns={'yhat': 'yref'})
        obs_df = pd.merge(obs_df, ref_df, on='.id')
        con_df = pd.merge(con_df, ref_df, on='.id')
        obs_df['centered yhat'] = obs_df['yhat'] - obs_df['yref']
        con_df['centered yhat'] = con_df['yhat'] - con_df['yref']
    yvar = 'yhat' if style == 'ice' else 'centered yhat'
    p = (
        p9.ggplot(data=obs_df, mapping=p9.aes(x=variable, y=yvar))
        + p9.geom_line(p9.aes(group='.id'), data=con_df, **kwargs)
    )
    if dots:
        p += p9.geom_point()
    if var_color is not None:
        p += p9.aes(color=var_color)
    if theme is not None:
        p += pt.scale_color_theme(theme)
    return p
