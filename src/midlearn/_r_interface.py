# src/midlearn/_r_interface.py

import re
import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import conversion, pandas2ri, numpy2ri
from rpy2.robjects.packages import importr, PackageNotInstalledError
from rpy2.rinterface_lib.embedded import RRuntimeError

from .exceptions import RPackageError, RExecutionError

# Setup

stats = importr('stats')
utils = importr('utils')
grDevices = importr('grDevices')

try:
    midr = importr('midr')
except (RRuntimeError, PackageNotInstalledError):
    try:
        print("R Package 'midr' not found. Attempting to install...")
        utils.install_packages('midr')
        midr = importr('midr')
    except Exception as e:
        raise RPackageError(
            f"Failed to install the R package midr. Please install it manually. Original error: {e}"
        )



# Custom Converter

cv = conversion.Converter('custom_converter', template=ro.default_converter)

@cv.rpy2py.register(ro.ListVector)
def _(obj):
    return obj

@cv.py2rpy.register(dict)
def _(obj):
    return ro.ListVector(obj)

@cv.py2rpy.register(list)
def _(obj):
    return _as_r_vector(obj)

@cv.py2rpy.register(type(None))
def _(obj):
    return ro.NULL



# Wrapper Functions

def _call_r_interpret(
    X,
    y,
    sample_weight=None,
    params_main=None,
    params_inter=None,
    penalty=0,
    link=None,
    kernel_type=[1,1],
    encoding_frames=dict(),
    model_terms=None,
    singular_ok=False,
    mode=1,
    method=None,
    centering_penalty=1e06,
    na_action='na.omit',
    verbosity=1,
    split='quantile',
    digits=None,
    lump='none',
    others='others',
    sep='>',
    max_nelements=1e+09,
    nil=1e-07,
    tol=1e-07,
    **kwargs
) -> object:
    """ Wrapper function for midr::interpret.default() """
    k_list = [
        ro.NA_Integer if params_main is None else int(params_main),
        ro.NA_Integer if params_inter is None else int(params_inter)
    ]
    r_kwargs = {
        'object': ro.NULL,
        'x': _sanitize_columns(X),
        'y': y,
        'weights': sample_weight,
        'link': ro.NULL if link is None else link,
        'k': ro.IntVector(k_list),
        'type': kernel_type,
        'terms': ro.Formula(model_terms) if isinstance(model_terms, str) else model_terms,
        'singular.ok': singular_ok,
        'mode': mode,
        'method': ro.NULL if method is None else method,
        'lambda': penalty,
        'kappa': centering_penalty,
        'na.action': na_action,
        'verbosity': verbosity,
        'frames': encoding_frames,
        'split': split,
        'digits': ro.NULL if digits is None else digits,
        'lump': lump,
        'others': others,
        'sep': sep,
        'max.nelements': max_nelements,
        'nil': nil,
        'tol': tol,
        **kwargs
    }
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = midr.interpret(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's midr::interpret() function: {e}")


def _call_r_predict(
    r_object: ro.ListVector,
    X: pd.DataFrame,
    output_type: str = 'response',
    terms: list[str] | None = None,
    **kwargs
) -> np.ndarray:
    """ Wrapper function for stats::predict() """
    r_kwargs = {
        'object': r_object,
        'newdata': _sanitize_columns(X),
        'type': output_type,
        **kwargs
    }
    if terms is not None:
        r_kwargs['terms'] = _as_r_vector(terms, 'character')
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = stats.predict(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's stats::predict() function: {e}")


def _call_r_mid_terms(
    r_object: ro.ListVector,
    **kwargs
) -> np.ndarray:
    """ Wrapper function for midr::mid.terms() """
    r_kwargs = {
        'object': r_object,
        **kwargs
    }
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = midr.mid_terms(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's midr::mid_terms() function: {e}")


def _call_r_mid_effect(
    r_object: ro.ListVector,
    term: str,
    x: np.ndarray | pd.DataFrame,
    y: np.ndarray | None = None
) -> np.ndarray:
    """ Wrapper function for midr::mid.effect() """
    r_kwargs = {
        'object': r_object,
        'term': term,
        'x': x,
        'y': y
    }
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = midr.mid_effect(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's midr::mid_effect() function: {e}")


def _call_r_mid_importance(
    r_object: ro.ListVector,
    data: pd.DataFrame | None = None,
    **kwargs
) -> object:
    """ Wrapper function for midr::mid.importance() """
    r_kwargs = {
        'object': r_object,
        'data': ro.NULL if data is None else _sanitize_columns(data),
        **kwargs
    }
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = midr.mid_importance(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's midr::mid.importance() function: {e}")


def _call_r_mid_breakdown(
    r_object: ro.ListVector,
    data: pd.DataFrame | None = None,
    row: int | None = None,
    **kwargs
) -> object:
    """ Wrapper function for midr::mid.breakdown() """
    if data is not None and data.shape[0] > 1:
        data = _sanitize_columns(data)
        data = data.iloc[[row if row is not None else 0]]
        row = None
    r_kwargs = {
        'object': r_object,
        'data': data,
        'row': ro.NULL if row is None else (row + 1),
        **kwargs
    }
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = midr.mid_breakdown(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's midr::mid.breakdown() function: {e}")


def _call_r_mid_conditional(
    r_object: ro.ListVector,
    variable: str,
    pred_type: str,
    data: pd.DataFrame | None = None,
    **kwargs
) -> object:
    """ Wrapper function for midr::mid.conditional() """
    r_kwargs = {
        'object': r_object,
        'variable': variable,
        'data': ro.NULL if data is None else _sanitize_columns(data),
        'type': pred_type,
        **kwargs
    }
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = midr.mid_conditional(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's midr::mid.conditional() function: {e}")

def _call_r_transform(env, x, **kwargs):
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = env['transform'](x, **kwargs)
        if isinstance(res, ro.FactorVector):
            return pd.Categorical.from_codes(
                codes=np.asarray(res).astype(int) - 1,
                categories=list(res.levels),
                ordered=res.isordered
            )
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error during R function call: {e}")


def _extract_and_convert(
    r_object: ro.ListVector,
    name: str
) -> object:
    try:
        idx = list(r_object.names).index(name)
        element = r_object[idx]
    except ValueError:
        raise KeyError(f"Name '{name}' not found in the R object.") from None
    # data.frame to pd.DataFrame
    if 'data.frame' in list(element.rclass):
        res = pandas2ri.rpy2py(element)
        for i, colname in enumerate(list(element.names)):
            if not isinstance(element[i], ro.FactorVector):
                continue
            res[colname] = pd.Categorical.from_codes(
                codes=res[colname].astype(int) - 1,
                categories=list(element[i].levels),
                ordered=element[i].isordered
            )
        return res
    # matrix to pd.DataFrame
    if hasattr(element, 'colnames') and element.colnames is not ro.NULL:
        res = pd.DataFrame(np.asarray(element), columns=list(element.colnames))
        return res
    # (named) list to ListVector
    if isinstance(element, ro.ListVector):
        return element
    # others to np.ndarray if applicable
    try:
        return numpy2ri.rpy2py(element)
    except Exception:
        return element


def _call_r_color_theme(
    theme,
    theme_type: str | None,
    **kwargs
) -> object:
    """ Wrapper function for midr::color.theme() """
    r_kwargs = {
        'type': theme_type,
        **kwargs
    }
    if isinstance(theme, str):
        r_kwargs['object'] = theme
    elif isinstance(theme, list):
        r_kwargs['object'] = _as_r_vector(theme, mode='character')
        r_kwargs.setdefault('name', 'newtheme')
    else:
        r_kwargs['object'] = theme._obj
    try:
        with conversion.localconverter(pandas2ri.converter + numpy2ri.converter + cv):
            res = midr.color_theme(**r_kwargs)
        return res
    except RRuntimeError as e:
        raise RExecutionError(f"Error in R's midr::color_theme() function: {e}")


def _as_r_vector(
    obj,
    mode: str | None = None
) -> object:
    if mode is None:
        if all(isinstance(x, str) for x in obj):
            mode = 'character'
        elif all(isinstance(x, bool) for x in obj):
            mode = 'logical'
        elif all(isinstance(x, int) for x in obj):
            mode = 'integer'
        elif all(isinstance(x, (int, float)) for x in obj):
            mode = 'numeric'
        else:
            mode = 'list'
    if mode == 'numeric':
        return ro.FloatVector(obj)
    if mode == 'integer':
        return ro.IntVector(obj)
    if mode == 'character':
        return ro.StrVector(obj)
    if mode == 'factor':
        return ro.FactorVector(obj)
    if mode == 'logical':
        return ro.BoolVector(obj)
    if mode == 'list':
        return ro.ListVector(obj)
    raise ValueError(f"Invalid mode '{mode}'.")


_R_COLOR_MAP = None

def _convert_r_color(color: str) -> str:
    global _R_COLOR_MAP
    if _R_COLOR_MAP is None:
        colors = grDevices.colors()
        mat = np.array(grDevices.col2rgb(colors), dtype=np.uint8).T
        hexrgb = [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in mat]
        _R_COLOR_MAP = {color: rgb for color, rgb in zip(colors, hexrgb)}
    return _R_COLOR_MAP.get(color, color)

def _convert_r_name(name: str) -> str:
    r_reserved = {
        "if", "else", "repeat", "while", "function", "for", "in", "next", "break", 
        "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA",
        "NA_integer_", "NA_real_", "NA_complex_", "NA_character_"
    }
    name = str(name)
    if not name:
        return 'X'
    name = re.sub(r'[^a-zA-Z0-9._]', '.', name)
    if re.match(r'^[0-9]|^\.[0-9]', name):
        name = 'X' + name
    if name in r_reserved:
        name = '.' + name
    return name

def _sanitize_columns(data: pd.DataFrame):
    p_names = data.columns.tolist()
    r_names = [_convert_r_name(c) for c in p_names]
    seen = {}
    unique_names = []
    for name in r_names:
        if name not in seen:
            unique_names.append(name)
            seen[name] = 0
        else:
            seen[name] += 1
            unique_names.append(f"{name}.{seen[name]}")
    data = data.copy()
    data.columns = unique_names
    return data
