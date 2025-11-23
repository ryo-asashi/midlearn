# src/midlearn/__init__.py

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("midlearn")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .api import MIDRegressor, MIDExplainer, MIDImportance, MIDBreakdown, MIDConditional
from .plotting import plot_effect, plot_importance, plot_breakdown, plot_conditional
from .plotting_theme import color_theme, scale_color_theme, scale_fill_theme, scale_colour_theme

__all__ = [
    # from api
    "MIDRegressor",
    "MIDExplainer",
    "MIDImportance",
    "MIDBreakdown",
    "MIDConditional",
    # from plotting
    "plot_effect",
    "plot_importance",
    "plot_breakdown",
    "plot_conditional",
    # from plotting_theme
    "color_theme",
    "scale_color_theme",
    "scale_colour_theme",
    "scale_fill_theme"
]
