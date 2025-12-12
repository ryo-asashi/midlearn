# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Imports

import sys
import datetime
from pathlib import Path

# File Path : (root)/docs/source/conf.py

FILE_PATH = Path(__file__).resolve()
ROOT_PATH = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(ROOT_PATH / "src"))

# Project information

project = "midlearn"
author = "Ryoichi Asashiba"
copyright = f"{datetime.datetime.now().year}, Ryoichi Asashiba"
version = (ROOT_PATH / "VERSION.txt").read_text(encoding="utf-8").strip()
release = version

# General Configuration

needs_sphinx = "2.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "myst_nb",
]

## Options for source files

exclude_patterns = []

## Options for templating

templates_path = ["_templates"]

# Builder options

## Options for HTML output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "includehidden": False,
    "logo_only": True,
}
html_logo = "../logo/logo.svg"
html_favicon = "../logo/logo.svg"
html_js_files = ["custom.js"]
html_static_path = ["_static"]

# Extension options

# autosummary
# https://www.sphinx-doc.org/ja/master/usage/extensions/autosummary.html
autosummary_generate = True

# intersphinx
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# myst-nb
# https://myst-nb.readthedocs.io/en/latest/configuration.html
nb_execution_mode = "off"
