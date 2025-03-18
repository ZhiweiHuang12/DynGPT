# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path
from pygments.styles import get_all_styles

# HERE = Path(__file__).parent
# sys.path[:0] = [str(HERE.parent)]
sys.path.insert(0, os.path.abspath('../'))

project = 'DynGPT'
copyright = '2025, ZhiweiHuang'
author = 'ZhiweiHuang'
release = '0.1.0'
pygments_style = 'default'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = [
    'nbsphinx',
    'sphinx.ext.mathjax', 
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary'
]
nbsphinx_allow_errors = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
nb_execution_mode = "off"

html_logo = "_static/image/logo.png"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
highlight_language = 'python'
html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#357473",
        "color-brand-content": "#357473",
    },
}
autosummary_generate = True 

html_css_files = [
    "css/override.css", 
    "css/custom.css"
]
