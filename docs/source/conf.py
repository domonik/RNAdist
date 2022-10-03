# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
FPATH = os.path.abspath(__file__)
__RNADISTPATH__ = os.path.abspath(os.path.join(FPATH, "../../"))
__RNADISTPATH__ = os.path.abspath("../../")
CP_PATH = os.path.join(__RNADISTPATH__, "CPExpectedDistance")
assert os.path.exists(CP_PATH)
sys.path.insert(1, __RNADISTPATH__)
sys.path.append(CP_PATH)
from RNAdist import _version

# -- Project information -----------------------------------------------------

project = 'RNAdist'
copyright = '2022, Domonik'
author = 'Domonik'

# The full version, including alpha/beta/rc tags
__version__ = _version.get_versions()["version"]
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    'sphinxarg.ext',
    'sphinx_design'
]
autosummary_generate = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # Toc options
    "collapse_navigation": True,
    "navigation_depth": 4,
    "logo": {
        "image_light": "RNAdist4.svg",
        "image_dark": "RNAdist4_dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/domonik/RNAdist",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },

    ]
}
mathjax3_config = {'chtml': {'displayAlign': 'left'}}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'args.css',
]
#html_logo = "_static/RNAdist4.svg"
html_favicon = '_static/RNAdist_tabbar_dark.svg'

autosummary_mock_imports = [
    'RNA',
    "RNAdist.NNModels.nn_helpers",
    "RNAdist.DPModels._dp_calulations",
    "RNAdist.DPModels._dp_calculations",
    "CPExpectedDistance.p_expected_distance",
    "networkx",
    "plotly",
    "numpy",
    "numpy.core.multiarray",
    "Bio",
    "pandas",
    "torch",
    "ConfigSpace",
    "smac",
    "dash_bootstrap_components",
    "dash",
    "plotly.colors.qualitative.Light24"

]