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


sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "pfh.glidersim"
copyright = "2022, Peter Heatwole"
author = "Peter Heatwole"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]

autodoc_default_options = {
    "inherited-members": True,
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__call__, __init__",
}

intersphinx_mapping = {
    "thesis": ("https://pfheatwole.github.io/thesis", None),
}

mathjax3_config = {
    "tex": {
        "macros": {
            "defas": r"\stackrel{\mathrm{def}}{=}",
            "vec": [r"\mathbf{#1}", 1],
            "mat": [r"\mathbf{#1}", 1],
            "crossmat": [r"\left[{#1}\right]^{\times}", 1],
        },
    },
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = project  # Override "<project>'s documentation"
html_logo = "_static/hook3_vectorized_opt.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "announcement": "<b>This documentation is a work-in-progress.</b>",
}
