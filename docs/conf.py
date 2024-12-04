import os
import sys

import sphinx_book_theme

import topotoolbox


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'nbsphinx',
    'myst_parser'
]

project = 'TopoToolbox'
copyright = '2024, TopoToolbox Team'
author = 'TopoToolbox Team'
release = '3.0.1'

# -- General configuration ---------------------------------------------------

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.md']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = 'logo.png'

html_context = {
    "default_mode": "light",
}

html_theme_options = {
    "repository_url": "https://github.com/TopoToolbox/pytopotoolbox",
    "use_repository_button": True,
}

# -- Options for nbsphinx ----------------------------------------------------

nbsphinx_allow_errors = True

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'inherited-members': True,
    'show-inheritance': True,
}

autosummary_generate = True  # Enable autosummary to generate stub files

# -- Options for nbgallery ---------------------------------------------------

nbsphinx_thumbnails = {
    'examples/downloading': '_static/thumbnails/placeholder.png',
    'examples/flowobject': '_static/thumbnails/placeholder.png',
    'examples/streamobject': '_static/thumbnails/placeholder.png',
}
