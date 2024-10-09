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
    '_temp/excesstopography': '_static/thumbnails/placeholder.png',
    '_temp/magicfunctions': '_static/thumbnails/placeholder.png',
    '_temp/plotting': '_static/thumbnails/placeholder.png',
    '_temp/downloading': '_static/thumbnails/placeholder.png',
    '_temp/flowobject': '_static/thumbnails/placeholder.png',
    '_temp/streamobject': '_static/thumbnails/placeholder.png',
}
