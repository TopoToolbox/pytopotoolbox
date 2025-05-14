Welcome to TopoToolbox's Python Documentation!
==============================================

**TopoToolbox** is a Python library that provides a set of functions and classes that support the analysis of relief and flow pathways in digital elevation models (DEMs). The major aim of TopoToolbox is to offer helpful analytical GIS utilities in a non-GIS environment to support the simultaneous application of GIS-specific and other quantitative methods.

This Python library is based on the `TopoToolbox <https://topotoolbox.wordpress.com/>`_ for Matlab and uses the API provided by `libtopotoolbox <https://topotoolbox.github.io/libtopotoolbox/>`_ to compute efficiently.

Installing and Getting Started
------------------------------

TopoToolbox is on PyPI, so you can run

.. code-block:: bash

    pip install --upgrade topotoolbox

to obtain the latest version.

Once you have the package installed, the :doc:`tutorial <tutorial/tutorial>` provides a basic walkthrough of loading and displaying a digital elevation model with TopoToolbox.

Further examples of TopoToolbox functionality can be found in the :doc:`example notebooks <examples>`, which can also be downloaded from the `GitHub repository <https://github.com/TopoToolbox/pytopotoolbox/tree/main/examples>`_.

API Documentation
-----------------

For further documentation regarding the functionality of this package, check out the :doc:`API documentation<api>`.

Contributing
------------

If you would like to contribute to pytopotoolbox, refer to the :doc:`CONTRIBUTING` page. A great way to get started is to :doc:`create a notebook <dev/template>` for our :doc:`examples` gallery. To get a better understanding of how we wrap libtopotoolbox functions for use in Python, the :doc:`dev/wrapping` page will help.

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting Started with TopoToolbox <tutorial/tutorial>
   Examples <examples>
   API <api>
   Contribution Guidelines <CONTRIBUTING>
   Developer Documentation <dev>

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
