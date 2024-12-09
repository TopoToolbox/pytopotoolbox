Developer Documentation
=======================

To set up pytopotoolbox for development, first make sure you have `set
up Git
<https://docs.github.com/en/get-started/getting-started-with-git>`_,
`fork the repository on GitHub
<https://github.com/TopoToolbox/pytopotoolbox/fork>`_ then use ``git
clone https://github.com/$YOUR_USERNAME/pytopotoolbox`` to download
your fork of the repository, where you have replaced ``$YOUR_USERNAME``
with your GitHub username.

To ensure you have a development environment that is consistent with
other developers and with our continuous integration and testing
setup, it is a good idea to set up a `virtual environment
<https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_
in Python. You can then install the developer environment by running

.. code-block:: bash
		
    pip install -r requirements.txt .

within the top-level directory of the repository.

Now you can make changes within pytopotoolbox. To rebuild
pytopotoolbox after making changes, you'll need to run

.. code-block:: bash
		
    pip install .

You can run the tests with

.. code-block:: bash
		
   python -m pytest

and the linter and type checks with

.. code-block:: bash
		
    pylint --rcfile=pyproject.toml src/topotoolbox
    mypy --ignore-missing-imports src/topotoolbox

It is a good idea to run the linter and type checks before making a
pull request to pytopotoolbox, because failing lints or type checks
will cause a test failure that must be fixed before your contribution
can be accepted.


Development environment on Windows
----------------------------------

pytopotoolbox requires C/C++ files to be compiled. If you are
developing on Windows, there are a few extra steps to ensure that you
have access to the Windows C/C++ toolchain.

1. Install the `Developer Command Prompt for VS 2022 <https://visualstudio.microsoft.com/downloads/>`_.

   * Scroll down to '*All Downloads*'
   * open '*Tools for Visual Studio*'
   * download '*Build Tools for Visual Studio 2022*'
   * install it while including the '*Desktop development with C++*' workload

2. To ensure the compiler is working with 64-bit architecture, that is necessary for python, **open 'x64 Native Tools Command Prompt for VS 2022'** instead of the '*Developer Command Prompt*' that defaults to 32-bit architecture.
3. In the opened command prompt, navigate to the pytopotoolbox directory and follow the instructions above for installing the development environment.
