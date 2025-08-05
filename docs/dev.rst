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

If you run into problems setting up development environment on any
platform, please `open an issue
<https://github.com/TopoToolbox/pytopotoolbox/issues/new>`_. While we
test TopoToolbox extensively on different platforms, we only develop
on a few, so feedback on how we can make this process as smooth as
possible is greatly appreciated.

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

2. To ensure the compiler is working with 64-bit architecture, that is
   necessary for python, **open 'x64 Native Tools Command Prompt for
   VS 2022'** instead of the '*Developer Command Prompt*' that
   defaults to 32-bit architecture.
3. In the opened command prompt, navigate to the pytopotoolbox
   directory and follow the instructions above for installing the
   development environment.

If you are using Anaconda, you will still need to install the Build
Tools for Visual Studio 2022 as above, but then you should be able to
build pytopotoolbox with pip from the Anaconda Prompt.

If you receive errors like::

   CMake Error: CMAKE_GENERATOR was set but the specified generator doesn't exist. Using CMake default.

You may need to either set it to the correct value with ``set
CMAKE_GENERATOR=Visual Studio 17 2022`` or unset it completely with
``set CMAKE_GENERATOR=``.

Creating a new release of pytopotoolbox
---------------------------------------

To release a new version of pytopotoolbox:

1. Increase the version number in pyproject.toml depending on whether
   this is a patch, minor or major release.
2. Manually draft a release on GitHub. Create a new tag matching the
   version number in pyproject.toml
3. Publish the release. This will trigger our release workflow, which
   will build and upload binary wheels to the GitHub release and to
   PyPi. It will also automatically be archived on Zenodo.
4. Update the CITATION.cff file with the new version number and DOI
   from Zenodo. Because the DOI will not be issued until after the
   release is made, this change must be made AFTER the release is
   issued. Any additional changes to the CITATION.cff file can be made
   at this time.

Pre-Commit Hooks
----------------

We suggest using `pre-commit <https://pre-commit.com/>`_ to run linters and
formatting checks before committing changes. This way, there will be no
suprises when the CI pipeline runs the same checks. If you installed the
requirements.txt pre-commit should already be installed. If not, run:

.. code-block:: bash

   pip install pre-commit

To  install the pre-commit hook, run:

.. code-block:: bash

   pre-commit install

If you want to disable the pre-commit hook, run :

.. code-block:: bash

   pre-commit uninstall

If you want to run the pre-commit checks manually, run:

.. code-block:: bash

   pre-commit run --all-files

The pre-commit-config contains the following hooks:

- Trims trailing whitespace at end of lines
- Ensures files end with a newline and only one
- Validates YAML files for syntax correctness
- Prevents accidentally committing large files
- Running pylint
- Running mypy
- Running nbstripout to clean metadata from notebooks

.. toctree::
   :maxdepth: 1
   :hidden:

   dev/template
   dev/wrapping
