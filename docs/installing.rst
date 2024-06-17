How to install
==============

Generating/Installing distribution archives
-------------------------------------------

For any operating system, install the following:

- **Python**
- **pip**
- **Git** (only when *building* the package yourself)

Linux
~~~~~

**Installing from .whl file**

Make sure to choose the appropriate file for your OS. For Linux, the file name should contain something like: `linux_x86_64`

.. code-block:: bash

    pip install dist_name.whl

**Installing directly from the repository:**

.. code-block:: bash

    cd path/to/pytopotoolbox
    pip install .

**Generating distribution archives**

.. code-block:: bash

    cd path/to/pytopotoolbox
    python3 -m pip install --upgrade build
    python3 -m build

Windows
~~~~~~~

**Installing from .whl file**

Make sure to choose the appropriate file for your OS. For Windows, the file name should contain something like: `win_amd64`.

.. code-block:: bash

    pip install dist_name.whl

**Installing directly from the repository:**

Since there are C/C++ files that need to be compiled in order to build the package, there are a few extra steps to take.

1. Install the `Developer Command Prompt for VS 2022 <https://visualstudio.microsoft.com/downloads/>`_.
    - Scroll down to '*All Downloads*'
    - Open '*Tools for Visual Studio*'
    - Download '*Build Tools for Visual Studio 2022*'
    - Install it while including the '*Desktop development with C++*' workload
2. To ensure the compiler is working with 64-bit architecture, that is necessary for python, **open 'x64 Native Tools Command Prompt for VS 2022'** instead of the '*Developer Command Prompt*' that defaults to 32-bit architecture.
3. In the newly opened terminal, run:

    .. code-block:: bash

        cd path\to\pytopotoolbox
        pip install .

**Generating distribution archives**

Open the 'x64 Native Tools Command Prompt for VS 2022' Terminal and run:

.. code-block:: bash

    cd path\to\pytopotoolbox
    py -m pip install --upgrade build
    py -m build

Mac
~~~

*work in progress*
