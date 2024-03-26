# pytopotoolbox

Python interface to TopoToolbox.

## Guide

Temporary guide for the functionality of the package. For Windows replace "python3" with "py".

### Generate distribution archives

Generates a .whl and .tar.gz which then can be used to install package with pip.
These Files can be distributet with PyPi or downloaded directly.

```bash
cd path/to/pytopotoolbox
python3 -m pip install --upgrade build
python3 -m build
```

### Installing distribution archives

Use "--force-reinstall" to overwrite previous install.

```bash
pip install dist_name.whl
```

### Installing from repository

If want to install the package direktly from the repository, without first generating a ".whl" file.

```bash
cd path/to/pytopotoolbox
pip install .
```

### Unittest

How to run a single unittest:

```bash
cd path/to/pytopotoolbox
python3 -m tests.filename 
```
