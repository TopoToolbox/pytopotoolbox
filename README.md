# pytopotoolbox

Python interface to TopoToolbox.

## Build distribution archives

To automatically build distribution archives for the python version of TopoToolbox. First make sure that both the `pytopotoolbox` and `libtopotoolbox` repo are downloaded and in the same folder like so:

```bash
.
├── libtopotoolbox
└── pytopotoolbox
```

### for Windows

```bash
cd .\path\to\pytopotoolbox\
build.bat
pip install .\dist\dist_name.whl
```

### for mac/Linux

To execute the installation bash script `build.sh`, you will need to give the file execution rights first.

```bash
cd path/to/pytopotoolbox
chmod +x build.sh
./build.sh
```

## Guide

Temporary guide for the functionality of the package. For Windows replace "python3" with "py".

### Generate distribution archives

Generates a .whl and .tar.gz which then can be used to install package with pip.
These Files can be distributed with PyPi or downloaded directly.

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

If want to install the package directly from the repository without first generating a ".whl" file.

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
