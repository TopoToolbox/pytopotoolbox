@echo off

cd ..\libtopotoolbox\
cmake -B build
cmake --build build

cd ..
copy .\libtopotoolbox\build\src\libtopotoolbox.so .\pytopotoolbox\src\topotoolbox\libtopotoolbox.so

cd .\pytopotoolbox\
py -m pip install --upgrade build
py -m build
