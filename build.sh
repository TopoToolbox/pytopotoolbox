#!/bin/bash

# compile C++/C library
cd ../libtopotoolbox/
cmake -B build
cmake --build build

# move libtopotoolbox.so to pytopotoolbox
cd ..
cp libtopotoolbox/build/src/libtopotoolbox.so pytopotoolbox/src/topotoolbox/libtopotoolbox.so

# build .whl file
cd pytopotoolbox/
python3 -m pip install --upgrade build
python3 -m build

# Prompt if user wants to install .whl file
read -p "Install the generated python package with pip? (y/n): " choice
if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    whl_file=$(find dist/ -type f -name "*.whl")
    pip install "$whl_file" --ignore-installed 
    # add "--no-deps" to skip installing dependencies
    echo "Installation completed."

elif [ "$choice" = "n" ] || [ "$choice" = "N" ]; then
    echo "Installation skipped."

else
    echo "Invalid choice. Installation skipped."
fi

