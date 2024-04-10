import os

def get_abs_path(path):
    # Function to generate the absolute path of given string.
    if os.path.isabs(path):
        return path
    else:
        return os.path.abspath(path)
    
def get_libpath():
    pass