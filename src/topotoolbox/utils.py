"""Provide general utility functions for topotoolbox.
"""
import sys
import os
from shutil import rmtree
from urllib.request import urlopen, urlretrieve

from .grid_object import GridObject

__all__ = ["load_dem", "get_dem_names", "read_tif",
           "get_cache_contents", "clear_cache"]

DEM_SOURCE = "https://raw.githubusercontent.com/TopoToolbox/DEMs/master"
DEM_NAMES = f"{DEM_SOURCE}/dem_names.txt"


def read_tif(path: str) -> GridObject:
    """Generate a new GridObject from a .tif file.

    Args:
        path (str): path to .tif file.

    Returns:
        GridObject: A new GridObject of the .tif file.
    """
    return GridObject(path)


def get_dem_names() -> list[str]:
    """Returns a list of provided example Digital Elevation Models (DEMs).
    Requires internet connection to download available names.

    Returns:
        list[str]: A list of strings, where each string is the name of a DEM.
    """
    with urlopen(DEM_NAMES) as dem_names:
        dem_names = dem_names.read().decode()

    return dem_names.splitlines()


def load_dem(dem: str, cache: bool = True) -> GridObject:
    """Downloads DEM from TopoToolbox/DEMs repository.
    Find possible names by using 'get_dem_names()'

    Args:
        dem (str): Name of dem about to be downloaded
        cache (bool, optional): If true the dem will be cached.
        Defaults to True.

    Returns:
        GridObject: A GridObject generated from the downloaded dem.
    """
    if dem not in get_dem_names():
        err = ("Selected DEM has to be selected from the provided examples." +
               " See which DEMs are available by using 'get_dem_names()'.")
        raise ValueError(err)

    url = f"{DEM_SOURCE}/{dem}.tif"

    if cache:
        cache_path = os.path.join(get_save_location(), f"{dem}.tif")

        if not os.path.exists(cache_path):
            urlretrieve(url, cache_path)

        full_path = cache_path
    else:
        full_path = url

    dem = GridObject(full_path)

    return dem


def get_save_location() -> str:
    """Generates filepath to file saved in cache.

    Returns:
        str: filepath to file saved in cache.
    """
    system = sys.platform

    if system == "win32":
        path = os.getenv('LOCALAPPDATA')
        path = os.path.join(path, "topotoolbox")

    elif system == 'darwin':
        path = os.path.expanduser('~/Library/Caches')
        path = os.path.join(path, "topotoolbox")

    else:
        path = os.path.expanduser('~/.cache')
        path = os.path.join(path, "topotoolbox")

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def clear_cache(filename: str = None) -> None:
    """Deletes the cache directory and it's contents. Can also delete a single
    file when using the argument filename. To get the contents of your cache,
    use 'get_cache_contents()'

    Args:
        filename (str, optional): Add a filename if only one specific file is
        to be deleted. Defaults to None.
    """
    path = get_save_location()

    if filename:
        path = os.path.join(path, filename)

    if os.path.exists(path):
        if os.path.isdir(path):
            # using shutil.rmtree since os.rmdir requires dir to be empty.
            rmtree(path)
        else:
            os.remove(path)
    else:
        print("Cache directory or file does not exist.")


def get_cache_contents() -> (list[str] | None):
    """Returns the contents of the cache directory.

    Returns:
        list[str]: List of all files in the topotoolbox cache. If cache does
        not exist, None is returned.
    """
    path = get_save_location()

    if os.path.exists(path):
        return os.listdir(path)

    print("Cache directory does not exist.")
    return None
