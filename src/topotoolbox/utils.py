"""Provide general utility functions for topotoolbox.
"""
import sys
import os

from urllib.request import urlopen, urlretrieve

from .grid_object import GridObject

__all__ = ["load_dem", "get_dem_names", "read_tif"]

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


def load_dem(dem: str, cache=True) -> GridObject:
    """Downloads DEM from wschwanghart/DEMs repository. 
    Find possible names by using 'get_dem_names()'

    Args:
        dem (str): Name of dem about to be downloaded
        cache (bool, optional): If true the dem will be cached. 
        Defaults to True.
        data_home (str, optional): optional name of cache. Defaults to None.

    Returns:
        GridObject: A GridObject generated from the downloaded dem.
    """
    if dem not in get_dem_names():
        err = ("DEM has to be chosen from the list of available DEMs."
               "Use 'get_dem_names()' to generate list of all available DEMs")
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

    Args:
        data_home (str, optional): name of directory in cache. Defaults to None
        which results in "topotoolbox" as name.

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
