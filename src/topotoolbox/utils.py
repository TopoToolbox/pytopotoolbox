import sys
import os

from urllib.request import urlretrieve

from .grid_object import GridObject

__all__ = ["load_dem", "get_dem_names"]

DEM_SOURCE = "https://raw.githubusercontent.com/wschwanghart/DEMs/master"
DEM_NAMES = ['kunashiri', 'perfectworld',
             'taalvolcano', 'taiwan', 'tibet', 'kedarnath']


def get_dem_names():
    """Returns a list of provided example Digital Elevation Models (DEMs).

    Returns:
        list: A list of strings, where each string is the name of a DEM.

    Note:
        If a file with all names is added to the 'wschwanghart/DEMs' repository, 
        the DEM_NAMES list could be generated dynamically from that file. This would 
        ensure the list remains up-to-date if the files ever change.
    """
    return DEM_NAMES


def load_dem(dem: str, cache=True, data_home=None):
    """Downloads DEM from wschwanghart/DEMs reposetory. 
    Find possible names by using 'get_dem_names()'

    Args:
        dem (str): Name of dem about to be downloaded
        cache (bool, optional): If true the dem will be cached. Defaults to True.
        data_home (str, optional): optional name of cache. Defaults to None.

    Returns:
        GridObject: A GridObject generated from the downloaded dem.
    """

    url = f"{DEM_SOURCE}/{dem}.tif"

    if cache:
        cache_path = os.path.join(get_save_location(data_home), f"{dem}.tif")

        # only download if not already downloaded
        if not os.path.exists(cache_path):
            urlretrieve(url, cache_path)

        full_path = cache_path
    else:
        full_path = url

    dem = GridObject(full_path)

    return dem


def get_save_location(data_home=None):
    """Genrates filepath to file saved in cache

    Args:
        data_home (str, optional): name of directory in cache. Defaults to None
        which results in "topotoolbox" as name.

    Returns:
        str: filepath to file saved in cache.
    """
    if data_home is None:
        data_home = cache_dir("topotoolbox")

    # turn ~ shorthand to full path
    data_home = os.path.expanduser(data_home)

    # create path if it dosnt exsist already
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home


def cache_dir(appname=None):
    """Generates filepath for cache directory.

    Args:
        appname (str, optional): Optional name for cache reposetory. Defaults to None.

    Returns:
        str: Path to cache directory.
    """
    system = sys.platform

    if system == "win32":
        path = os.getenv('LOCALAPPDATA')
        if appname:
            path = os.path.join(path, appname)

    elif system == 'darwin':
        path = os.path.expanduser('~/Library/Caches')
        if appname:
            path = os.path.join(path, appname)

    else:
        path = os.getenv(os.path.expanduser('~/.cache'))
        if appname:
            path = os.path.join(path, appname)

    return path
