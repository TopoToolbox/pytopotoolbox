"""Provide general utility functions for topotoolbox.
"""
import sys
import os
import random
from shutil import rmtree
from urllib.request import urlopen, urlretrieve

import rasterio
import numpy as np

from .grid_object import GridObject

__all__ = ["load_dem", "get_dem_names", "read_tif", "gen_random", "write_tif",
           "gen_random_bool", "get_cache_contents", "clear_cache"]

DEM_SOURCE = "https://raw.githubusercontent.com/TopoToolbox/DEMs/master"
DEM_NAMES = f"{DEM_SOURCE}/dem_names.txt"


def write_tif(dem: GridObject, path: str) -> None:
    """
    Write a GridObject instance to a GeoTIFF file.

    Parameters
    ----------
    dem : GridObject
        The GridObject instance to be written to a GeoTIFF file.
    path : str
        The file path where the GeoTIFF will be saved.

    Raises
    ------
    TypeError
        If `dem` is not an instance of GridObject.

    Examples
    --------
    >>> dem = topotoolbox.load_dem('taiwan')
    >>> topotoolbox.write_tif(dem, 'dem.tif')
    """

    if not isinstance(dem, GridObject):
        err = "The provided dem is not an instance of GridObject."
        raise TypeError(err) from None

    with rasterio.open(
            fp=path,
            mode='w',
            count=1,
            driver='GTiff',
            height=dem.rows,
            width=dem.columns,
            dtype=np.float32,
            crs=dem.crs,
            transform=dem.transform
    ) as dataset:
        dataset.write(dem.z, 1)

def read_tif(path: str) -> GridObject:
    """Generate a new GridObject from a .tif file.

    Parameters
    ----------
    path : str
        path to .tif file.

    Returns
    -------
    GridObject
        A new GridObject of the .tif file.
    """

    grid = GridObject()

    if path is not None:
        try:
            dataset = rasterio.open(path)

        except TypeError as err:
            raise TypeError(err) from None
        except Exception as err:
            raise ValueError(err) from None

        grid.path = path
        grid.name = os.path.splitext(os.path.basename(grid.path))[0]

        grid.z = dataset.read(1).astype(np.float32, order='F')

        grid.cellsize = dataset.res[0]
        grid.bounds = dataset.bounds
        grid.transform = dataset.transform
        grid.crs = dataset.crs

    return grid


def gen_random(hillsize: int = 24, rows: int = 128, columns: int = 128,
               cellsize: float = 10.0, seed: int = 3,
               name: str = 'random grid') -> 'GridObject':
    """Generate a GridObject instance that is generated with OpenSimplex noise.

    Parameters
    ----------
    hillsize : int, optional
        Controls the "smoothness" of the generated terrain. Defaults to 24.
    rows : int, optional
        Number of rows. Defaults to 128.
    columns : int, optional
        Number of columns. Defaults to 128.
    cellsize : float, optional
        Size of each cell in the grid. Defaults to 10.0.
    seed : int, optional
        Seed for the terrain generation. Defaults to 3
    name : str, optional
        Name for the generated GridObject. Defaults to 'random grid'

    Raises
    ------
    ImportError
        If OpenSimplex has not been installed.

    Returns
    -------
    GridObject
        An instance of GridObject with randomly generated values.
    """
    try:
        import opensimplex as simplex  # pylint: disable=C0415

    except ImportError:
        err = ("For gen_random to work, use \"pip install topotool" +
               "box[opensimplex]\" or \"pip install .[opensimplex]\"")
        raise ImportError(err) from None

    noise_array = np.empty((rows, columns), dtype=np.float32, order='F')

    simplex.seed(seed)
    for y in range(0, rows):
        for x in range(0, columns):
            value = simplex.noise4(x / hillsize, y / hillsize, 0.0, 0.0)
            color = int((value + 1) * 128)
            noise_array[y, x] = color

    grid = GridObject()

    grid.z = noise_array
    grid.cellsize = cellsize
    grid.name = name
    return grid


def gen_random_bool(
        rows: int = 32, columns: int = 32, cellsize: float = 10.0,
        name: str = 'random grid') -> 'GridObject':
    """Generate a GridObject instance that contains only randomly generated
    Boolean values.

    Parameters
    ----------
    rows : int, optional
        Number of rows. Defaults to 32.
    columns : int, optional
        Number of columns. Defaults to 32.
    cellsize : float, optional
        Size of each cell in the grid. Defaults to 10.0.

    Returns
    -------
    GridObject
        An instance of GridObject with randomly generated Boolean values.
    """
    bool_array = np.empty((rows, columns), dtype=np.float32)

    for y in range(0, rows):
        for x in range(0, columns):
            bool_array[x][y] = random.choice([0, 1])

    grid = GridObject()

    grid.path = ''
    grid.z = bool_array
    grid.cellsize = cellsize
    grid.name = name

    return grid


def get_dem_names() -> list[str]:
    """Returns a list of provided example Digital Elevation Models (DEMs).
    Requires internet connection to download available names.

    Returns
    -------
    list[str]
        A list of strings, where each string is the name of a DEM.
    """
    with urlopen(DEM_NAMES) as dem_names:
        dem_names = dem_names.read().decode()

    return dem_names.splitlines()


def load_dem(dem: str, cache: bool = True) -> GridObject:
    """Downloads a DEM from the TopoToolbox/DEMs repository.
    Find possible names by using 'get_dem_names()'.

    Parameters
    ----------
    dem : str
        Name of the DEM to be downloaded.
    cache : bool, optional
        If true, the DEM will be cached. Defaults to True.

    Returns
    -------
    GridObject
        A GridObject generated from the downloaded DEM.
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

    grid_object = read_tif(full_path)

    return grid_object


def get_save_location() -> str:
    """Generates filepath to file saved in cache.

    Returns
    -------
    str
        Filepath to file saved in cache.
    """
    system = sys.platform

    if system == "win32":
        path = os.getenv('LOCALAPPDATA')
        if path is None:
            raise EnvironmentError(
                "LOCALAPPDATA environment variable is not set." +
                " Unable to generate path to cache.") from None
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


def clear_cache(filename: str | None = None) -> None:
    """Deletes the cache directory and its contents. Can also delete a single
    file when using the argument filename. To get the contents of your cache,
    use 'get_cache_contents()'.

    Parameters
    ----------
    filename : str, optional
        Add a filename if only one specific file is to be deleted.
        Defaults to None.
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

    Returns
    -------
    list[str]
        List of all files in the TopoToolbox cache. If cache does
        not exist, None is returned.
    """
    path = get_save_location()

    if os.path.exists(path):
        return os.listdir(path)

    print("Cache directory does not exist.")
    return None
