"""Provide general utility functions for topotoolbox.
"""
import sys
import os
import random
from shutil import rmtree
from urllib.request import urlopen, urlretrieve
import urllib.error

import rasterio
import numpy as np

from .grid_object import GridObject

__all__ = ["load_dem", "get_dem_names", "read_tif", "gen_random", "write_tif",
           "gen_random_bool", "get_cache_contents", "clear_cache",
           "read_from_cache", "load_opentopography"]


DEM_SOURCE = "https://raw.githubusercontent.com/TopoToolbox/DEMs/master"
DEM_NAMES = f"{DEM_SOURCE}/dem_names.txt"
OPEN_TOPO_SOURCE = "https://portal.opentopography.org/API/globaldem"
OPEN_TOPO_DATASETS = ('SRTMGL3', 'SRTMGL1', 'SRTMGL1_E', 'AW3D30', 'AW3D30_E',
                      'SRTM15Plus', 'NASADEM', 'COP30', 'COP90', 'EU_DTM',
                      'GEDI_L3', 'GEBCOIceTopo', 'GEBCOSubIceTopo',
                      'CA_MRDEM_DTM', 'CA_MRDEM_DSM')


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

    Example
    -------
    >>> dem = topotoolbox.load_dem('perfectworld')
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
            crs=dem.georef,
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

    Example
    -------
    >>> dem = topotoolbox.read_tif('dem.tif')
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
        grid.georef = dataset.crs

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

    Example
    -------
    >>> dem = topotoolbox.gen_random(seed=8192)
    >>> dem.plot(cmap='terrain')
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

    Example
    -------
    >>> dem = topotoolbox.gen_random_bool()
    >>> dem.plot(cmap='grey')
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

    Example
    -------
    >>> print(topotoolbox.get_dem_names())
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

    Example
    -------
    >>> dem = topotoolbox.load_dem('taiwan')
    >>> dem.plot(cmap='terrain')
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

    Example
    -------
    >>> print(topotoolbox.get_save_location())
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

    Example
    -------
    >>> topotoolbox.clear_cache()
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

    Example
    -------
    >>> print(topotoolbox.get_cache_contents())
    """
    path = get_save_location()

    if os.path.exists(path):
        return os.listdir(path)

    print("Cache directory does not exist.")
    return None


def read_from_cache(filename: str) -> GridObject:
    """Read a GeoTIFF file from the cache directory and return it as a
    GridObject. The filename should be the name of the GeoTIFF file. Find
    available files by using 'get_cache_contents()'.

    Parameters
    ----------
    filename : str
        Name of the file to be read from the cache directory. Requires the
        whole filename including the extension, like "dem.tif".

    Returns
    -------
    GridObject
        The GridObject generated from the cached GeoTIFF file.

    Example
    -------
    >>> topotoolbox.load_dem('bigtujunga')
    >>> dem = topotoolbox.read_from_cache('bigtujunga.tif')
    >>> dem.plot()
    """
    cache_path = os.path.join(get_save_location(), f"{filename}")
    grid_object = read_tif(cache_path)
    return grid_object


def load_opentopography(south: float, north: float, west: float, east: float,
                        api_key: str | None = None, dem_type: str = "SRTMGL3",
                        api_path: str | None = None, overwrite: bool = False,
                        save_path: str | None = None
                        ) -> GridObject:
    """Download a DEM from Open Topography. The DEM is downloaded as a
    GeoTIFF file and saved in the cache directory. The DEM is then
    read into a GridObject. The DEMs come in geographic coordinates (WGS84).
    The API key is required for accessing the OpenTopography API
    (opentopography.org). To avoid having to pass the API key as an argument,
    save it in .opentopography.txt in the working or home directory. To
    overwrite an existing downloaded DEM, use the overwrite parameter. To save
    the DEM to a different location, use the save_path parameter.

    Parameters
    ----------
    south : float
        WGS 84 bounding box south coordinates
    north : float
        WGS 84 bounding box north coordinates
    west : float
        WGS 84 bounding box west coordinates
    east : float
        WGS 84 bounding box east coordinates
    api_key : str | None
        The API key is needed to access the Open Topography API. To get an API
        key, visit opentopography.org, log in, navigate to MyOpenTopoDashboard
        and request a new API key. This argument is used to pass the
        API key directly as a string. By default None.
    api_path : str | None
        The path to a file that contains the API key. If provided, the API key
        will be read from the file. The file should contain only the API key
        without any additional text or formatting. By default None.
    dem_type : str, optional
        Choose one of the available global raster types, by default "SRTMGL3"

        - SRTMGL3 (SRTM GL3 90m)
        - SRTMGL1 (SRTM GL1 30m)
        - SRTMGL1_E (SRTM GL1 Ellipsoidal 30m)
        - AW3D30 (ALOS World 3D 30m)
        - AW3D30_E (ALOS World 3D Ellipsoidal, 30m)
        - SRTM15Plus (Global Bathymetry SRTM15+ V2.1 500m)
        - NASADEM (NASADEM Global DEM)
        - COP30 (Copernicus Global DSM 30m)
        - COP90 (Copernicus Global DSM 90m)
        - EU_DTM (DTM 30m)
        - GEDI_L3 (DTM 1000m)
        - GEBCOIceTopo (Global Bathymetry 500m)
        - GEBCOSubIceTopo (Global Bathymetry 500m)
        - CA_MRDEM_DSM (DSM 30m)
        - CA_MRDEM_DTM (DTM 30m)

    overwrite : bool, optional
        If True cached DEM will be overwritten if it has the same bounds
        and dem_type, by default False
    save_path : str | None, optional
        If provided, the downloaded GeoTIFF will be saved to this path. Like
        this `"path/to/file.tif"` for example. If it is None, files will
        be saved in cache. By default None

    Returns
    -------
    GridObject
        A GridObject generated from the downloaded DEM.

    Raises
    ------
    ValueError
        If the provided DEM type is not valid.
    ConnectionError
        If the API request fails or returns an error.

    Example
    -------
    >>> dem = topotoolbox.load_open_topography(south=50, north=50.1, west=14.35,
                    east=14.6, dem_type="SRTMGL3", api_key="demoapikeyot2022")
    >>> dem = dem.reproject(rasterio.CRS.from_epsg(32633), resolution=90)
    >>> im = dem.plot(cmap="terrain")
    >>> plt.show()
    """

    # Check if an API key is provided
    if isinstance(api_key, str):
        api = api_key
    elif isinstance(api_path, str):
        try:
            # since encoding is not predefined, use unspecified encoding
            # pylint: disable=unspecified-encoding
            with open(str(api_path), 'r') as file:
                api = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"API key file not found: {api_path}") from None
        except PermissionError:
            raise PermissionError(
                f"Cannot read API key file: {api_path}") from None
    else:
        # Since no API key was provided, try defaults:
        if os.path.exists(os.path.abspath('.opentopography.txt')):
            try:
                # pylint: disable=unspecified-encoding
                with open(os.path.abspath(".opentopography.txt"), 'r') as file:
                    api = file.read().strip()
            except (FileNotFoundError, PermissionError):
                pass
        elif os.path.exists(os.path.expanduser('~/.opentopography.txt')):
            try:
                # pylint: disable=unspecified-encoding
                with open(os.path.expanduser("~/.opentopography.txt"), 'r') as file:
                    api = file.read().strip()
            except (FileNotFoundError, PermissionError):
                pass
        else:
            # No API key has been passed or found
            err = ("Neither api_key, api_env or api_path have been provided. "
                   "Default environment variable 'OPENTOPOGRAPHY_API_KEY' or "
                   "file '~/.opentopography.txt' are not set. Use"
                   " api_key='demoapikeyot2022' as the demo key.")
            raise ValueError(err) from None

    if dem_type not in OPEN_TOPO_DATASETS:
        raise ValueError(
            f"Invalid DEM type. Available types are: {OPEN_TOPO_DATASETS}")

    # Assemble the cache path. Create unique/identifiable name for the GeoTIFF
    dem = f"OpenTopo_{south}_{north}_{west}_{east}_{dem_type}"

    # if no save_path is provided, use the cache path
    if save_path is None:
        save_path = os.path.join(get_save_location(), f"{dem}.tif")
    elif os.path.exists(save_path) and not overwrite:
        err = (f"File {save_path} already exists. Use overwrite=True to "
               "overwrite and the file.")
        raise FileExistsError(err)

    if not os.path.exists(save_path) or overwrite:
        url = (f"{OPEN_TOPO_SOURCE}"
               f"?demtype={dem_type}"
               f"&south={south}"
               f"&north={north}"
               f"&west={west}"
               f"&east={east}"
               f"&outputFormat=GTiff"
               f"&API_Key={api}")

        try:
            urlretrieve(url, save_path)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                err = ("Error: 401 - Unauthorized. Check your API key. "
                       "If you don't have one, use this guide to generate one:"
                       " https://opentopography.org/blog/introducing-api-keys"
                       "-access-opentopography-global-datasets")
                raise ConnectionError(err) from None

            error_dict = {
                204: "Bad Data",
                400: "Bad Request",
                500: "Internal Server Error"
            }
            code = e.code
            err = f"Error: {code} - {error_dict.get(code, 'Unknown Error')}"
            raise ConnectionError(err) from None

    grid_obj = read_tif(save_path)
    return grid_obj
