"""This module contains the GridObject class.
"""

import copy

import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=import-error
from ._grid import (  # type: ignore
    grid_fillsinks,
    grid_identifyflats,
    grid_excesstopography_fsm2d,
    grid_excesstopography_fmm2d
)

__all__ = ['GridObject']


class GridObject():
    """A class containing all information of a Digital Elevation Model (DEM).
    """

    def __init__(self) -> None:
        """Initialize a GridObject instance.
        """
        # path to file
        self.path = ''
        # name of DEM
        self.name = ''

        # raster metadata
        self.z = np.empty((), order='F')
        self.rows = 0
        self.columns = 0
        self.shape = self.z.shape

        self.cellsize = 0

        # georeference
        self.bounds = None
        self.transform = None
        self.crs = None

    def fillsinks(self) -> 'GridObject':
        """Fill sinks in the digital elevation model (DEM).

        Returns
        -------
        GridObject
            The filled DEM.
        """

        dem = self.z.astype(np.float32, order='F')
        output = np.zeros_like(dem)

        grid_fillsinks(output, dem, self.rows, self.columns)

        result = copy.copy(self)
        result.z = output

        return result

    def identifyflats(
            self, raw: bool = False, output: list[str] = None) -> tuple:
        """Identifies flats and sills in a digital elevation model (DEM).

        Parameters
        ----------
        raw : bool, optional
            If True, returns the raw output grid as np.ndarray. 
            Defaults to False.
        output : list of str, optional
            List of strings indicating desired output types. Possible values 
            are 'sills', 'flats'. Order of inputs in list are irrelevant,
            first entry in output will always be sills. 
            Defaults to ['sills', 'flats'].

        Returns
        -------
        tuple
            A tuple containing copies of the DEM with identified 
            flats and/or sills.

        Notes
        -----
        Flats are identified as 1s, sills as 2s, and presills as 5s 
        (since they are also flats) in the output grid. 
        Only relevant when using raw=True.
        """

        if output is None:
            output = ['sills', 'flats']

        dem = self.z.astype(np.float32, order='F')
        output_grid = np.zeros_like(dem, dtype=np.int32)

        grid_identifyflats(output_grid, dem, self.rows, self.columns)

        if raw:
            return output_grid

        result = []
        if 'flats' in output:
            flats = copy.copy(self)
            flats.z = np.zeros_like(flats.z, order='F')
            flats.z = np.where((output_grid & 1) == 1, 1, flats.z)
            result.append(flats)

        if 'sills' in output:
            sills = copy.copy(self)
            sills.z = np.zeros_like(sills.z, order='F')
            sills.z = np.where((output_grid & 2) == 2, 1, sills.z)
            result.append(sills)

        return tuple(result)

    def excesstopography(
            self, threshold: "float | int | np.ndarray | GridObject" = 0.2,
            method: str = 'fsm2d',) -> 'GridObject':
        """
    Compute the two-dimensional excess topography using the specified method.

    Parameters
    ----------
    threshold : float, int, GridObject, or np.ndarray, optional
        Threshold value or array to determine slope limits, by default 0.2.
        If a float or int, the same threshold is applied to the entire DEM.
        If a GridObject or np.ndarray, it must match the shape of the DEM.
    method : str, optional
        Method to compute the excess topography, by default 'fsm2d'.
        Options are:
        - 'fsm2d': Uses the fast sweeping method.
        - 'fmm2d': Uses the fast marching method.

    Returns
    -------
    GridObject
        A new GridObject with the computed excess topography.

    Raises
    ------
    ValueError
        If `method` is not one of ['fsm2d', 'fmm2d'].
        If `threshold` is an np.ndarray and doesn't match the shape of the DEM.
    TypeError
        If `threshold` is not a float, int, GridObject, or np.ndarray.
        """

        if method not in ['fsm2d', 'fmm2d']:
            err = (f"Invalid method '{method}'. Supported methods are" +
                   " 'fsm2d' and 'fmm2d'.")
            raise ValueError(err) from None

        dem = self.z

        if isinstance(threshold, (float, int)):
            threshold_slopes = np.full(
                dem.shape, threshold, order='F', dtype=np.float32)
        elif isinstance(threshold, GridObject):
            threshold_slopes = threshold.z
        elif isinstance(threshold, np.ndarray):
            threshold_slopes = threshold
        else:
            err = "Threshold must be a float, int, GridObject, or np.ndarray."
            raise TypeError(err) from None

        if not dem.shape == threshold_slopes.shape:
            err = "Threshold array must have the same shape as the DEM."
            raise ValueError(err) from None
        if not threshold_slopes.flags['F_CONTIGUOUS']:
            threshold_slopes = np.asfortranarray(threshold)
        if not np.issubdtype(threshold_slopes.dtype, np.float32):
            threshold_slopes = threshold_slopes.astype(np.float32)

        excess = np.zeros_like(dem)
        cellsize = self.cellsize
        nrows, ncols = self.shape

        if method == 'fsm2d':
            grid_excesstopography_fsm2d(
                excess, dem, threshold_slopes, cellsize, nrows, ncols)

        elif method == 'fmm2d':
            heap = np.zeros_like(dem, dtype=np.int64)
            back = np.zeros_like(dem, dtype=np.int64)

            grid_excesstopography_fmm2d(
                excess, heap, back, dem, threshold_slopes, cellsize, nrows, ncols)

        result = copy.copy(self)
        result.z = excess

        return result

    def info(self) -> None:
        """Prints all variables of a GridObject.
        """
        print(f"name: {self.name}")
        print(f"path: {self.path}")
        print(f"rows: {self.rows}")
        print(f"cols: {self.columns}")
        print(f"cellsize: {self.cellsize}")
        print(f"bounds: {self.bounds}")
        print(f"transform: {self.transform}")
        print(f"crs: {self.crs}")

    def show(self, cmap='terrain') -> None:
        """
        Display the GridObject instance as an image using Matplotlib.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap that will be used in the plot. 
        """
        plt.imshow(self, cmap=cmap)
        plt.title(self.name)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # 'Magic' functions:
    # ------------------------------------------------------------------------

    def __eq__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] == other.z[x][y]

        return dem

    def __ne__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] != other.z[x][y]

        return dem

    def __gt__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] > other.z[x][y]

        return dem

    def __lt__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] < other.z[x][y]

        return dem

    def __ge__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] >= other.z[x][y]

        return dem

    def __le__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] <= other.z[x][y]

        return dem

    def __add__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z + other.z
            return dem

        dem.z = self.z + other
        return dem

    def __sub__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z - other.z
            return dem

        dem.z = self.z - other
        return dem

    def __mul__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z * other.z
            return dem

        dem.z = self.z * other
        return dem

    def __div__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z / other.z
            return dem

        dem.z = self.z / other
        return dem

    def __and__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'and' can only compare " +
                        "True (1) and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) & int(other.z[x][y]))

        return dem

    def __or__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'or' can only compare True (1)" +
                        " and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) | int(other.z[x][y]))

        return dem

    def __xor__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'xor' can only compare True (1)" +
                        " and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) ^ int(other.z[x][y]))

        return dem

    def __len__(self):
        return len(self.z)

    def __iter__(self):
        return iter(self.z)

    def __getitem__(self, index):
        return self.z[index]

    def __setitem__(self, index, value):
        try:
            value = np.float32(value)
        except:
            raise TypeError(
                f"{value} can't be converted to float32.") from None

        self.z[index] = value

    def __array__(self):
        return self.z

    def __str__(self):
        return str(self.z)
