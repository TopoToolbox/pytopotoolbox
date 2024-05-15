"""This module contains the GridObject class.
"""
import random
from typing import Union

import numpy as np
import rasterio

from .gridmixins.fillsinks import FillsinksMixin
from .gridmixins.info import InfoMixin
from .gridmixins.magic import MagicMixin


class GridObject(
        InfoMixin,
        FillsinksMixin,
        MagicMixin
):
    """A class containing all information of a Digital Elevation Model (DEM).
    This class combines mixins to provide various functionalities for working with DEMs.

    Args:
        InfoMixin: A mixin class providing methods to retrieve information about the DEM.
        FillsinksMixin: A mixin class providing a method to fill sinks in the DEM.
        MagicMixin: A mixin class providing magical methods for the DEM.
    """

    def __init__(self, path: Union[str, None] = None) -> None:
        """Initialize a GridObject instance.

        Args:
            path (str, optional): The path to the raster file. Defaults to None.

        Raises:
            TypeError: If an invalid type is passed as the `path`.
            ValueError: If an error occurs while processing the `path` argument.
        """

        if path is not None:
            try:
                dataset = rasterio.open(path)

            except TypeError as err:
                raise TypeError(err) from None
            except Exception as err:
                raise ValueError(err) from None

            self.path = path
            self.z = dataset.read(1).astype(np.float32)
            self.rows = dataset.height
            self.columns = dataset.width
            self.shape = self.z.shape
            self.cellsize = dataset.res[0]

        else:
            self.path = ''
            self.z = np.empty(())
            self.rows = 0
            self.columns = 0
            self.shape = self.z.shape
            self.cellsize = 0

    @classmethod
    def gen_random(
            cls, hillsize: int = 24, rows: int = 128, columns: int = 128,
            cellsize: float = 10.0) -> 'GridObject':
        """Generate a GridObject instance that is generated with OpenSimplex noise.

        Args:
            hillsize (int, optional): Controls the "smoothness" of the 
                                      generated terrain. Defaults to 24.
            rows (int, optional): Number of rows. Defaults to 128.
            columns (int, optional): Number of columns. Defaults to 128.
            cellsize (float, optional): Size of each cell in the grid. 
                                        Defaults to 10.0.

        Raises:
            ImportError: If OpenSimplex has not been installed.

        Returns:
            GridObject: An instance of GridObject with randomly generated values.
        """

        try:
            import opensimplex as simplex

        except ImportError:
            raise ImportError(
                """For gen_random to work, use \"pip install topotoolbox[opensimplex]\"
                  or \"pip install .[opensimplex]\"""") from None

        noise_array = np.empty((rows, columns), dtype=np.float32)
        for y in range(0, rows):
            for x in range(0, columns):
                value = simplex.noise4(x / hillsize, y / hillsize, 0.0, 0.0)
                color = int((value + 1) * 128)
                noise_array[y, x] = color

        instance = cls(None)
        instance.path = ''
        instance.z = noise_array
        instance.rows = rows
        instance.columns = columns
        instance.shape = instance.z.shape
        instance.cellsize = cellsize

        return instance

    # TODO: implement gen_empty

    @classmethod
    def gen_empty(cls) -> None:
        pass

    @classmethod
    def gen_random_bool(
            cls, rows: int = 32, columns: int = 32, cellsize: float = 10.0) -> 'GridObject':
        """Generate a GridObject instance that caontains only randomly
        generated Boolean values. 

        Args:
            rows (int, optional): Number of rows. Defaults to 32.
            columns (int, optional): Number of columns. Defaults to 32.
            cellsize (float, optional): size of each cell in the grid. Defaults to 10.

        Returns:
            GridObject: _description_
        """
        bool_array = np.empty((rows, columns), dtype=np.float32)

        for y in range(0, rows):
            for x in range(0, columns):
                bool_array[x][y] = random.choice([0, 1])

        instance = cls(None)
        instance.path = ''
        instance.z = bool_array
        instance.rows = rows
        instance.columns = columns
        instance.shape = instance.z.shape
        instance.cellsize = cellsize

        return instance
