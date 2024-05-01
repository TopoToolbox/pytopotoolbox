import numpy as np
import rasterio
import random

from .gridmixins.info import InfoMixin
from .gridmixins.fillsinks import FillsinksMixin
from .gridmixins.magic import MagicMixin


class GridObject(
        InfoMixin,
        FillsinksMixin,
        MagicMixin
):

    def __init__(self, path=None):
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
            self.cellsize = dataset.res[0]

    @classmethod
    def gen_random(cls, hillsize=24, rows=128, columns=128, cellsize=10):

        try:
            import opensimplex as simplex

        except ImportError as err:
            raise ImportError(
                "For gen_random to work, use \"pip install topotoolbox[opensimplex]\" or \"pip install .[opensimplex]\"") from None

        noise_array = np.empty((rows, columns), dtype=np.float32)
        for y in range(0, rows):
            for x in range(0, columns):
                value = simplex.noise4(x / hillsize, y / hillsize, 0.0, 0.0)
                color = int((value + 1) * 128)
                noise_array[y, x] = color

        instance = cls(None)
        instance.path = None
        instance.z = noise_array
        instance.rows = rows
        instance.columns = columns
        instance.cellsize = cellsize

        return instance

    # TODO: implement gen_empty

    @classmethod
    def gen_empty(cls):
        pass

    @classmethod
    def gen_random_bool(cls, rows=32, columns=32, cellsize=10):
        bool_array = np.empty((rows, columns), dtype=np.float32)

        for y in range(0, rows):
            for x in range(0, columns):
                bool_array[x][y] = random.choice([0, 1])

        instance = cls(None)
        instance.path = None
        instance.z = bool_array
        instance.rows = rows
        instance.columns = columns
        instance.cellsize = cellsize

        return instance
