"""This module contains the Mixin class FillsinksMixin for the GridObject. 
"""
import copy

import numpy as np

# Importing the wrapper of the fillsinks function provided by pybind11.
from .._grid import grid_fillsinks  # pylint: disable=import-error


class FillsinksMixin():
    """This class is a Mixin for the GridObject class.
    It contains the fillsinks() function.
    """

    def fillsinks(self):
        """Fill sinks in the digital elevation model (DEM).

        Returns:
            GridObject: The filled DEM.
        """

        dem = self.z.astype(np.float32)

        output = np.zeros_like(dem)

        grid_fillsinks(output, dem, self.rows, self.columns)

        result = copy.copy(self)
        result.z = output

        return result
