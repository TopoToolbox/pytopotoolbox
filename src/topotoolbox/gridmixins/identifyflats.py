import copy

import numpy as np

from .._grid import grid_identifyflats  # pylint: disable=import-error


class IdentifyflatsMixin():

    def identifyflats(self, raw=False, output=['sills', 'flats']) -> tuple:
        """
        Identifies flats and sills in a digital elevation model (DEM).

        Args:
            raw (bool): If True, returns the raw output grid as np.ndarray. 
                Defaults to False.
            output (list): List of strings indicating desired output types.
                Possible values are 'sills', 'flats'. Defaults to ['sills', 'flats'].

        Returns:
            tuple: A tuple containing copies of the DEM with identified flats and/or sills.

        Note:
            Flats are identified as 1s, sills as 2s and presills as 5s (since they are also flats)
              in the output grid. Only relevant when using raw=True.
        """

        dem = self.z.astype(np.float32)
        output_grid = np.zeros_like(dem).astype(np.int32)

        grid_identifyflats(output_grid, dem, self.rows, self.columns)

        if raw:
            return output_grid

        result = []
        if 'flats' in output:
            flats = copy.copy(self)
            flats.z = np.zeros_like(flats.z)
            flats.z = np.where((output_grid & 1) == 1, 1, flats.z)
            result.append(flats)

        if 'sills' in output:
            sills = copy.copy(self)
            sills.z = np.zeros_like(sills.z)
            sills.z = np.where((output_grid & 2) == 2, 1, sills.z)
            result.append(sills)

        return tuple(result)
