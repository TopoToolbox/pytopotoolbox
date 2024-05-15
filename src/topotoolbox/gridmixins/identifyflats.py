from .._grid import grid_identifyflats

import numpy as np
import copy


class IdentifyflatsMixin():
    def identifyflats(self):

        dem = self.z.astype(np.float32)

        output = np.zeros_like(dem).astype(np.int32)

        grid_identifyflats(output, dem, self.rows, self.columns)

        result = copy.copy(self)
        result.z = output

        return result
