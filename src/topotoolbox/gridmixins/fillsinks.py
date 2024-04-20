from .._grid import grid_fillsinks

import numpy as np
import copy

class FillsinksMixin():
    def fillsinks(self):

        dem = self.z.astype(np.float32)

        output = np.zeros_like(dem)

        grid_fillsinks(output, dem, self.rows, self.columns)

        result = copy.copy(self)
        result.z = output.copy()
        
        return result
