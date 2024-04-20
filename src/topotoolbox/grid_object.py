import numpy as np
import rasterio

from .gridmixins.info import InfoMixin
from .gridmixins.fillsinks import FillsinksMixin

class GridObject(
        InfoMixin,
        FillsinksMixin
        ):
    
    def __init__(self, path=None):

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

    # TODO: classmethods random, empty, [..] 
    @classmethod
    def random(cls, size=30):
        pass

    @classmethod
    def empty(cls):
        pass

