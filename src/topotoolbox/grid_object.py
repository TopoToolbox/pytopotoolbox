import numpy as np
import rasterio

from .gridmixins.info import InfoMixin

class GridObject(
        InfoMixin
        ):
    
    def __init__(self, path=None) -> None:

        # Try to open file with rasterio.
        # Since rasterio can handle relaive and absolute paths,
        # there is no need to generate the absolute path first.
        # Exceptions can be reused from rasterio.
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


    # future implementation of init alternatives to generate
    # random or empty Gridobject 
    @classmethod
    def random(cls, size=30):
        pass

    @classmethod
    def empty(cls):
        pass

