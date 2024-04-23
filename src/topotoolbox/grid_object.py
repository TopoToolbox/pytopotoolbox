import numpy as np
import rasterio

from .gridmixins.info import InfoMixin
from .gridmixins.fillsinks import FillsinksMixin

class GridObject(
        InfoMixin,
        FillsinksMixin
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
            raise ImportError("For gen_random to work, opensimplex needs to be installed. (pip install opensimplex)") from None
        
        noise_array = np.empty((rows, columns), dtype=np.float32)
        for y in range(0, rows):
            for x in range(0, columns):
                value = simplex.noise4(x / hillsize, y / hillsize, 0.0, 0.0)
                color = int((value + 1) * 128)
                noise_array[y, x] = color

        instance = cls(None)
        instance.path=None    
        instance.z = noise_array
        instance.rows=rows
        instance.columns=columns
        instance.cellsize=cellsize

        return instance


    # TODO: implement gen_empty 
    @classmethod
    def gen_empty(cls):
        pass

