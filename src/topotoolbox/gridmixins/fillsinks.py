import numpy as np
import ctypes
import os
import copy

class FillsinksMixin():
    def fillsinks(self, **kwargs): # -> GridObject: # TODO: fillsinks needs (circular) GridObject import

        basedir = os.path.dirname(__file__)
        parentdir = os.path.dirname(basedir)
        libpath = os.path.join(parentdir, 'libtopotoolbox.so')

        dll = ctypes.CDLL(libpath)

        dll.fillsinks.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # output
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # dem
            ctypes.c_long,   #nrows
            ctypes.c_long    #ncols
        ]
        dll.fillsinks.restype = None

        dem = self.z.astype(np.float32)
        output = np.zeros_like(dem)
        nrows, ncols = dem.shape

        dll.fillsinks(output, dem, nrows, ncols)

        copy_self = copy.copy(self)
        copy_self.z = output.copy()
        
        return copy_self
    