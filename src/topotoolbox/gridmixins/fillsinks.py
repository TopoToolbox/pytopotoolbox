import numpy as np
import ctypes
import os
import copy

class FillsinksMixin():
    def fillsinks(self, maxdepth=None, sinks=None):

        #TODO: arguments need to be evaluated.

        basedir = os.path.dirname(__file__)
        parentdir = os.path.dirname(basedir)
        libpath = os.path.join(parentdir, 'libtopotoolbox.so')

        dll = ctypes.CDLL(libpath)

        dll.fillsinks.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # output
            np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'), # dem
            ctypes.c_ssize_t,   #nrows
            ctypes.c_ssize_t    #ncols
        ]
        dll.fillsinks.restype = None

        dem = self.z.astype(np.float32)
        output = np.zeros_like(dem)

        dll.fillsinks(output, dem, self.rows, self.columns)

        result = copy.copy(self)
        result.z = output.copy()
        
        return result
    