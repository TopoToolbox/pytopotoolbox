import tempfile
import os

import topotoolbox as tt3


def test_write_tif_georef():
    dem = tt3.load_dem('bigtujunga')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'output.tif')
        tt3.write_tif(dem, path)
        dem2 = tt3.read_tif(path)
        assert dem2.georef == dem.georef

        os.remove(path)
