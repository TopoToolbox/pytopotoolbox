import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LightSource
from matplotlib import colormaps


class ImageschsMixin:
    def imageschs(
            self, 
            colorbar=True,
            colorbarlabel="",
            colormap="gist_earth",
            azimuth=310,
            altitude=60,
            exaggerate=1,
            shade_alpha=0.25,
            ):
        
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        hillshade = ls.hillshade(self.z, vert_exag=exaggerate/self.cellsize)

        img = plt.imshow(self.z, cmap=colormap, interpolation='nearest')
        img_overlay = plt.imshow(hillshade,  cmap="gray", alpha=shade_alpha)

        if colorbar:
            cbar = plt.colorbar(img)
            cbar.set_label(colorbarlabel)

        plt.show()
