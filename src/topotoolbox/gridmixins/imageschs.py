import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LightSource
from matplotlib import colormaps


class ImageschsMixin:
    def imageschs(self, **kwargs):

        # caxis = kwargs.get()
        colorbar = kwargs.get("colorbar",True)
        colorbarlabel = kwargs.get("colorbarlabel", "")
        colorbarylabel = kwargs.get("colorbarylabel", "")
        colormap = kwargs.get('colormap', 'gist_earth')
        percentclip = kwargs.get("percentclip", 0)
        truecolor = kwargs.get("truecolor", "g")
        falsecolor = kwargs.get("falsecolor", "w")
        nancolor = kwargs.get("nancolor", "w")
        brighten = kwargs.get("brighten", 0)
        usepermanent = kwargs.get("usepermanent", False)
        medfilt = kwargs.get("medfilt", False)
        # method = kwargs.get()
        azimuth = kwargs.get("azimuth", 315)
        altitude = kwargs.get("altitude", 60)
        exaggerate = kwargs.get("exaggerate", 2)
        # ticklabels = kwargs.get()
        # tickstokm = kwargs.get()
        # gridmarkers = kwargs.get()
        # gridmarkercolor = kwargs.get()
        

        #if not isinstance(caxis, ):
        #    raise ValueError("")
        if not isinstance(colorbar, bool):
            raise ValueError("Colorbar must be a boolean value.")
        if not isinstance(colorbarlabel, str):
            raise ValueError("Colorbarlabel must be a string.")
        if not isinstance(colorbarylabel, str):
            raise ValueError("Colorbarylabel must be a string.")
        if not isinstance(colormap, str) or colormap not in colormaps:
            raise ValueError("Colormap must be a string chosen from the matplotlib colormap variants.")
        if not isinstance(percentclip, (int,float)):
            raise ValueError("Percentclip must be a numeric value.")
        if not isinstance(truecolor, str):
            raise ValueError("Truecolor must be a string.")
        if not isinstance(falsecolor, str):
            raise ValueError("Falsecolor must be a string.")
        if not isinstance(nancolor, str):
            raise ValueError("Nancolor must be a string.")
        if not isinstance(brighten, (int, float)):
            raise ValueError("Brighten must be numeric value.")
        if not isinstance(usepermanent, bool):
            raise ValueError("Usepermanent must be a boolean value.")
        if not isinstance(medfilt, bool):
            raise ValueError("Medfilt must be a boolean value.")
        # if not isinstance(method, str):
        #     raise ValueError("")
        if not isinstance(azimuth, (int, float)):
            raise ValueError("Azimuth must be a numeric value.")
        if not isinstance(altitude, (int, float)):
            raise ValueError("Altitude must be a numeric value.")
        if not isinstance(exaggerate, (int, float)):
            raise ValueError("Exaggerate must be a numric value.")
        # if not isinstance(ticklabels, ):
        #     raise ValueError("")
        # if not isinstance(tickstokm, ):
        #     raise ValueError("")
        # if not isinstance(gridmarkers, ):
        #     raise ValueError("")
        # if not isinstance(gridmarkercolor, ):
        #     raise ValueError("")
        
        # Plotting the DEM:
        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        hillshade = ls.hillshade(self.z, vert_exag=exaggerate/self.cellsize)

        img = plt.imshow(self.z, cmap=colormap, interpolation='nearest')
        img_overlay = plt.imshow(hillshade,  cmap="gray", alpha=0.25)
        if colorbar:
            cbar = plt.colorbar(img)
            if not colorbarlabel == "":
                cbar.set_label(colorbarlabel)
            if not colorbarylabel == "":
                cbar.set_label(colorbarylabel, rotation=270, labelpad=15)

        # Show plot
        plt.show()