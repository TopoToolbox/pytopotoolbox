import matplotlib.pyplot as plt
from matplotlib import colormaps

class ImagescMixin:
    def imagesc(self, **kwargs):

        colormap = kwargs.get('colormap', "terrain")
        if not isinstance(colormap, str) or colormap not in colormaps:
            raise ValueError("colormap must be a string chosen from the matplotlib colormap variants.")

        plt.imshow(self.z, cmap=colormap, interpolation='nearest')
        plt.colorbar()
        plt.show()