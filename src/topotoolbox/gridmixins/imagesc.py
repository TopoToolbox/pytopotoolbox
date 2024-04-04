import matplotlib.pyplot as plt
from matplotlib import colormaps

class ImagescMixin:
    def imagesc(self, colormap="terrain"):    
            
        if not isinstance(colormap, str):
            raise TypeError(str(colormap) + " is not a valid value for colormap; choose from the matplotlib colormaps.") from None
        if colormap not in colormap:
            raise ValueError(str(colormap) + " is not a vaild value for colormap; choose from the matplotlib colormaps.") from None
        
        plt.imshow(self.z, cmap=colormap, interpolation='nearest')
        plt.colorbar()
        plt.show()
        