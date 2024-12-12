"""This module contains the GridObject class.
"""
import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    convolve,
    median_filter,
    generic_filter,
    grey_erosion,
    grey_dilation)
from scipy.signal import wiener
from rasterio import CRS
from rasterio.warp import reproject
from rasterio.enums import Resampling

# pylint: disable=no-name-in-module
from . import _grid  # type: ignore

__all__ = ['GridObject']


class GridObject():
    """A class containing all information of a Digital Elevation Model (DEM).
    """

    def __init__(self) -> None:
        """Initialize a GridObject instance.
        """
        # path to file
        self.path = ''
        # name of DEM
        self.name = ''

        # raster metadata
        self.z = np.empty((), order='F', dtype=np.float32)

        self.cellsize = 0.0  # in meters if crs.is_projected == True

        # georeference
        self.bounds = None
        self.transform = None
        self.crs = None

    @property
    def shape(self):
        """Tuple of grid dimensions
        """
        return self.z.shape

    @property
    def rows(self):
        """The size of the first dimension of the grid
        """
        return self.z.shape[0]

    @property
    def columns(self):
        """The size of the second dimension of the grid
        """
        return self.z.shape[1]

    def reproject(self,
                  crs: 'CRS',
                  resolution: 'float | None' = None,
                  resampling: 'Resampling' = Resampling.bilinear):
        """Reproject GridObject to a new coordinate system.

        Parameters
        ----------
        crs : rasterio.CRS
            Target coordinate system
        resolution : float, optional
            Target resolution.
            If one is not provided, a resolution that approximately
            matches that of the source coordinate system will be used.
        resampling : rasterio.enums.Resampling, optional
            Resampling method.
            The default is bilinear resampling.

        Returns
        -------
        GridObject
            The reprojected data.

        """
        dst = GridObject()

        dst.z, dst.transform = reproject(
            self.z,
            src_transform=self.transform,
            src_crs=self.crs,
            dst_transform=None,  # Let rasterio derive the transform for us
            dst_crs=crs,
            dst_nodata=np.nan,
            dst_resolution=resolution,
            resampling=resampling,
        )
        # reproject gives us a 3D array, we want the first band
        # We also want it in column-major order
        dst.z = np.asfortranarray(dst.z[0, :, :])

        dst.crs = crs

        # Get cellsize from transform in case we did not specify one
        if dst.transform is not None:
            dst.cellsize = abs(dst.transform[0])

        return dst

    def fillsinks(self,
                  bc: 'np.ndarray | GridObject | None' = None,
                  hybrid: bool = True) -> 'GridObject':
        """Fill sinks in the digital elevation model (DEM).

        Parameters
        ----------
        bc : ndarray or GridObject, optional
            Boundary conditions for sink filling. `bc` should be an array
            of np.uint8 that matches the shape of the DEM. Values of 1
            indicate pixels that should be fixed to their values in the
            original DEM and values of 0 indicate pixels that should be
            filled.
        hybrid: bool, optional
            Should hybrid reconstruction algorithm be used? Defaults to True. Hybrid
            reconstruction is faster but requires additional memory be allocated
            for a queue.

        Returns
        -------
        GridObject
            The filled DEM.

        """

        dem = self.z.astype(np.float32, order='F')
        output = np.zeros_like(dem)

        restore_nans = False

        if bc is None:
            bc = np.ones_like(dem, dtype=np.uint8)
            bc[1:-1, 1:-1] = 0  # Set interior pixels to 0

            nans = np.isnan(dem)
            dem[nans] = -np.inf
            bc[nans] = 1  # Set NaNs to 1
            restore_nans = True

        if bc.shape != self.shape:
            err = ("The shape of the provided boundary conditions does not "
                   f"match the shape of the DEM. {self.shape}")
            raise ValueError(err)from None

        if isinstance(bc, GridObject):
            bc = bc.z

        if hybrid:
            queue = np.zeros_like(dem, dtype=np.int64)
            _grid.fillsinks_hybrid(output, queue, dem, bc, self.shape)
        else:
            _grid.fillsinks(output, dem, bc, self.shape)

        if restore_nans:
            dem[nans] = np.nan
            output[nans] = np.nan

        result = copy.copy(self)
        result.z = output

        return result

    def identifyflats(
            self, raw: bool = False, output: list[str] | None = None) -> tuple:
        """Identifies flats and sills in a digital elevation model (DEM).

        Parameters
        ----------
        raw : bool, optional
            If True, returns the raw output grid as np.ndarray.
            Defaults to False.
        output : list of str, optional
            List of strings indicating desired output types. Possible values
            are 'sills', 'flats'. Order of inputs in list are irrelevant,
            first entry in output will always be sills.
            Defaults to ['sills', 'flats'].

        Returns
        -------
        tuple
            A tuple containing copies of the DEM with identified
            flats and/or sills.

        Notes
        -----
        Flats are identified as 1s, sills as 2s, and presills as 5s
        (since they are also flats) in the output grid.
        Only relevant when using raw=True.
        """

        # Since having lists as default arguments can lead to problems, output
        # is initialized with None by default and only converted to a list
        # containing default output here:
        if output is None:
            output = ['sills', 'flats']

        dem = self.z.astype(np.float32, order='F')
        output_grid = np.zeros_like(dem, dtype=np.int32)

        _grid.identifyflats(output_grid, dem, self.shape)

        if raw:
            return tuple(output_grid)

        result = []
        if 'flats' in output:
            flats = copy.copy(self)
            flats.z = np.zeros_like(flats.z, order='F')
            flats.z = np.where((output_grid & 1) == 1, 1, flats.z)
            result.append(flats)

        if 'sills' in output:
            sills = copy.copy(self)
            sills.z = np.zeros_like(sills.z, order='F')
            sills.z = np.where((output_grid & 2) == 2, 1, sills.z)
            result.append(sills)

        return tuple(result)

    def excesstopography(
            self, threshold: "float | int | np.ndarray | GridObject" = 0.2,
            method: str = 'fsm2d',) -> 'GridObject':
        """
    Compute the two-dimensional excess topography using the specified method.

    Parameters
    ----------
    threshold : float, int, GridObject, or np.ndarray, optional
        Threshold value or array to determine slope limits, by default 0.2.
        If a float or int, the same threshold is applied to the entire DEM.
        If a GridObject or np.ndarray, it must match the shape of the DEM.
    method : str, optional
        Method to compute the excess topography, by default 'fsm2d'.
        Options are:

        - 'fsm2d': Uses the fast sweeping method.
        - 'fmm2d': Uses the fast marching method.

    Returns
    -------
    GridObject
        A new GridObject with the computed excess topography.

    Raises
    ------
    ValueError
        If `method` is not one of ['fsm2d', 'fmm2d'].
        If `threshold` is an np.ndarray and doesn't match the shape of the DEM.
    TypeError
        If `threshold` is not a float, int, GridObject, or np.ndarray.
        """

        if method not in ['fsm2d', 'fmm2d']:
            err = (f"Invalid method '{method}'. Supported methods are" +
                   " 'fsm2d' and 'fmm2d'.")
            raise ValueError(err) from None

        dem = self.z

        if isinstance(threshold, (float, int)):
            threshold_slopes = np.full(
                dem.shape, threshold, order='F', dtype=np.float32)
        elif isinstance(threshold, GridObject):
            threshold_slopes = threshold.z
        elif isinstance(threshold, np.ndarray):
            threshold_slopes = threshold
        else:
            err = "Threshold must be a float, int, GridObject, or np.ndarray."
            raise TypeError(err) from None

        if not dem.shape == threshold_slopes.shape:
            err = "Threshold array must have the same shape as the DEM."
            raise ValueError(err) from None
        if not threshold_slopes.flags['F_CONTIGUOUS']:
            threshold_slopes = np.asfortranarray(threshold)
        if not np.issubdtype(threshold_slopes.dtype, np.float32):
            threshold_slopes = threshold_slopes.astype(np.float32)

        excess = np.zeros_like(dem)
        cellsize = self.cellsize

        if method == 'fsm2d':
            _grid.excesstopography_fsm2d(
                excess, dem, threshold_slopes, cellsize, self.shape)

        elif method == 'fmm2d':
            heap = np.zeros_like(dem, dtype=np.int64)
            back = np.zeros_like(dem, dtype=np.int64)

            _grid.excesstopography_fmm2d(excess, heap, back, dem,
                                         threshold_slopes, cellsize,
                                         self.shape)

        result = copy.copy(self)
        result.z = excess

        return result

    def filter(self, method: str = 'mean', kernelsize: int = 3) -> 'GridObject':
        """The function filter is a wrapper around various image filtering
        algorithms. Only filters with rectangular kernels of uneven
        dimensions are supported.

        Parameters
        ----------
        method : str, optional
            Which method will be used to filter the DEM: ['mean', 'average',
            'median', 'sobel', 'scharr', 'wiener', 'std'], by default 'mean'
        kernelsize : int, optional
            Size of the kernel that will be applied. Note that ['sobel',
            'scharr'] require that the kernelsize is 3, by default 3

        Returns
        -------
        GridObject
            The filtered DEM as a GridObject.

        Raises
        ------
        ValueError
            If the kernelsize does not match the requirements of this function
            or the selected method is not implemented in the function.
        """

        valid_methods = ['mean', 'average', 'median',
                         'sobel', 'scharr', 'wiener', 'std']

        if method in ['mean', 'average']:
            kernel = np.ones((kernelsize, kernelsize)) / kernelsize**2
            filtered = convolve(self.z, kernel, mode='nearest')

        elif method in ['median']:
            filtered = median_filter(self.z, size=kernelsize, mode='reflect')

        elif method in ['sobel', 'scharr']:
            if kernelsize != 3:
                arr = f"The method '{method}' only works with a 3x3 kernel'."
                raise ValueError(arr) from None

            if method == 'sobel':
                ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            else:  # 'scharr'
                ky = np.array([[3, 10, 3], [0, 0, 0], [-3, - 10, - 3]])

            kx = ky.T
            filtered = np.hypot(
                convolve(self.z, ky, mode='nearest'),
                convolve(self.z, kx, mode='nearest'))

        elif method in ['wiener']:
            filtered = wiener(self.z, mysize=kernelsize)

        elif method in ['std']:
            # This solution is based on this thread:
            # https://stackoverflow.com/questions/19518827/what-is-the-python
            # -equivalent-for-matlabs-stdfilt-function
            filtered = generic_filter(self.z, np.std, size=kernelsize)
            factor = np.sqrt(kernelsize**2 / (kernelsize**2 - 1))
            np.multiply(filtered, factor, out=filtered)

        else:
            err = (f"Argument 'method={method}' has to be"
                   f"one of {valid_methods}.")
            raise ValueError(err) from None

        # Keep NaNs like they are in self.z
        filtered[np.isnan(self.z)] = np.nan

        result = copy.copy(self)
        result.z = filtered
        return result

    def gradient8(self, unit: str = 'tangent', multiprocessing: bool = True):
        """
    Compute the gradient of a digital elevation model (DEM) using an
    8-direction algorithm.

    Parameters
    ----------
    unit : str, optional
        The unit of the gradient to be calculated. Options are:
        - 'tangent' : Calculate the gradient as a tangent (default).
        - 'radian'  : Calculate the gradient in radians.
        - 'degree'  : Calculate the gradient in degrees.
        - 'sine'    : Calculate the gradient as the sine of the angle.
        - 'percent' : Calculate the gradient as a percentage.
    multiprocessing : bool, optional
        If True, use multiprocessing for computation. Default is True.

    Returns
    -------
    GridObject
        A new GridObject with the calculated gradient.
        """

        if multiprocessing:
            use_mp = 1
        else:
            use_mp = 0

        dem = self.z.astype(np.float32, order='F')
        output = np.zeros_like(dem)

        _grid.gradient8(output, dem, self.cellsize, use_mp, self.shape)
        result = copy.copy(self)

        if unit == 'radian':
            output = np.arctan(output)
        elif unit == 'degree':
            output = np.arctan(output) * (180.0 / np.pi)
        elif unit == 'sine':
            output = np.sin(np.arctan(output))
        elif unit == 'percent':
            output = output * 100.0

        result.z = output

        return result

    def curvature(self, ctype='profc', meanfilt=False) -> 'GridObject':
        """curvature returns the second numerical derivative (curvature) of a
        digital elevation model. By default, curvature returns the profile
        curvature (profc).

        Parameters
        ----------
        ctype : str, optional
            What type of curvature will be computed, by default 'profc'
            - 'profc' : profile curvature [m^(-1)],
            - 'planc' : planform curvature [m^(-1))],
            - 'tangc' : tangential curvature [m^(-1)],
            - 'meanc' : mean curvature [m^(-1)],
            - 'total' : total curvature [m^(-2)]
        meanfilt : bool, optional
            True if a mean filter will be applied before comuting the
            curvature, by default False

        Returns
        -------
        GridObject
            A GridObject storing the computed values.

        Raises
        ------
        ValueError
            If wrong ctype has been used.

        Examples
        --------
        >>> dem = topotoolbox.load_dem('tibet')
        >>> curv = dem.curvature()
        >>> curv.show()
        """

        if meanfilt:
            kernel = np.ones((3, 3)) / 9
            dem = convolve(self.z, kernel, mode='nearest')
        else:
            dem = self.z

        kernel_fx = np.array(
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / (6 * self.cellsize)
        kernel_fy = np.array(
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / (6 * self.cellsize)

        fx = convolve(dem, kernel_fx, mode='nearest')
        fy = convolve(dem, kernel_fy, mode='nearest')

        kernel_fxx = np.array(
            [[1, -2, 1], [1, -2, 1], [1, -2, 1]]) / (3 * self.cellsize**2)
        kernel_fyy = kernel_fxx.T
        kernel_fxy = np.array(
            [[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * self.cellsize**2)

        fxx = convolve(dem, kernel_fxx, mode='nearest')
        fyy = convolve(dem, kernel_fyy, mode='nearest')
        fxy = convolve(dem, kernel_fxy, mode='nearest')

        epsilon = 1e-10
        if ctype == 'profc':
            denominator = (fx**2 + fy**2) * (1 + fx **
                                             2 + fy**2)**(3/2) + epsilon
            curvature = - (fx**2 * fxx + 2 * fx * fy * fxy + fy **
                           2 * fyy) / denominator
        elif ctype == 'tangc':
            denominator = (fx**2 + fy**2) * (1 + fx **
                                             2 + fy**2)**(1/2) + epsilon
            curvature = - (fy**2 * fxx - 2 * fx * fy * fxy + fx **
                           2 * fyy) / denominator
        elif ctype == 'planc':
            denominator = (fx**2 + fy**2)**(3/2) + epsilon
            curvature = - (fy**2 * fxx - 2 * fx * fy * fxy + fx **
                           2 * fyy) / denominator
        elif ctype == 'meanc':
            denominator = 2 * (fx**2 + fy**2 + 1)**(3/2) + epsilon
            curvature = -((1 + fy ** 2) * fxx - 2 * fxy * fx * fy +
                          (1 + fx ** 2) * fyy) / denominator
        elif ctype == 'total':
            curvature = fxx**2 + 2 * fxy**2 + fyy**2
        else:
            raise ValueError(
                "Invalid curvature type. Must be one of: 'profc', 'planc',"
                "'tangc', 'meanc', 'total'.")

        # Keep NaNs like they are in dem
        curvature[np.isnan(dem)] = np.nan

        result = copy.copy(self)
        result.z = curvature
        return result

    def dilate(
            self, size: tuple | None = None, footprint: np.ndarray | None = None,
            structure: np.ndarray | None = None) -> 'GridObject':
        """A simple wrapper around th scipy.ndimage.grey_dilation function,
        that also handles NaNs in the input GridObject. Either size, footprint
        or structure has to be passed to this function. If nothing is provided,
        the function will raise an error.

        size : tuple of ints, optional
            A tuple of ints containing the shape of the structuring element.
            Only needed if neither footprint nor structure is provided. Will
            result in a full and flat structuring element.
            Defaults to None
        footprint : np.ndarray of ints, optional
            A array defining the footprint of the erosion operation.
            Non-zero elements define the neighborhood over which the erosion
            is applied. Defaults to None
        structure : np.ndarray of ints, optional
            A array defining the structuring element used for the erosion. 
            This defines the connectivity of the elements. Defaults to None

        Returns
        -------
        GridObject
            A GridObject storing the computed values.
        Raises
        ------
        ValueError
            If size, structure and footprint are all None.
        """

        if size is None and structure is None and footprint is None:
            err = ("Dilate requires a structuring element to be specified."
                   " Use the size argument for a full and flat structuring"
                   " element (equivalent to a a minimum filter) or the"
                   " structure and footprint arguments to specify"
                   " a non-flat structuring element.")
            raise ValueError(err) from None

        # Replace NaN values with inf
        dem = self.z.copy()
        dem[np.isnan(dem)] = -np.inf

        dilated = grey_dilation(
            input=dem, size=size, structure=structure, footprint=footprint)

        # Keep NaNs like they are in dem
        dilated[np.isnan(self.z)] = np.nan

        result = copy.copy(self)
        result.z = dilated
        return result

    def erode(
            self, size: tuple | None = None, footprint: np.ndarray | None = None,
            structure: np.ndarray | None = None) -> 'GridObject':
        """Apply a morphological erosion operation to the GridObject. Either
        size, footprint or structure has to be passed to this function. If
        nothing is provided, the function will raise an error.

        Parameters
        ----------
        size : tuple of ints
            A tuple of ints containing the shape of the structuring element.
            Only needed if neither footprint nor structure is provided. Will
            result in a full and flat structuring element.
            Defaults to None
        footprint : np.ndarray of ints, optional
            A array defining the footprint of the erosion operation.
            Non-zero elements define the neighborhood over which the erosion
            is applied. Defaults to None
        structure : np.ndarray of ints, optional
            A array defining the structuring element used for the erosion. 
            This defines the connectivity of the elements. Defaults to None

        Returns
        -------
        GridObject
            A GridObject storing the computed values.

        Raises
        ------
        ValueError
            If size, structure and footprint are all None."""

        if size is None and structure is None and footprint is None:
            err = ("Erode requires a structuring element to be specified."
                   " Use the size argument for a full and flat structuring"
                   " element (equivalent to a a minimum filter) or the"
                   " structure and footprint arguments to specify"
                   " a non-flat structuring element.")
            raise ValueError(err) from None

        # Replace NaN values with inf
        dem = self.z.copy()
        dem[np.isnan(dem)] = np.inf

        eroded = grey_erosion(
            dem, size=size, structure=structure, footprint=footprint)

        # Keep NaNs like they are in dem
        eroded[np.isnan(self.z)] = np.nan

        result = copy.copy(self)
        result.z = eroded
        return result

    def _gwdt_computecosts(self) -> np.ndarray:
        """
        Compute the cost array used in the gradient-weighted distance
        transform (GWDT) algorithm.


        Returns
        -------
        np.ndarray
            A 2D array of costs corresponding to each grid cell in the DEM.
        """
        dem = self.z
        flats = self.identifyflats(raw=True)
        filled_dem = self.fillsinks().z
        dims = self.shape
        costs = np.zeros_like(dem, dtype=np.float32, order='F')
        conncomps = np.zeros_like(dem, dtype=np.int64, order='F')

        _grid.gwdt_computecosts(costs, conncomps, flats, dem, filled_dem, dims)
        del conncomps, flats, filled_dem
        return costs

    def _gwdt(self) -> np.ndarray:
        """
        Perform the grey-weighted distance transform (GWDT) on the DEM.

        Returns
        -------
        np.ndarray
            A 2D array representing the GWDT distances for each grid cell.
        """
        costs = self._gwdt_computecosts()
        flats = self.identifyflats(raw=True)
        dims = self.shape
        dist = np.zeros_like(flats, dtype=np.float32, order='F')
        prev = np.zeros_like(flats, dtype=np.int64, order='F')
        heap = np.zeros_like(flats, dtype=np.int64, order='F')
        back = np.zeros_like(flats, dtype=np.int64, order='F')

        _grid.gwdt(dist, prev, costs, flats, heap, back, dims)
        del costs, prev, heap, back
        return dist

    def _flow_routing_d8_carve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the flow routing using the D8 algorithm with carving
        for flat areas.

        Returns
        -------
        np.ndarray
            array indicating the source cells for flow routing. (source)
        np.ndarray
            array indicating the flow direction for each grid cell. (direction)
        """
        filled_dem = self.fillsinks().z
        dist = self._gwdt()
        flats = self.identifyflats(raw=True)
        dims = self.shape
        source = np.zeros_like(flats, dtype=np.int64, order='F')
        direction = np.zeros_like(flats, dtype=np.uint8, order='F')

        _grid.flow_routing_d8_carve(
            source, direction, filled_dem, dist, flats, dims)
        del filled_dem, dist, flats
        return source, direction

    def _flow_routing_targets(self) -> np.ndarray:
        """
        Identify the target cells for flow routing.

        Returns
        -------
        np.ndarray
            A 2D array where each cell points to its downstream target cell.
        """

        source, direction = self._flow_routing_d8_carve()
        dims = self.shape
        target = np.zeros_like(source, dtype=np.int64, order='F')

        _grid.flow_routing_targets(target, source, direction, dims)
        del source, direction
        return target

    def info(self) -> None:
        """Prints all variables of a GridObject.
        """
        print(f"name: {self.name}")
        print(f"path: {self.path}")
        print(f"rows: {self.rows}")
        print(f"cols: {self.columns}")
        print(f"cellsize: {self.cellsize}")
        print(f"bounds: {self.bounds}")
        print(f"transform: {self.transform}")
        print(f"crs: {self.crs}")

    def show(self, cmap='terrain') -> None:
        """
        Display the GridObject instance as an image using Matplotlib.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap that will be used in the plot.
        """
        plt.imshow(self, cmap=cmap)
        plt.title(self.name)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # 'Magic' functions:
    # ------------------------------------------------------------------------

    def __eq__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] == other.z[x][y]

        return dem

    def __ne__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] != other.z[x][y]

        return dem

    def __gt__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] > other.z[x][y]

        return dem

    def __lt__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] < other.z[x][y]

        return dem

    def __ge__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] >= other.z[x][y]

        return dem

    def __le__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                dem.z[x][y] = self.z[x][y] <= other.z[x][y]

        return dem

    def __add__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z + other.z
            return dem

        dem.z = self.z + other
        return dem

    def __sub__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z - other.z
            return dem

        dem.z = self.z - other
        return dem

    def __mul__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z * other.z
            return dem

        dem.z = self.z * other
        return dem

    def __div__(self, other):
        dem = copy.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z / other.z
            return dem

        dem.z = self.z / other
        return dem

    def __and__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'and' can only compare " +
                        "True (1) and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) & int(other.z[x][y]))

        return dem

    def __or__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'or' can only compare True (1)" +
                        " and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) | int(other.z[x][y]))

        return dem

    def __xor__(self, other):
        dem = copy.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        for x in range(0, self.columns):
            for y in range(0, self.rows):

                if (self.z[x][y] not in [0, 1]
                        or other.z[x][y] not in [0, 1]):

                    raise ValueError(
                        "Invalid cell value. 'xor' can only compare True (1)" +
                        " and False (0) values.")

                dem.z[x][y] = (int(self.z[x][y]) ^ int(other.z[x][y]))

        return dem

    def __len__(self):
        return len(self.z)

    def __iter__(self):
        return iter(self.z)

    def __getitem__(self, index):
        return self.z[index]

    def __setitem__(self, index, value):
        try:
            value = np.float32(value)
        except (ValueError, TypeError):
            raise TypeError(
                f"{value} can't be converted to float32.") from None

        self.z[index] = value

    def __array__(self):
        return self.z

    def __str__(self):
        return str(self.z)
