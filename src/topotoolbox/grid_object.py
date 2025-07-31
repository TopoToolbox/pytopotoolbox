"""This module contains the GridObject class.
"""
import copy as cp
from typing import Tuple, List

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.ndimage import (
    convolve,
    median_filter,
    generic_filter,
    grey_erosion,
    grey_dilation,
    distance_transform_edt
)
from scipy.signal import wiener

from rasterio import CRS, Affine
from rasterio.coords import BoundingBox
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling

# pylint: disable=no-name-in-module
from . import _grid, _morphology  # type: ignore
from .interface import validate_alignment

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
        self.z: npt.NDArray = np.empty((), order='F', dtype=np.float32)

        self.cellsize = 0.0  # in meters if crs.is_projected == True

        # georeference
        self.bounds = None
        self.transform = Affine.identity()
        self.georef = None

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

    @property
    def dims(self):
        """The dimensions of the grid in the correct order for libtopotoolbox
        """
        if self.z.flags.c_contiguous:
            return (self.columns, self.rows)

        if self.z.flags.f_contiguous:
            return (self.rows, self.columns)

        raise TypeError(
            "Grid is not stored as a contiguous row- or column-major array")

    @property
    def extent(self):
        """The bounding box of the grid in the order needed for plotting

        Returns
        -------
        tuple
            The bounding box in the order (left, right, bottom, top)
        """
        if self.bounds:
            return (self.bounds.left, self.bounds.right, self.bounds.bottom, self.bounds.top)

        return (-0.5, self.columns-0.5, self.rows-0.5, -0.5)

    @property
    def coordinates(self):
        """Coordinate arrays for the DEM

        Returns
        -------
        X,Y : tuple of ndarrays
            The two returned arrays are of the same shape as the
        DEM. The first contains the coordinates of each pixel in the
        horizontal dimension, and the second contains the coordinates
        in the vertical dimension.
        """
        x, y = np.meshgrid(np.arange(self.columns), np.arange(self.rows))
        return self.transform * (x, y)

    def astype(self, dtype):
        """Copy of the GridObject, cast to specified type

        Parameters
        ----------
        dtype: str or np.dtype
            The numpy data type to which the GridObject is cast

        Returns
        -------
        GridObject
            A copy of the original GridObject with the given data type

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> dem = dem.astype(np.float32)
        """
        result = GridObject()
        result.path = self.path
        result.name = self.name
        result.z = self.z.astype(dtype, copy=True)

        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.georef = self.georef

        return result

    def reproject(self,
                  georef: 'CRS',
                  resolution: 'float | None' = None,
                  resampling: 'Resampling' = Resampling.bilinear):
        """Reproject GridObject to a new coordinate system.

        Parameters
        ----------
        georef : rasterio.CRS
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

        Example
        -------
        >>> dem = topotoolbox.load_open_topography(south=50, north=50.1, west=14.35,
                    east=14.6, dem_type="SRTMGL3", api_key="demoapikeyot2022")
        >>> dem = dem.reproject(rasterio.CRS.from_epsg(32633), resolution=90)
        >>> im = dem.plot(cmap="terrain")
        >>> plt.show()
        """
        dst = GridObject()

        z, dst.transform = reproject(
            self.z,
            src_transform=self.transform,
            src_crs=self.georef,
            dst_transform=None,  # Let rasterio derive the transform for us
            dst_crs=georef,
            dst_nodata=np.nan,
            dst_resolution=resolution,
            resampling=resampling,
        )
        # reproject gives us a 3D array, we want the first band.
        dst.z = np.zeros_like(self.z, shape=z.shape[1:3])
        dst.z[:, :] = z[0, :, :]

        dst.georef = georef

        # Get cellsize from transform in case we did not specify one
        if dst.transform is not None:
            dst.cellsize = abs(dst.transform[0])

        if self.bounds:
            dst.bounds = BoundingBox(
                *transform_bounds(self.georef, dst.georef, *self.bounds))

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

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> filled_dem = dem.fillsinks()
        >>> filled_dem.plot(cmap='terrain')
        """
        dem = self.z.astype(np.float32)
        output = np.zeros_like(dem)

        restore_nans = False

        if bc is None:
            bc = np.ones_like(dem, dtype=np.uint8)
            bc[1:-1, 1:-1] = 0  # Set interior pixels to 0

            nans = np.isnan(dem)
            dem[nans] = -np.inf
            bc[nans] = 1  # Set NaNs to 1
            restore_nans = True

        if not validate_alignment(self, bc):
            err = ("The shape of the provided boundary conditions does not "
                   f"match the shape of the DEM. {self.shape}")
            raise ValueError(err)from None

        if isinstance(bc, GridObject):
            bc = bc.z

        if hybrid:
            queue = np.zeros_like(dem, dtype=np.int64)
            _grid.fillsinks_hybrid(output, queue, dem, bc, self.dims)
        else:
            _grid.fillsinks(output, dem, bc, self.dims)

        if restore_nans:
            dem[nans] = np.nan
            output[nans] = np.nan

        result = cp.copy(self)
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

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> flats, sills = dem.identifyflats()
        >>> flats.plot(cmap='terrain')
        """

        # Since having lists as default arguments can lead to problems, output
        # is initialized with None by default and only converted to a list
        # containing default output here:
        if output is None:
            output = ['sills', 'flats']

        dem = np.asarray(self, dtype=np.float32)
        output_grid = np.zeros_like(dem, dtype=np.int32)

        _grid.identifyflats(output_grid, dem, self.dims)

        if raw:
            return (output_grid,)

        result = []
        if 'flats' in output:
            flats = cp.copy(self)
            flats.z = np.zeros_like(flats.z)
            flats.z = np.where((output_grid & 1) == 1, 1, flats.z)
            result.append(flats)

        if 'sills' in output:
            sills = cp.copy(self)
            sills.z = np.zeros_like(sills.z)
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

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> excess = dem.excesstopography(threshold=0.3, method='fsm2d')
        >>> excess.plot(cmap='terrain')
        """

        if method not in ['fsm2d', 'fmm2d']:
            err = (f"Invalid method '{method}'. Supported methods are" +
                   " 'fsm2d' and 'fmm2d'.")
            raise ValueError(err) from None

        dem = np.asarray(self, dtype=np.float32)

        if isinstance(threshold, (float, int)):
            threshold_slopes = np.full_like(dem, threshold)
        elif isinstance(threshold, GridObject):
            threshold_slopes = np.asarray(threshold, dtype=np.float32)
        elif isinstance(threshold, np.ndarray):
            threshold_slopes = np.asarray(threshold, dtype=np.float32)
        else:
            err = "Threshold must be a float, int, GridObject, or np.ndarray."
            raise TypeError(err) from None

        if not validate_alignment(dem, threshold_slopes):
            err = "Threshold array must have the same shape as the DEM."
            raise ValueError(err) from None

        excess = np.zeros_like(dem)
        cellsize = self.cellsize

        if method == 'fsm2d':
            _grid.excesstopography_fsm2d(
                excess, dem, threshold_slopes, cellsize, self.dims)

        elif method == 'fmm2d':
            heap = np.zeros_like(dem, dtype=np.int64)
            back = np.zeros_like(dem, dtype=np.int64)

            _grid.excesstopography_fmm2d(excess, heap, back, dem,
                                         threshold_slopes, cellsize,
                                         self.dims)

        result = cp.copy(self)
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

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> sharr = dem.filter(method='scharr', kernelsize=3)
        >>> sharr.plot(cmap='terrain')
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

        result = cp.copy(self)
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

    Example
    -------
    >>> dem = topotoolbox.load_dem('perfectworld')
    >>> grad = = dem.gradient8()
    >>> grad.plot(cmap='terrain')
        """

        if multiprocessing:
            use_mp = 1
        else:
            use_mp = 0

        dem = np.asarray(self.z, dtype=np.float32)
        output = np.zeros_like(dem)

        _grid.gradient8(output, dem, self.cellsize, use_mp, self.dims)
        result = cp.copy(self)

        if unit == 'radian':
            output = np.arctan(output)
        elif unit == 'degree':
            output = np.rad2deg(np.arctan(output))
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
        >>> curv.plot(cmap='terrain')
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

        result = cp.copy(self)
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

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> dilate = dem.dilate(size=10)
        >>> dilate.plot(cmap='terrain')
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

        result = cp.copy(self)
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
            If size, structure and footprint are all None.

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> eroded = dem.erode()
        >>> eroded.plot(cmap='terrain')
        """

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

        result = cp.copy(self)
        result.z = eroded
        return result

    def evansslope(
            self, partial_derivatives: bool = False, mode: str = 'nearest',
            modified: bool = False) -> 'GridObject' | Tuple['GridObject', 'GridObject']:
        """Evans method fits a second-order polynomial to 3x3 subgrids. The
        parameters of the polynomial are the partial derivatives which are
        used to calculate the surface slope = sqrt(Gx**2 + Gy**2).

        Evans method approximates the surface by regression surfaces.
        Gradients are thus less susceptible to noise in the DEM.

        Parameters
        ----------
        mode : str, optional
            The mode parameter determines how the input DEM is extended
            beyond its boundaries: ['reflect', 'constant', 'nearest', 'mirror',
            'wrap', 'grid-mirror', 'grid-constant', 'grid-wrap']. See
            scipy.ndimage.convolve for more information, by default 'nearest'
        modified : bool, optional
            If True, the surface is weakly smoothed before gradients are
            calculated (see Shary et al., 2002), by default False
        partial_derivatives : bool, optional
            If True, both partial derivatives [fx, fy] will be returned as
            GridObjects instead of just the evansslope, by default False

        Returns
        -------
        GridObject
            A GridObject containing the computed evansslope data.

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> slope = dem.evansslope()
        >>> slope.plot(cmap='terrain')
        """
        dem = self.z.copy()
        # NaN replacement not optional since convolve can't handle NaNs
        indices = distance_transform_edt(
            np.isnan(dem), return_distances=False, return_indices=True)
        dem = dem[tuple(indices)]

        if modified:
            kernel = np.array([[0, 1, 0], [1, 41, 1], [0, 1, 0]])/45
            dem = convolve(dem, kernel, mode=mode)

        kx = np.array(
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])/(6*self.cellsize)
        fx = convolve(dem, kx, mode=mode)
        # kernel for dz/dy
        ky = np.array(
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/(6*self.cellsize)
        fy = convolve(dem, ky, mode=mode)

        if partial_derivatives:
            result_kx = cp.copy(self)
            result_ky = cp.copy(self)
            result_kx.z = kx
            result_ky.z = ky
            return result_kx, result_ky

        slope = np.sqrt(fx**2 + fy**2)
        slope[np.isnan(self.z)] = np.nan

        result = cp.copy(self)
        result.z = slope
        return result

    def aspect(self, classify: bool = False) -> 'GridObject':
        """Aspect returns the slope exposition of each cell in a digital
        elevation model in degrees. In contrast to the second output of
        gradient8 which returns the steepest slope direction, aspect
        returns the angle of the slope.

        Parameters
        ----------
        classify : bool, optional
            directions are classified according to the scheme proposed by
            Gomez-Plaza et al. (2001), by default False

        Returns
        -------
        GridObject
            A GridObject containing the computed aspect data.

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> aspect = dem.aspect()
        >>> aspect.plot(cmap='terrain')
        """

        grad_y, grad_x = np.gradient(self.z, edge_order=2)
        aspect: np.ndarray = np.arctan2(-grad_x, grad_y)
        aspect = np.degrees(aspect)
        aspect = np.mod(aspect, 360)

        if classify:
            aspclass = np.array([1, 3, 5, 7, 8, 6, 4, 2])
            aspedges = aspect // 45
            aspedges = aspedges.astype(np.int8)

            aspect = aspclass[aspedges]
            aspect = aspect.astype(np.int8)

        result = cp.copy(self)
        result.z = aspect
        return result

    def prominence(self, tolerance: float, use_hybrid=True) -> Tuple:
        """This function calculates the prominence of peaks in a DEM. The
        prominence is the minimal amount one would need to descend from a peak
        before being able to ascend to a higher peak. The function uses image
        reconstruct (see function imreconstruct) to calculate the prominence.
        It may take a while to run for large DEMs. The algorithm iteratively
        finds the next higher prominence and stops if the prominence is less
        than the tolerance, the second input parameter to the function.

        Parameters
        ----------
        tolerance : float
            The minimum tolerance for the second to last found peak. (meters)
            Will always find one peak.
        use_hybrid : bool, optional
            If True, use the hybrid reconstruction algorithm. Defaults to True.

        Returns
        -------
        Tuple[np.ndarray, Tuple]
            A Tuple containing a ndarray storing the computed prominence and
            a tuple of ndarray. Each array in the inner tuple has the same
            shape as the indices array (as returned by np.unravel_index).

        Examples
        --------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> prom, idx = dem.prominence(tolerance=90)
        >>> plt.subplot()
        >>> dem.plot(cmap='terrain')
        >>> plt.plot(idx[0], idx[1], 'ro')
        """
        dem = np.nan_to_num(self.z)
        p = np.full_like(dem, np.min(dem))

        prominence: List[float] = []
        indices = []

        queue = np.zeros_like(dem, dtype=np.int64)

        while not prominence or prominence[-1] > tolerance:
            diff = dem - p
            prominence.append(np.max(diff))

            # By default argmax returns indices into the row-major
            # flattened array even if the array is not
            # row-major. However, unravel_index unravels by default in
            # row-major order, so the resulting pair of indices are
            # valid regardless of the memory order.
            indices.append(np.unravel_index(np.argmax(diff), self.shape))

            p[indices[-1]] = dem[indices[-1]]
            if use_hybrid:
                _morphology.reconstruct_hybrid(p, queue, dem, self.dims)
            else:
                _morphology.reconstruct(p, dem, self.dims)

        prominence_array = np.array(prominence)
        indices_array = np.array(indices)
        indices_array = indices_array[:, [1, 0]]  # swap columns 0 and 1
        # transpose to get (x, y) instead of (y, x)
        indices_array = indices_array.T
        return prominence_array, indices_array

    def hillshade(self,
                  azimuth: float = 315.0,
                  altitude: float = 60.0,
                  exaggerate: float = 1.0,
                  fused=True):
        """Compute a hillshade of a digital elevation model

        Parameters
        ----------
        azimuth : float
            The azimuth angle of the light source measured in degrees
            clockwise from north. Defaults to 315 degrees.
        altitude : float
            The altitude angle of the light source measured in degrees
            above the horizon. Defaults to 60 degrees.
        exaggerate : float
            The amount of vertical exaggeration. Increase to emphasize
            elevation differences in flat terrain. Defaults to 1.0
        fused : bool, optional
            If true, use the fused hillshade computation in
            libtopotoolbox, which requires less memory but can be
            slightly slower. If you have a small DEM, and are
            repeatedly creating hillshades consider
            setting to False for increased performance. Defaults to True.

        Returns
        -------
        GridObject
            A GridObject containing the resulting hillshade data

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> hillshade = dem.hillshade()
        >>> hillshade.plot(cmap='gray')
        >>> dem.plot(cmap='terrain', alpha=0.2)

        """

        z = np.asarray(self, dtype=np.float32)
        h = np.zeros_like(z)

        # Computing the azimuth angle is a bit tricky
        gt = self.transform

        # Remove the translation from the geotransform. It is not
        # needed to rotate the coordinate system, but it makes things
        # harder to work with.
        gt = gt.translation(-gt.xoff, -gt.yoff)*gt

        if self.z.flags.f_contiguous:
            # If the array is column-major, we need to swap the x and
            # y coordinates, which is achieved with a matrix like
            #
            # [[0  -1],
            #  [-1  0]]
            #
            # which we can construct as a permutation followed by a
            # 180 degree rotation.
            gt = gt.rotation(180) * gt.permutation() * gt

        # This is the /east/ component of the azimuth vector
        sx = np.sin(np.deg2rad(azimuth))
        # This is the /north/ component of the azimuth vector
        sy = np.cos(np.deg2rad(azimuth))

        # Apply the inverse of the geotransform to convert from
        # geographic coordinates to image coordinates.
        dx, dy = ~gt * (sx, sy)

        # And retrieve the azimuth angle.
        azimuth_radians = np.arctan2(dy, dx)

        # NOTE(wkearn): This angle is then immediately used within
        # hillshade to compute vector components again. It would be
        # somewhat more efficient and numerically stable to work
        # directly with the vectors. libtopotoolbox's hillshade should
        # probably take a vector rather than an angle.
        #
        # See Inigo Quilez (2013). Avoiding trigonometry
        # (https://iquilezles.org/articles/noacos/) for an argument
        # against using angles in graphics APIs.

        altitude_radians = np.deg2rad(altitude)

        if fused:
            _grid.hillshade_fused(h, exaggerate * z,
                                  azimuth_radians, altitude_radians,
                                  self.cellsize, self.dims)
        else:
            nx = np.zeros_like(z)
            ny = np.zeros_like(z)
            _grid.hillshade(h, nx, ny, exaggerate * z,
                            azimuth_radians, altitude_radians,
                            self.cellsize, self.dims)

        result = cp.copy(self)
        result.z = h
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
        flats = self.identifyflats(raw=True)[0]
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
        flats = self.identifyflats(raw=True)[0]
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
        flats = self.identifyflats(raw=True)[0]
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
        if self.georef is not None and self.georef.is_projected:
            print(f"coordinate system (Projected): {self.georef}")
        elif self.georef is not None and self.georef.is_geographic:
            print(f"coordinate system (Geographic): {self.georef}")
        else:
            print(f"coordinate system: {self.georef}")
        print(f"maximum z-value: {np.nanmax(self.z)}")
        print(f"minimum z-value: {np.nanmin(self.z)}")

    def plot(self, ax=None, extent=None, **kwargs):
        """Plot the GridObject

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes in which to plot the GridObject. If no axes
            are given, the current axes are used.

        extent: floats (left, right, bottom, top), optional
            The bounding box used to set the axis limits. If no extent
            is supplied, defaults to self.extent, which plots the
            GridObject in geographic coordinates.

        **kwargs
            Additional keyword arguments are forwarded to
            matplotlib.axes.Axes.imshow

        Returns
        -------
        matplotlib.image.AxesImage
            The image constructed by imshow

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> dem.plot(cmap='terrain')
        """
        if ax is None:
            ax = plt.gca()

        if extent is None:
            extent = self.extent

        return ax.imshow(self.z, extent=extent, **kwargs)

    def plot_hs(self, ax=None,
                elev=None,
                azimuth=315, altitude=60, exaggerate=1,
                filter_method=None, filter_size=3,
                cmap='terrain', norm=None,
                blend_mode='soft',
                extent=None,
                colorbar = False,
                **kwargs):
        """Plot a shaded relief map of the GridObject

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes in which to plot the GridObject. If no axes
            are given, the current axes are used.
        elev: GridObject, optional
            The digital elevation model used for shading. If no DEM is
            provided, the data GridObject is also used for shading.
        azimuth: float, optional
            The azimuth angle of the light source in degrees from
            North. Defaults to 315 degrees.
        altitude: float, optional
            The altitude angle of the light source in degrees above
            the horizon. Defaults to 60 degrees.
        exaggerate: float, optional
            The amount of vertical exaggeration to apply to the
            elevation. Defaults to 1.
        filter_method: 'str', optional
            The method used to filter the DEM before computing the
            hillshade. The data GridObject is not filtered. This
            should be one of the methods provided by
            `GridObject.filter`. Defaults to None, which does not
            apply a filter.
        filter_size: int, optional
            The size of the filter kernel in pixels. Defaults to 3.
        cmap: colors.Colormap or str or None
            The colormap to use in coloring the data. Defaults to
            'terrain'.
        norm: colors.Normalize, optional
            The normalization method that scales the color data to the
            [0,1] interval. Defaults to a linear scaling from the
            minimum of the data to the maximum.
        blend_mode: {'multiply', 'overlay', 'soft'}, optional
            The algorithm used to combine the shaded elevation with
            the data. Defaults to 'soft'.
        extent: floats (left, right, bottom, top), optional
            The bounding box used to set the axis limits. If no extent
            is supplied, defaults to self.extent
        **kwargs
            Additional keyword arguments are forwarded to
            matplotlib.axes.Axes.imshow

        Returns
        -------
        matplotlib.image.AxesImage
            The image constructed by imshow

        Raises
        ------
        TypeError
            The `elev` argument is not a GridObject

        ValueError
            The `elev` argument is not the same shape as the data

        ValueError
            A `blend_mode` other than 'multiply', 'overlay' or 'soft' is provided.

        ValueError
            The `filter_method` or `filter_size` arguments are not
            accepted by `GridObject.filter`.

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> dem.plot_hs(exaggerate=dem.cellsize)
        """
        if ax is None:
            ax = plt.gca()

        if elev is None:
            shade = self
        elif isinstance(elev, GridObject):
            if not validate_alignment(self, elev):
                err = "elev GridObject must have the same shape as the GridObject."
                raise ValueError(err) from None
            shade = elev
        else:
            err = "elev must be a GridObject"
            raise TypeError(err) from None

        if filter_method is not None:
            shade = shade.filter(method=filter_method, kernelsize=filter_size)

        h = shade.hillshade(azimuth, altitude, exaggerate)
        cmap = plt.get_cmap(cmap)

        if norm is None:
            norm = colors.Normalize(vmin=np.nanmin(
                self.z), vmax=np.nanmax(self.z))

        base = cmap(norm(self.z))
        top = np.expand_dims(np.clip(h, 0, 1), 2)
        if blend_mode == "multiply":
            rgb = base * top
        elif blend_mode == "overlay":
            rgb = np.where(base < 0.5, 2*base*top, 1 - 2*(1-base)*(1-top))
        elif blend_mode == "soft":
            rgb = (1 - 2*top)*base**2 + 2 * top * base
        else:
            raise ValueError("blend_mode not supported") from None

        if extent is None:
            extent = self.extent

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        return ax.imshow(np.clip(rgb, 0, 1), extent=extent, **kwargs)

    def plot_surface(self, ax=None, **kwargs):
        """Plot DEM as a 3D surface

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes in which to plot the GridObject. If no axes
            are given, the current axes are used.

        **kwargs
            Additional keyword arguments are forwarded to
            matplotlib.axes.Axes3D.plot_surface.

        Example
        -------
        >>> dem = topotoolbox.load_dem('bigtujunga')
        >>> fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        >>> dem.plot_surface(ax=ax)
        >>> ax.set_aspect('equal')
        >>> ax.set_zticks([0,np.nanmax(dem)])
        >>> plt.show()
        """
        if ax is None:
            ax = plt.gca()

        x, y = self.coordinates

        return ax.plot_surface(x, y, self.z, **kwargs)

    def shufflelabel(self, seed=None):
        """Randomize the labels of a GridObject

        This function is helpful when plotting drainage basins. It will work with
        any kind of data, but is most useful when given ordinal data such as an
        integer-valued GridObject.

        Parameters
        ----------
        seed: optional

          The seed used to generate the random permutation of labels.

          The seed is passed directly to `numpy.random.default_rng`__.

          .. __:
             https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng

        Returns
        -------
        GridObject
          A grid identical to the input, but with randomly reassigned labels.

        Example
        -------
        >>> dem = topotoolbox.load_dem('bigtujunga')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> D = fd.drainagebasins()
        >>> D.shufflelabel().plot(cmap="Pastel1",interpolation="nearest")
        """
        result = cp.copy(self)

        u, indices = np.unique(self, return_inverse=True)
        rng = np.random.default_rng(seed)
        result.z = np.reshape(rng.permutation(u)[indices], self.shape)

        return result

    def duplicate_with_new_data(self, data: np.ndarray) -> 'GridObject':
        """Duplicate a GridObject with different data

        This function is helpful when one wants to create a GridObject from
        a numpy array with the exact same properties (e.g. georef, ...) but
        different data

        Parameters
        ----------
        data: np.ndarray

          The new data (needs to be in the same shape than the current GridObject)

        Returns
        -------
        GridObject
          A grid identical to the input, but with new data.

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> new_dem = dem.duplicate_with_new_data(np.zeros(dem.shape))
        """
        rows, columns = data.shape

        if self.columns != columns or self.rows != rows:
            raise ValueError("Both GridObjects have to be the same size.")

        result = cp.deepcopy(self)

        result.z = np.array(data, copy=True)

        return result

    def zscore(self):
        """Returns the z-score for each element of GridObject such that
        all values are centered to have mean 0 and scaled to have
        standard deviation 1.

        Returns
        -------
        GridObject
            A GridObject containing the z-scores of the input GridObject.

        Example
        -------
        >>> dem = topotoolbox.load_dem('tibet')
        >>> dem_zscore = dem.zscore()
        >>> dem_zscore.plot()
        """
        result = cp.copy(self)
        result.z = (self.z - np.nanmean(self.z)) / np.nanstd(self.z)
        return result

    def crop(self, left: float | int, right: float | int,
               top: float | int, bottom: float | int,
               mode: str) -> 'GridObject':
        """Crop the Gridobject by specifying new boundaries.

        Supports three input modes (percent, coordinate, pixel) to define
        the crop region. Method of crop has to be chosen by using the mode
        argument. In case of reversed boundaries, automatically swaps them to
        ensure the crop region is valid. The resulting grid will have a new
        transform and bounds based on the specified boundaries.

        Parameters
        ----------
        left : float or int
            Left boundary in one of three modes:
            - Percent: 0.0 to 1.0 (relative to grid width)
            - Coordinate: Within grid's horizontal bounds
            - Pixel: Column index (0 to grid width-1)
        right : float or int
            Right boundary (same modes as `left`).
        top : float or int
            Top boundary in one of three modes:
            - Percent: 0.0-1.0 (relative to grid height)
            - Coordinate: Within grid's vertical bounds
            - Pixel: Row index (0 to grid height-1)
        bottom : float or int
            Bottom boundary (same modes as `top`).
        mode : str
            The mode of the crop operation. Can be 'percent', 'coordinate',
            or 'pixel'.

        Returns
        -------
        GridObject
            Cropped grid with updated transform, bounds, and data.

        Raises
        ------
        ValueError
            If boundaries are not in a consistent valid mode.

        Example
        -------
        >>> dem = topotoolbox.load_dem('tibet')
        >>> new_dem = dem.crop(0.6, 0.8, 0.3, 0.5, 'percent')
        >>> dem.plot()
        >>> b = new_dem.bounds
        >>> plt.plot([b.left, b.right, b.right, b.left, b.left],
                [b.top, b.top, b.bottom, b.bottom, b.top],
                'r-', lw=2)
        >>> plt.show()
        """
        height, width = self.shape[0], self.shape[1]
        left_bound, right_bound = self.extent[0], self.extent[1]
        top_bound, bottom_bound = self.extent[3], self.extent[2]

        if mode == 'percent':
            if not all(0.0 <= float(val) <= 1.0
                       for val in [top, bottom, left, right]):
                err = ("All values must be between 0.0 and 1.0 for mode "
                       "'percent'.")
                raise ValueError(err) from None

            y_start = int(top * height)
            y_end = int(bottom * height)
            x_start = int(left * width)
            x_end = int(right * width)

        elif mode == 'coordinate':
            if not (all(left_bound<= val <= right_bound for
                    val in [left, right]) and
                    all(bottom_bound <= val <= top_bound for
                    val in [top, bottom])):
                err = (f"All values must be within the grid bounds: "
                      f"left: {left_bound}, right: {right_bound}, "
                      f"top: {top_bound}, bottom: {bottom_bound}.")
                raise ValueError(err) from None

            y_start = int((top_bound - top) / self.cellsize)
            y_end = int((top_bound - bottom) / self.cellsize)
            x_start = int((left - left_bound) / self.cellsize)
            x_end = int((right - left_bound) / self.cellsize)

        elif mode == 'pixel':
            if not (all(0 <= val < height for val in [top, bottom]) and
                    all(0 <= val < width for val in [left, right])):
                err = (f"All values must be within the grid pixel indices: "
                       f"left: 0-{width-1}, right: 0-{width-1}, "
                       f"top: 0-{height-1}, bottom: 0-{height-1}.")
                raise ValueError(err) from None
            y_start, y_end = int(top), int(bottom)
            x_start, x_end = int(left), int(right)

        else:
            err = (f"Invalid mode {mode} provided. Please use one of the "
                   "following modes:\n 1. 'percent' for relative values "
                   "(0.0 to 1.0)\n 2. 'coordinate' for absolute coordinates "
                   "within the grid bounds\n"
                   "3. 'pixel' for pixel indices (0 to grid width/height-1)")
            raise ValueError(err) from None

        # Ensure x_start < x_end and y_start < y_end to handle switched
        # bounds instead of raising an error
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        if y_start > y_end:
            y_start, y_end = y_end, y_start

        result = cp.copy(self)

        # Calculate new transform
        new_x_origin, new_y_origin = self.transform * (x_start, y_start)
        #new_transform = self.transform * (x_start, y_start)
        new_transform = Affine(
            self.transform.a, self.transform.b, new_x_origin,
            self.transform.d, self.transform.e, new_y_origin)
        result.transform = new_transform

        # Calculate new bounds
        xs = np.array([x_start, x_end, x_end, x_start])
        ys = np.array([y_start, y_start, y_end, y_end])

        ws, zs = self.transform * (xs, ys)
        new_left = min(ws)
        new_right = max(ws)
        new_bottom = min(zs)
        new_top = max(zs)

        new_bounds = BoundingBox(new_left, new_bottom, new_right, new_top)
        result.bounds = new_bounds

        # Crop DEM
        result.z = result.z[y_start:y_end, x_start:x_end]
        return result

    # 'Magic' functions:
    # ------------------------------------------------------------------------

    def __eq__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        dem.z = self.z == other.z

        return dem

    def __ne__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        dem.z = self.z != other.z

        return dem

    def __gt__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        dem.z = self.z > other.z

        return dem

    def __lt__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        dem.z = self.z < other.z

        return dem

    def __ge__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        dem.z = self.z >= other.z

        return dem

    def __le__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        dem.z = self.z <= other.z

        return dem

    def __add__(self, other):
        dem = cp.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z + other.z
            return dem

        dem.z = self.z + other
        return dem

    def __sub__(self, other):
        dem = cp.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z - other.z
            return dem

        dem.z = self.z - other
        return dem

    def __mul__(self, other):
        dem = cp.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z * other.z
            return dem

        dem.z = self.z * other
        return dem

    def __div__(self, other):
        dem = cp.copy(self)

        if isinstance(other, self.__class__):
            dem.z = self.z / other.z
            return dem

        dem.z = self.z / other
        return dem

    def __and__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        # Check for invalid values
        if np.any((self.z != 0) & (self.z != 1)) or np.any((other.z != 0) & (other.z != 1)):
            error = "Invalid cell value. 'and' can only compare True (1) and False (0) values."
            raise ValueError(error)

        # Perform element-wise bitwise AND operation
        dem.z = np.logical_and(self.z, other.z)

        return dem

    def __or__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        # Check for invalid values
        if np.any((self.z != 0) & (self.z != 1)) or np.any((other.z != 0) & (other.z != 1)):
            error = "Invalid cell value. 'and' can only compare True (1) and False (0) values."
            raise ValueError(error)

        # Perform element-wise bitwise OR operation
        dem.z = np.logical_or(self.z, other.z)

        return dem

    def __xor__(self, other):
        dem = cp.deepcopy(self)

        if not isinstance(other, self.__class__):
            raise TypeError("Can only compare two GridObjects.")

        if self.columns != other.columns or self.rows != other.rows:
            raise ValueError("Both GridObjects have to be the same size.")

        # Check for invalid values
        if np.any((self.z != 0) & (self.z != 1)) or np.any((other.z != 0) & (other.z != 1)):
            error = "Invalid cell value. 'and' can only compare True (1) and False (0) values."
            raise ValueError(error)

        # Perform element-wise bitwise XOR operation
        dem.z = np.logical_xor(self.z, other.z)

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

    # pylint: disable=unused-argument
    def __array__(self, dtype=None, copy=None):
        if copy:
            return self.z.copy()
        return self.z

    def __str__(self):
        return str(self.z)

    def __repr__(self):

        # Determine the coordinate system
        str_coord = ''
        if self.georef is not None and self.georef.is_projected:
            str_coord = f'coordinate system (Projected): {self.georef}'
        elif self.georef is not None and self.georef.is_geographic:
            str_coord = f'coordinate system (Geographic): {self.georef}'
        else:
            str_coord = f'coordinate system: {self.georef}'

        return f"""name: {self.name}
        path: {self.path}
        rows: {self.rows}
        cols: {self.columns}
        cellsize: {self.cellsize}
        bounds: {self.bounds}
        transform: {self.transform}
        {str_coord}
        maximum z-value: {np.nanmax(self.z)}
        minimum z-value: {np.nanmin(self.z)}
        """
