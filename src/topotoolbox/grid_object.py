"""This module contains the GridObject class.
"""
import copy
from typing import Tuple, List

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors

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
from rasterio.warp import reproject
from rasterio.enums import Resampling

# pylint: disable=no-name-in-module
from . import _grid, _morphology  # type: ignore

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
        self.transform = Affine.identity()
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

        if bc.shape != self.shape:
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
            result_kx = copy.copy(self)
            result_ky = copy.copy(self)
            result_kx.z = kx
            result_ky.z = ky
            return result_kx, result_ky

        slope = np.sqrt(fx**2 + fy**2)
        slope[np.isnan(self.z)] = np.nan

        result = copy.copy(self)
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

        result = copy.copy(self)
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
        """
        dem = np.nan_to_num(self.z)
        p = np.full_like(dem, np.min(dem), order='F')

        prominence: List[float] = []
        indices = []

        while not prominence or prominence[-1] > tolerance:
            diff = dem - p
            prominence.append(np.max(diff))
            indices.append(np.unravel_index(np.argmax(diff), self.shape))

            p[indices[-1]] = dem[indices[-1]]
            if use_hybrid:
                queue = np.zeros_like(dem, dtype=np.int64, order='F')
                _morphology.reconstruct_hybrid(p, queue, dem, self.shape)
            else:
                _morphology.reconstruct(p, dem, self.shape)

        prominence_array = np.array(prominence)
        indices_array = np.array(indices)
        indices_array = indices_array[:, [1, 0]]  # swap columns 0 and 1
        indices_array = indices_array.T  # transpose to get (x, y) instead of (y, x)
        return prominence_array, indices_array

    def hillshade(self,
                  azimuth: float = 315.0,
                  altitude: float = 60.0,
                  exaggerate: float = 1.0):
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

        Returns
        -------
        GridObject
            A GridObject containing the resulting hillshade data
        """

        h = np.zeros_like(self.z)
        nx = np.zeros_like(self.z)
        ny = np.zeros_like(self.z)
        nz = np.zeros_like(self.z)

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
        azimuth_radians = np.atan2(dy,dx)

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

        _grid.hillshade(h, nx, ny, nz, exaggerate * self.z,
                        azimuth_radians, altitude_radians, self.cellsize, self.dims)

        result = copy.copy(self)
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
        if self.crs is not None and self.crs.is_projected:
            print(f"coordinate system (Projected): {self.crs}")
        elif self.crs is not None and self.crs.is_geographic:
            print(f"coordinate system (Geographic): {self.crs}")
        else:
            print(f"coordinate system: {self.crs}")
        print(f"maximum z-value: {np.nanmax(self.z)}")
        print(f"minimum z-value: {np.nanmin(self.z)}")

    def plot(self, ax=None, **kwargs):
        """Plot the GridObject

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes in which to plot the GridObject. If no axes
            are given, the current axes are used.

        **kwargs
            Additional keyword arguments are forwarded to
            matplotlib.axes.Axes.imshow

        Returns
        -------
        matplotlib.image.AxesImage
            The image constructed by imshow
        """
        if ax is None:
            ax = plt.gca()
        return ax.imshow(self.z, **kwargs)

    def plot_hs(self, ax=None,
                elev=None,
                azimuth=315, altitude=60, exaggerate=1,
                filter_method=None, filter_size = 3,
                cmap='terrain', norm = None,
                blend_mode='soft',
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

        """
        if ax is None:
            ax = plt.gca()

        if elev is None:
            shade = self
        elif isinstance(elev, GridObject):
            if not elev.shape == self.shape:
                err = "elev GridObject must have the same shape as the GridObject."
                raise ValueError(err) from None
            shade = elev
        else:
            err = "elev must be a GridObject"
            raise TypeError(err) from None

        if filter_method is not None:
            shade = shade.filter(method=filter_method,kernelsize=filter_size)

        h = shade.hillshade(azimuth, altitude, exaggerate)
        cmap = plt.get_cmap(cmap)

        if norm is None:
            norm = colors.Normalize(vmin=np.nanmin(self.z),vmax=np.nanmax(self.z))

        base = cmap(norm(self.z))
        top = np.expand_dims(np.clip(h,0,1),2)
        if blend_mode == "multiply":
            rgb = base * top
        elif blend_mode == "overlay":
            rgb = np.where(base < 0.5, 2*base*top, 1 - 2*(1-base)*(1-top))
        elif blend_mode == "soft":
            rgb = (1 - 2*top)*base**2 + 2 * top * base
        else:
            raise ValueError("blend_mode not supported") from None

        return ax.imshow(np.clip(rgb,0,1), **kwargs)

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

        """
        result = copy.copy(self)

        labels = self.z
        u, indices = np.unique(labels, return_inverse=True)
        rng = np.random.default_rng(seed)
        result.z = rng.permutation(u)[indices]

        return result

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
