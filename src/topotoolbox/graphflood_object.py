"""This module contains the GFObject class.
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module
from . import graphflood as tgf
from .grid_object import GridObject

__all__ = ['GFObject']


class GFObject():
    """GraphFlood object for stationnary, large-scale ready hydrodynamic modeling.

    See Gailleton et al. 2024 for details (https://esurf.copernicus.org/articles/12/1295/2024/)

    Contains flow information and parameters for water routing on a DEM.
    Handles boundary conditions, precipitation, and Manning roughness coefficients.
    """

    def __init__(self,
        grid: GridObject,
        bcs: np.ndarray | GridObject | None = None,
        p: float | np.ndarray | GridObject = 10 * 1e-3 / 3600,
        manning: float | np.ndarray | GridObject = 0.033
    ):
        """Initialize flood routing object.

        Parameters
        ----------
        grid : GridObject
            Topographic grid object
        bcs : ndarray or GridObject, optional
            Boundary conditions (None for default open boundaries)
        p : float or ndarray or GridObject, optional
            Precipitation rate [m/s] (default: 10mm/h)
        manning : float or ndarray or GridObject, optional
            Manning roughness coefficient (default: 0.033)
        """
        # Deep copy grid and ensure float64 precision for elevation
        self.grid = deepcopy(grid)
        self.grid.z = self.grid.z.astype(np.float64)

        # Initialize water height GridObject
        self._hw = deepcopy(self.grid)
        self._hw.z = np.zeros_like(self.grid.z)

        # Process boundary conditions
        if bcs is None:
            # Default: open boundaries on edges, internal cells are normal flow
            self._bcs = np.ones((grid.rows, grid.columns), dtype=np.uint8)
            self._bcs[[0, -1], :] = 3  # Open boundary on top/bottom
            self._bcs[:, [0, -1]] = 3  # Open boundary on left/right
        else:
            # Validate boundary condition dimensions
            if bcs.shape != grid.shape:
                raise RuntimeError(
                    "Boundary conditions must match grid dimensions"
                )
            if isinstance(bcs, GridObject):
                self._bcs = bcs.z
            else:
                self._bcs = bcs


        # Process precipitation input
        if isinstance(p, np.ndarray):
            if p.shape != grid.shape:
                raise RuntimeError(
                    "Precipitation array must match grid dimensions"
                )
            self._precipitations = p.ravel(order="C")
        elif isinstance(p, GridObject):
            if p.shape != grid.shape:
                raise RuntimeError(
                    "Precipitation GridObject must match grid dimensions"
                )
            self._precipitations = p.z
        else:
            # Uniform precipitation across grid
            self._precipitations = np.full_like(self.grid.z, p)

        # Process Manning roughness coefficient
        if isinstance(manning, np.ndarray):
            if manning.shape != grid.shape:
                raise RuntimeError(
                    "Manning coefficient array must match grid dimensions"
                )
            self._manning = manning
        elif isinstance(manning, GridObject):
            if manning.shape != grid.shape:
                raise RuntimeError(
                    "Manning coefficient GridObject must match grid dimensions"
                )
            self._manning = manning.z
        else:
            # Uniform Manning coefficient across grid
            self._manning = np.full_like(self.grid.z, manning)

        # Placeholder for the results from the model
        self.res: dict = {}

    # Water height getters and setters
    @property
    def hw(self) -> GridObject:
        """Get water height GridObject."""
        return self._hw

    @hw.setter
    def hw(self, value: np.ndarray | GridObject) -> None:
        """Set water height array."""
        if isinstance(value, GridObject):
            if value.shape != self.grid.shape:
                raise ValueError("Water height GridObject must match grid dimensions")
            self._hw = deepcopy(value)
            self._hw.z = self._hw.z.astype(np.float64)
        else:
            if value.shape != self.grid.shape:
                raise ValueError("Water height array must match grid dimensions")
            self._hw.z = value.astype(np.float64)

    # Boundary conditions getters and setters
    @property
    def bcs(self) -> np.ndarray:
        """Get boundary conditions array."""
        return self._bcs

    @bcs.setter
    def bcs(self, value: np.ndarray | GridObject) -> None:
        """Set boundary conditions array."""
        if isinstance(value, GridObject):
            if value.shape != self.grid.shape:
                raise ValueError("Boundary conditions must match grid dimensions")
            self._bcs = value.z.ravel(order="C").astype(np.uint8)
        else:
            if value.size != self.grid.z.size:
                raise ValueError("Boundary conditions must match grid size")
            self._bcs = value.ravel(order="C").astype(np.uint8)

    # Precipitation getters and setters
    @property
    def precipitations(self) -> np.ndarray:
        """Get precipitation array."""
        return self._precipitations

    @precipitations.setter
    def precipitations(self, value: float | np.ndarray | GridObject) -> None:
        """Set precipitation array."""
        if isinstance(value, np.ndarray):
            if value.size != self.grid.z.size:
                raise ValueError("Precipitation array must match grid size")
            self._precipitations = value.ravel(order="C")
        elif isinstance(value, GridObject):
            if value.shape != self.grid.shape:
                raise ValueError("Precipitation GridObject must match grid dimensions")
            self._precipitations = value.z.ravel(order="C")
        else:
            self._precipitations = np.full_like(self.grid.z.ravel(), value)

    # Manning coefficient getters and setters
    @property
    def manning(self) -> np.ndarray:
        """Get Manning roughness coefficient array."""
        return self._manning

    @manning.setter
    def manning(self, value: float | np.ndarray | GridObject) -> None:
        """Set Manning roughness coefficient array."""
        if isinstance(value, np.ndarray):
            if value.size != self.grid.z.size:
                raise ValueError("Manning coefficient array must match grid size")
            self._manning = value.ravel(order="C")
        elif isinstance(value, GridObject):
            if value.shape != self.grid.shape:
                raise ValueError("Manning coefficient GridObject must match grid dimensions")
            self._manning = value.z.ravel(order="C")
        else:
            self._manning = np.full_like(self.grid.z.ravel(), value)

    def run_n_iterations(self, dt: float = 1e-3, sfd: bool = False,
                         d8: bool = True, n_iterations: int = 100):
        """Run graphflood model for n iterations.

        Executes the hydrodynamic simulation using the graphflood algorithm.
        Updates water height and stores flow outputs in self.res.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds (default: 1e-3)
        sfd : bool, optional
            Use single flow direction routing (default: False)
        d8 : bool, optional
            Use D8 flow routing (default: True)
        n_iterations : int, optional
            Number of simulation iterations (default: 100)
        """
        # Run the graphflood simulation with current parameters
        self.res = tgf.run_graphflood(
            self.grid,
            initial_hw=self._hw.z,
            bcs=self._bcs,
            dt=dt,
            p=self._precipitations,
            manning=self._manning,
            sfd=sfd,
            d8=d8,
            n_iterations=n_iterations)

        # Update water height from results and remove from res dict
        if isinstance(self.res['hw'], GridObject):
            self._hw = self.res['hw']
        else:
            self._hw.z = self.res['hw']
        del self.res['hw']

    # Model output getters (read-only)
    @property
    def qvol_i(self) -> GridObject:
        """Incoming discharge GridObject [m³/s]

        Returns
        -------
        GridObject
            Incoming discharge for each cell

        Raises
        ------
        RuntimeError
            If model hasn't been run yet
        """
        if not self.res:
            raise RuntimeError("Model must be run before accessing results")
        return self.res['Qi']

    @property
    def qvol_o(self) -> GridObject:
        """Outgoing discharge GridObject [m³/s]

        Returns
        -------
        GridObject
            Outgoing discharge for each cell

        Raises
        ------
        RuntimeError
            If model hasn't been run yet
        """
        if not self.res:
            raise RuntimeError("Model must be run before accessing results")
        return self.res['Qo']

    @property
    def q(self) -> GridObject:
        """Specific outgoing discharge GridObject [m²/s]

        Returns
        -------
        GridObject
            Outgoing discharge per unit width for each cell

        Raises
        ------
        RuntimeError
            If model hasn't been run yet
        """
        if not self.res:
            raise RuntimeError("Model must be run before accessing results")
        return self.res['qo']

    @property
    def u(self) -> GridObject:
        """Flow velocity GridObject [m/s]

        Returns
        -------
        GridObject
            Flow velocity for each cell

        Raises
        ------
        RuntimeError
            If model hasn't been run yet
        """
        if not self.res:
            raise RuntimeError("Model must be run before accessing results")
        return self.res['u']

    @property
    def sw(self) -> GridObject:
        """Wetted area GridObject [m²]

        Returns
        -------
        GridObject
            Wetted surface area for each cell

        Raises
        ------
        RuntimeError
            If model hasn't been run yet
        """
        if not self.res:
            raise RuntimeError("Model must be run before accessing results")
        return self.res['Sw']

    # API methods (equivalent to property getters)
    def get_qvol_i(self) -> GridObject:
        """Incoming discharge GridObject [m³/s]
        """
        return self.qvol_i

    def get_qvol_o(self) -> GridObject:
        """Outgoing discharge GridObject [m³/s]
        """
        return self.qvol_o

    def get_q(self) -> GridObject:
        """Specific outgoing discharge GridObject [m²/s]
        """
        return self.q

    def get_u(self) -> GridObject:
        """Flow velocity GridObject [m/s]
        """
        return self.u

    def get_sw(self) -> GridObject:
        """Wetted area GridObject [m²]
        """
        return self.sw

    def compute_tau(self, gravity: float = 9.81, flow_density: float = 1000) -> GridObject:
        """Compute 2D maps of river shear stress (following Manning's approximation).

        Parameters
        ----------
        gravity : float, optional
            Gravitational acceleration [m/s²] (default: 9.81)
        flow_density : float, optional
            Water density [kg/m³] (default: 1000)

        Returns
        -------
        GridObject
            GridObject containing shear stress values [Pa]

        Raises
        ------
        RuntimeError
            If model hasn't been run yet
        """
        if not self.res:
            raise RuntimeError("Model must be run before computing shear stress")

        # Create mask for valid flow cells
        mask = (self._bcs > 0) & (self._hw.z > 0) & (self.sw.z > 0)

        # River shear stress calculation
        tau = np.zeros_like(self._hw.z)
        tau[mask] = self.sw.z[mask] * self._hw.z[mask] * flow_density * gravity

        # Return as GridObject with same georeferencing as input grid
        tau_grid = deepcopy(self.grid)
        tau_grid.z = tau
        return tau_grid


    def get_convergence_metrics(self):
        """Compute convergence metrics for flow conservation analysis.

        Calculates the ratio of outgoing to incoming discharge for each cell
        to assess flow conservation and model convergence quality. Data is binned
        by log10(incoming discharge) and comprehensive percentile statistics are
        computed for each bin.

        Returns
        -------
        tuple
            - convergence_map_grid : GridObject
                GridObject containing convergence ratios (Qo/Qi) for each cell
            - conv : dict
                Dictionary containing:
                - 'bin_centers': discharge bin centers [m³/s]
                - 'bin_counts': number of cells in each bin
                - 'percentile_X': percentile X of convergence ratios for each bin
                  (where X = 1, 5, 9, 13, ..., 97 - every 4th percentile)

        Raises
        ------
        RuntimeError
            If model hasn't been run yet
        """
        if not self.res:
            raise RuntimeError("Model must be run before accessing convergence metrics")

        # Create mask for valid flow cells (excluding boundaries)
        mask = self._bcs > 0

        # Initialize convergence map with default value of 1 (perfect conservation)
        convergence_map = np.zeros_like(self._hw.z)
        convergence_map[mask] = 1.0

        # Update mask to include cells with positive incoming discharge
        mask = mask & (self.qvol_i.z >= 0)

        # Calculate convergence ratio (outgoing/incoming discharge)
        # Values close to 1 indicate good conservation
        with np.errstate(divide='ignore', invalid='ignore'):
            convergence_map[mask] = self.qvol_o.z[mask] / self.qvol_i.z[mask]

        # Return as GridObject with same georeferencing as input grid
        convergence_map_grid = deepcopy(self.grid)
        convergence_map_grid.z = convergence_map

        # Calculate binned percentile statistics based on log10(qvol_i)
        conv = {}

        # Flatten arrays for binning
        qvol_i_flat = self.qvol_i.z.ravel()
        convergence_flat = convergence_map.ravel()

        # Create mask for valid data (non-NaN and positive discharge)
        valid_mask = ~np.isnan(qvol_i_flat) & ~np.isnan(convergence_flat) & (qvol_i_flat > 0)
        qvol_i_valid = qvol_i_flat[valid_mask]
        convergence_valid = convergence_flat[valid_mask]

        if len(qvol_i_valid) > 0:
            # Transform to log10 space for binning
            log_qvol_i_valid = np.log10(qvol_i_valid)

            # Create linear bins in log10 space
            n_bins = 20
            log_min = np.min(log_qvol_i_valid)
            log_max = np.max(log_qvol_i_valid)
            bin_edges = np.linspace(log_min, log_max, n_bins + 1)

            # Digitize log10(qvol_i) values into bins
            bin_indices = np.digitize(log_qvol_i_valid, bin_edges)

            # Define percentile values to calculate (every 4th from 1 to 100)
            percentile_values = np.arange(1, 101, 4)

            # Initialize storage for results
            bin_centers = []
            percentiles_dict = {f'percentile_{p}': [] for p in percentile_values}
            bin_counts = []

            # Calculate statistics for each bin
            for i in range(1, len(bin_edges)):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    bin_convergence = convergence_valid[bin_mask]
                    bin_log_qvol = log_qvol_i_valid[bin_mask]

                    # Store bin center in original (non-log) space
                    bin_centers.append(10**np.mean(bin_log_qvol))

                    # Calculate all percentiles for this bin
                    for p in percentile_values:
                        percentiles_dict[f'percentile_{p}'].append(
                            np.percentile(bin_convergence, p))

                    bin_counts.append(len(bin_convergence))

            # Convert to numpy arrays
            conv['bin_centers'] = np.array(bin_centers)
            conv['bin_counts'] = np.array(bin_counts)

            # Add all percentiles to conv dictionary
            for p in percentile_values:
                conv[f'percentile_{p}'] = np.array(percentiles_dict[f'percentile_{p}'])

        else:
            # No valid data - initialize empty arrays
            conv['bin_centers'] = np.array([])
            conv['bin_counts'] = np.array([])

            # Initialize empty arrays for all percentiles
            percentile_values = np.arange(1, 101, 4)
            for p in percentile_values:
                conv[f'percentile_{p}'] = np.array([])
        return convergence_map_grid, conv


    def plot_convergence(self):
        """Plot convergence metrics as spatial map and log10(discharge)-binned statistics.

        Creates a two-panel plot showing:
        1. Spatial map of convergence ratios (Qo/Qi)
        2. Convergence percentiles vs log10(incoming discharge) with uncertainty bands

        The convergence analysis bins data by log10(qvol_i) and calculates every 4th
        percentile from 1 to 100, providing comprehensive statistical information
        about flow conservation across different discharge magnitudes.

        Returns
        -------
        tuple
            - fig : matplotlib.figure.Figure
                The figure object
            - ax : array of matplotlib.axes.Axes
                Array containing the two subplot axes [spatial_map, percentiles_plot]
        """
        convergence_map_grid, conv = self.get_convergence_metrics()

        fig,ax = plt.subplots(1,2)

        self.grid.plot_hs(ax=ax[0])

        # Getting the extent
        im = ax[0].images[0]        # first image in this Axes
        ext = im.get_extent()

        im = ax[0].imshow(convergence_map_grid.z, cmap='RdBu_r',
                         vmin=0.8, vmax=1.2, extent=ext)
        plt.colorbar(im, label = 'Convergence (perfect = 1.)')


        # Plot convergence percentiles vs discharge bins
        if len(conv['bin_centers']) > 0:
            # Plot key percentiles with fill_between for uncertainty bands
            ax[1].fill_between(conv['bin_centers'], conv['percentile_5'],
                             conv['percentile_97'], alpha=0.2, color='lightblue',
                             label='5th-97th percentile')
            ax[1].fill_between(conv['bin_centers'], conv['percentile_25'],
                             conv['percentile_73'], alpha=0.3, color='blue',
                             label='25th-73rd percentile')
            ax[1].plot(conv['bin_centers'], conv['percentile_49'],
                      color='red', linewidth=2, label='~Median (49th percentile)')
            ax[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7,
                         label='Perfect conservation')
            ax[1].set_xscale('log')
            ax[1].legend()

        ax[1].grid(True, alpha=0.3)
        ax[1].set_xlabel('Incoming Discharge [m³/s]')
        ax[1].set_ylabel('Convergence Ratio (Qo/Qi)')

        return fig, ax
