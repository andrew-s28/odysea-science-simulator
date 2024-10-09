import numpy as np
import xarray as xr

from odysim import utils

# class WebGriddedModel:
# TODO: Implement the LLC 4320 model through the xmitgcm llcreader interface
#       or some other cloud readable interface.


class GriddedModel:

    """

    A class that holds functions for loading and co-locating ocean/atmosphere model data.
    Ocean model data is expected to be gridded in lat/lon with individual files for each
    time step.

    Take this code as a starting point and adapt to your own specific model as needed.

    """
    def __init__(self, files=None, min_time=None, max_time=None, grid_file='D:/avg/croco_grd_nemoified.nc.1b'):
        """
        Initialize a GriddedModel object.

        Args:
            model_folder (str): Top-level folder for model data. Contains sub folders for each variable.
            u_folder (str): Sub-folder containing model U current data (East-West currents).
            v_folder (str): Sub-folder containing model V current data (North-South currents).
            tau_x_folder (str): Sub-folder containing model U wind stress current data (East-West wind stress).
            tau_y_folder (str): Sub-folder containing model V wind stress current data (North-South wind stress).
            u_varname (str): Variable name inside model netcdf files for U current.
            v_varname (str): Variable name inside model netcdf files for V current.
            tau_x_varname (str): Variable name inside model netcdf files for U wind stress.
            tau_y_varname (str): Variable name inside model netcdf files for V wind stress.
            search_string (str): File extension for model data files.
            preprocess (function): function to pass to xarray.open_mfdataset for preprocessing.
            n_files (int): number of files to load, 0:n_files. Used to reduce load if many files are available in the model folder.

        Returns:
            GriddedModel obect

        """
        if files is None:
            raise ValueError("files must be provided")

        self.ds = xr.open_mfdataset(files)
        self.grid = xr.open_dataset(grid_file)
        self.ds = self.convert_time(self.ds, self.grid, min_time, max_time)
        self.ds = self.ds.assign_coords(
            {
                'lat': self.grid.lat_psi[:, 0],
                'lon': self.grid.lon_psi[0, :]
            }
        ).swap_dims(
            {
                'eta_psi': 'lat',
                'xi_psi': 'lon'
            }
        ).rename(
            {
                'eta_rho': 'lat',
                'xi_rho': 'lon',
                'eta_v': 'lat',
                'xi_u': 'lon'
            }
        )

        # self.ds.load()

        self.U = self.ds['u']
        self.V = self.ds['v']
        self.W = self.ds['w']
        self.TX = self.ds['sustr']
        self.TY = self.ds['svstr']

        self.interp_kwargs = {
            'bounds_error': False,
            'fill_value': np.nan,
        }

        # self.U = xr.open_mfdataset(u_files,parallel=True,preprocess=preprocess)
        # self.V = xr.open_mfdataset(v_files,parallel=True,preprocess=preprocess)
        # self.TX = xr.open_mfdataset(tau_x_files,parallel=True,preprocess=preprocess)
        # self.TY = xr.open_mfdataset(tau_y_files,parallel=True,preprocess=preprocess)

        # self.u_varname = u_varname
        # self.v_varname = v_varname
        # self.tau_x_varname = tau_x_varname
        # self.tau_y_varname = tau_y_varname

    def convert_time(self, ds: xr.Dataset, grid: xr.Dataset, min_time=None, max_time=None) -> xr.Dataset:
        ds = ds.copy()
        ds['time'] = np.datetime64('1950-01-01') + [np.timedelta64(int(t), 's') for t in ds.scrum_time.values] - np.timedelta64(90, 's')
        ds = ds.drop_dims('auxil')
        min_time = min_time - np.timedelta64(10, 'D') if min_time is not None else None
        max_time = max_time + np.timedelta64(10, 'D') if max_time is not None else None
        ds = ds.sel(time=slice(min_time, max_time))
        # grid file uses eta_u, xi_v, but model uses eta_rho and xi_rho since they are redundant, which screws up masking with DataArrays
        ds['u'] = ds['u'].where(grid.mask_u.values == 1)
        ds['v'] = ds['v'].where(grid.mask_v.values == 1)
        ds['w'] = ds['w'].where(grid.mask_rho.values == 1)
        ds['temp'] = ds['temp'].where(grid.mask_rho.values == 1)
        ds['salt'] = ds['salt'].where(grid.mask_rho.values == 1)
        return ds

    def colocatePoints(self, lats: np.array, lons: np.array, times: np.array):
        """
        Colocate model data to a set of lat/lon/time query points.
            Ensure that lat/lon/time points of query exist within the loaded model data.

        Args:
            lats (numpy.array): latitudes in degrees
            lons (numpy.array): longitudes in degrees
            times (numpy.array): times represented as np.datetime64

        Returns:
           Model data linearly interpolated to the lat/lon/time query points.
           u: colocated model u currents.
           v: colocated model v currents.
           tx: colocated model u wind stress.
           ty: colocated model v wind stress.

        """
        times = np.array(times)
        if len(times) == 0:
            return [], []

        ds_u = self.U.interp(
            time=times,
            lat=lats,
            lon=lons,
            method='linear',
            kwargs=self.interp_kwargs,
        )

        ds_v = self.V.interp(
            time=xr.DataArray(times.flatten(), dims='z'),
            lat=xr.DataArray(lats.flatten(), dims='z'),
            lon=xr.DataArray(lons.flatten(), dims='z'),
            method='linear',
            kwargs=self.interp_kwargs,
        )

        ds_w = self.W.interp(
            time=xr.DataArray(times.flatten(), dims='z'),
            lat=xr.DataArray(lats.flatten(), dims='z'),
            lon=xr.DataArray(lons.flatten(), dims='z'),
            method='linear',
            kwargs=self.interp_kwargs,
        )

        ds_tx = self.TX.interp(
            time=xr.DataArray(times.flatten(), dims='z'),
            lat=xr.DataArray(lats.flatten(), dims='z'),
            lon=xr.DataArray(lons.flatten(), dims='z'),
            method='linear',
            kwargs=self.interp_kwargs,
        )

        ds_ty = self.TY.interp(
            time=xr.DataArray(times.flatten(), dims='z'),
            lat=xr.DataArray(lats.flatten(), dims='z'),
            lon=xr.DataArray(lons.flatten(), dims='z'),
            method='linear',
            kwargs=self.interp_kwargs,
        )

        u = np.reshape(ds_u.values, np.shape(lats))
        v = np.reshape(ds_v.values, np.shape(lats))
        w = np.reshape(ds_w.values, np.shape(lats))
        tx = np.reshape(ds_tx.values, np.shape(lats))
        ty = np.reshape(ds_ty.values, np.shape(lats))

        return u, v, w, tx, ty

    def colocateSwathCurrents(self, orbit):
        """
        Colocate model current data to a swath (2d continuous array) of lat/lon/time query points.
            Ensure that lat/lon/time points of query exist within the loaded model data.

        Args:
            orbit (object): xarray dataset orbit generated via the orbit.getOrbit() call.
        Returns:
           original orbit containing model data linearly interpolated to the orbit swath.
                   new data is contained in u_model, v_model

        """
        ds = self.ds.interp(
            time=orbit.sample_time,
            lat=orbit.lat,
            lon=orbit.lon,
            method='linear',
            kwargs=self.interp_kwargs,
        ).reset_coords()

        orbit['u_model'] = ds['u']
        orbit['v_model'] = ds['v']
        orbit['w_model'] = ds['w']

        return orbit

    def colocateSwathWinds(self, orbit):

        """
        Colocate model wind data to a swath (2d continuous array) of lat/lon/time query points.
            Ensure that lat/lon/time points of query exist within the loaded model data.

        Args:
            orbit (object): xarray dataset orbit generated via the orbit.getOrbit() call.
        Returns:
           original orbit containing model data linearly interpolated to the orbit swath.
                   new data is contained in u10_model, v10_model, tx_model, ty_model, wind_speed_model, wind_dir_model

        """

        lats = orbit['lat'].values.flatten()
        lons = orbit['lon'].values.flatten()
        times = orbit['sample_time'].values.flatten()

        ds_tx = self.TX.interp(
            time=xr.DataArray(times, dims='z'),
            lat=xr.DataArray(lats, dims='z'),
            lon=xr.DataArray(lons, dims='z'),
            method='linear',
            kwargs=self.interp_kwargs,
        )

        ds_ty = self.TY.interp(
            time=xr.DataArray(times, dims='z'),
            lat=xr.DataArray(lats, dims='z'),
            lon=xr.DataArray(lons, dims='z'),
            method='linear',
            kwargs=self.interp_kwargs,
        )

        tx_interp = np.reshape(ds_tx.values, np.shape(orbit['lat'].values))
        ty_interp = np.reshape(ds_ty.values, np.shape(orbit['lat'].values))

        wind_speed = utils.stressToWind(np.sqrt(tx_interp**2 + ty_interp**2))
        wind_dir = np.arctan2(tx_interp, ty_interp)  # in rad
        u10 = wind_speed * np.sin(wind_dir)
        v10 = wind_speed * np.cos(wind_dir)

        orbit = orbit.assign(
            {
                'u10_model': (['along_track', 'cross_track'], u10),
                'v10_model': (['along_track', 'cross_track'], v10)
            }
        )

        orbit = orbit.assign(
            {
                'tx_model': (['along_track', 'cross_track'], tx_interp),
                'ty_model': (['along_track', 'cross_track'], ty_interp)
            }
        )

        orbit = orbit.assign(
            {
                'wind_speed_model': (['along_track', 'cross_track'], wind_speed),
                'wind_dir_model': (['along_track', 'cross_track'], wind_dir*180/np.pi)
            }
        )

        return orbit
