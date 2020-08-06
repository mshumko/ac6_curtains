import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
from datetime import datetime
import dateutil.parser
import scipy.spatial
import pathlib

import skyfield.api
import cdflib # https://github.com/MAVENSDC/cdflib

from ac6_curtains import dirs
import IRBEM

class THEMIS_ASI:
    def __init__(self, site, time):
        """
        This class handles finding and loading the All-Sky Imager images and 
        calibration files. Once loaded, the 

        Attributes
        ----------
        asi_dir : str
            The ASI directory
        site : str
            Shorthand ASI site name
        t : datetime.datetime
            The time used to lookup and load the ASI data.
        asi : cdflib.cdfread.CDF object
            The all sky imager object.
        keys : list
            List of ASI data keys
        cal : dict
            Once load_themis_cal() is called, cal will be avaliable and it 
            contains the metadata about the ground station such as lat, lon,
            alt, and the mapping between image pixel indicies to AzEl.

        Methods
        -------
        load_themis_asi()
            Looks for and loads a THEMIS ASI file that contains self.site
            and the date + hour from self.t (the time argument for __init__)
        get_asi_frames()
            Given a time, or an array of times, return the closest ASI image
            frames and time(s) within the max_diff_s time threshold.
        load_themis_cal()
            Looks for, and loads the THEMIS ASI azimuth and elevation (AzEl)
            calibration files.
        plot_themis_asi_frame()
            Plots one ASI frame given a time.
        plot_azel_contours()
            Plots the AzEl contours on top of the ASI frame, useful for developing
            and debugging.
        """
        self.asi_dir = dirs.ASI_DIR
        self.site = site.lower()
        self.t = time

        if isinstance(self.t, str):
            # Convert to datetime object if passes a tring time
            self.t = dateutil.parser.parse(self.t)
        self.load_themis_asi()
        return

    def load_themis_asi(self):
        """
        Load the THEMIS ASI data and convert the time array to a 
        datetime array. The data is searched in self.asi_dir with
        the following glob string: 
        f'*{self.site}_{self.t.strftime("%Y%m%d%H")}_v*.cdf'.
        The frame times are converted to datetime objects in 
        self.time and self.imgs is the 3D ASI data cube.

        Parameters
        ----------
        None

        Returns
        -------
        self.asi : cdflib.cdfread.CDF object
            The ASI object with the data.
        self.time : datetime.datetime array
            The array of ASI frame time stamps.
        self.imgs : 3D array
            A THEMIS ASI frame image cube with the first axis indexes 
            frames at times corrsponding to self.time.
        """
        file_glob_str = f'*{self.site}_{self.t.strftime("%Y%m%d%H")}_v*.cdf'
        cdf_paths = list(pathlib.Path(self.asi_dir).rglob(file_glob_str))
        assert len(cdf_paths) == 1, (f'{len(cdf_paths)} THEMIS ASI paths found'
            f' for search string {file_glob_str} at {self.asi_dir}')
        cdf_path = cdf_paths[0]

        self.asi = cdflib.cdfread.CDF(cdf_path)
        self.keys = self.asi.cdf_info()['zVariables'] # Key variables.

        # Convert time
        try:
            self.time = cdflib.cdfepoch().to_datetime(self.asi[f"thg_asf_{self.site}_epoch"][:], 
                                            to_np=True)
        except ValueError as err:
            if 'not found' in str(err):
                print(cdf_path, '\n', self.asi.cdf_info()['zVariables'])
                raise

        # Copy images into another variable
        try:
            self.imgs = self.asi[f"thg_asf_{self.site}"]
        except ValueError as err:
            if str(err) == 'read length must be non-negative or -1':
                print(cdf_path, '\n', self.asi[f"thg_asf_{self.site}"].shape)
                raise
        return self.asi, self.time, self.imgs

    def get_asi_frames(self, t, max_diff_s=5):
        """ 
        Given the array of times, t, return an image cube (or 2D array) 
        of ASI images near those times, as well as the ASI times that 
        correpond to each image.

        Parameters
        ----------
        t : datetime.datetime
            A datetime.datetime object or an array of datetime.datetime 
            objects that is used to find the nearest ASI frames and 
            frame times.
        max_diff_s : float (optional)
            Maximum differnece between t and the ASI frame time array.

        Returns
        -------
        frames: 2D or 3D array
            The ASI frames. If t is a single time, the frames will be
            one 2D ASI frame. If t is an array of times, frames will 
            be a 3D ASI frame cube with the first axis time.        
        frame_times : datetime.datetime or an array of datetime.datetimes
            The ASI frame times closest to each time in t. If t is a 
            single time, the frame_times will be one time stamp. If t 
            is an array of times, the frame_times will also be an array 
            of times.
        """
        if not hasattr(t, '__len__'):
            t = [t]

        frame_idx = -10000*np.ones(len(t), dtype=int)
        frame_times = np.nan*np.ones(len(t), dtype=object)

        for i, t_i in enumerate(t):
            dt = self.time-t_i
            dt_sec = np.abs([dt_i.total_seconds() for dt_i in dt])
            frame_idx[i] = np.argmin(dt_sec)
            frame_times[i] = self.time[frame_idx[i]]

            if np.abs((frame_times[i] - t_i).total_seconds()) > max_diff_s:
                raise ValueError(f'No THEMIS ASI image found within {max_diff_s} seconds.')
        if len(t) == 1:
            return self.imgs[frame_idx[0], :, :], frame_times[0]
        else:
            return self.imgs[frame_idx, :, :], frame_times

    def load_themis_cal(self, cal_file_name=None):
        """
        Loads the THEMIS ASI calibration file. This is a simple program
        that loads only the most recent calibration file, if multiple
        files exist. Otherwise if cal_file_name is specified, that exact 
        file name will be searched for.

        Parameters
        ----------
        cal_file_name : str (optional)
            The exact calibration filename to overwrite the default glob
            search string: cal_file_name = f'thg_l2_asc_{self.site}_*.cdf'.

        Returns
        -------
        self.cal : dict
            A calibration dictionary inluding: the az and el 
            calibration data, as well as the station lat, lon, 
            alt_m, site name, calibration filename, and 
            calibration epoch.
        """
        if cal_file_name is None:
            cal_file_name = f'thg_l2_asc_{self.site}_*.cdf'
            cal_paths = sorted(pathlib.Path(self.asi_dir).glob(cal_file_name))
            cal_path = cal_paths[-1] # Grab the most recent cal file.
        else:
            cal_path = pathlib.Path(self.asi_dir, cal_file_name)
            

        cal = cdflib.cdfread.CDF(cal_path)
        az = cal[f"thg_asf_{self.site}_azim"][0]
        el = cal[f"thg_asf_{self.site}_elev"][0]
        lat = cal[f"thg_asc_{self.site}_glat"]
        lon = (cal[f"thg_asc_{self.site}_glon"] + 180) % 360 - 180  # [0,360] -> [-180,180]
        alt_m = cal[f"thg_asc_{self.site}_alti"]
        x = y = cal[f"thg_asf_{self.site}_c256"]
        time = datetime.utcfromtimestamp(cal[f"thg_asf_{self.site}_time"][-1])

        self.cal = {
            "az":az, "el":el, 'coords':x, "lat": lat, "lon": lon, "alt_m": alt_m, 
            "site": self.site, "calfilename": cal_path.name, "caltime": time
                }
        return self.cal

    def plot_themis_asi_frame(self, t0, ax=None, max_diff_s=60, imshow_vmin=None, 
                            imshow_vmax=None, imshow_norm='log', colorbar=True):
        """
        Plot a ASI frame with a time nearest to t0. The subplot and the image
        are set as class attributes.

        Parameters
        ----------
        t0 : str or datetime.datetime
            The time to plot the ASI frame for.
        ax : plt.subplot (optional)
            The subplot object to plot the ASI frame.
        max_diff_s : float (optional)
            Maximum differnece between t0 and the ASI frame time array.
        imshow_vmin : float (optional)
            The pixel intensity value corresponding to the maximum color 
            value for the ASI frame.
        imshow_vmin : float (optional)
            The pixel intensity value corresponding to the minimum color 
            value for the ASI frame.
        imshow_norm : str
            The color normalization. Can be either linear or log.
        colorbar : bool
            Flag to plot the horizontal colorbar.

        Returns
        -------
        frame_time : datetime.datetime
            The time of the plotted frame. 
        """
        if ax is None:
            _, self.ax = plt.subplots()
        else:
            self.ax = ax

        if isinstance(t0, str):
            # Convert to datetime object if passes a tring time
            t0 = dateutil.parser.parse(t0) 

        frame, frame_time = self.get_asi_frames(t0, max_diff_s=max_diff_s)

        if self.cal['lon'] < 0:
            lon_label='W'
        else:
            lon_label='E'

        if imshow_norm == 'linear':
            norm = None
        elif imshow_norm == 'log':
            norm = matplotlib.colors.LogNorm()
        else:
            raise ValueError('The imshow_norm kwarg must be "linear" or "log".')
        
        title_text = (f'{self.site.upper()} ({round(self.cal["lat"])}N, '
                     f'{np.abs(round(self.cal["lon"]))}{lon_label})\n{frame_time}')
        self.hi = self.ax.imshow(frame, cmap="gray", 
                                origin="lower", interpolation="none",
                                vmin=imshow_vmin, vmax=imshow_vmax, norm=norm)
        if colorbar:
            plt.colorbar(self.hi, ax=self.ax, orientation='horizontal')
        self.ht = self.ax.set_title(title_text, color="k")
        self.ax.axis('off')
        return frame_time

    def plot_azel_contours(self, ax=None):
        """ 
        Superpose the azimuth and elivation contours on the ASI frame,
        on the self.ax subplot.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if ax is None:
            _, self.ax = plt.subplots()
        else:
            self.ax = ax
   
        az_contours = self.ax.contour(self.cal["az"], colors='blue', linestyles='dotted', 
                        levels=np.arange(0, 360, 90), alpha=1)
        el_contours = self.ax.contour(self.cal["el"], colors='red', linestyles='dotted', 
                        levels=np.arange(0, 91, 30), alpha=1)
        plt.clabel(az_contours, inline=True, fontsize=8)
        plt.clabel(el_contours, inline=True, fontsize=8, rightside_up=True)
        return

class THEMIS_ASI_map_azel(THEMIS_ASI):
    def __init__(self, site, time):
        """
        This class uses the THEMIS_ASI class to map the lat-lon-alt location
        of a satellite, ballon, etc. to AzEl coordinates using the THEMIS
        ASI calibration data and the Skyfield library. 

        Attributes
        ----------
        asi_dir : str
            The ASI directory
        site : str
            Shorthand ASI site name
        time : datetime.datetime
            The time used to lookup and load the ASI data.

        Methods
        -------
        map_satazel_to_asiazel()
            Maps from the satellite azimuth and elevation (azel) coordimates
            and finds the nearest all sky imager (ASI) azel calibration pixel. 
        map_lla_to_sat_azel()
            Maps the satellite lat, lon, and alt (km) coorinates to azel 
            coordinates for the ground station. The Skyfield library is used
            for this mapping.
        """
        super().__init__(site, time)
        return

    def map_lla_to_asiazel(self, lla):
        """
        Wrapper for map_satazel_to_asiazel() and map_lla_to_sat_azel().
        """
        self.sat_azel = self.map_lla_to_sat_azel(lla)
        print(self.sat_azel)
        self.asi_azel = self.map_satazel_to_asiazel(self.sat_azel)
        print(self.asi_azel)
        return self.asi_azel

    def map_satazel_to_asiazel(self, azel, deg_thresh=1, debug=False):
        """
        Given the azimuth and elevation of the satellite, locate the THEMIS ASI 
        calibration pixel (value and index) that is closest to az and el. Use
        scipy.spatial.KDTree() to find the nearest values.

        deg_thresh is the starting degree threshold between az, el and the 
        calibration file. If no matches are found during the first pass,
        the deg_thresh is scaled by deg_thresh_scale_factor to expand the search
        to a wider range of calibration pixels. 

        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        n_dims = len(azel.shape)
        
        az_grid = self.cal['az'].copy().ravel()
        el_grid = self.cal['el'].copy().ravel()
        # An extreme dummy value that won't match any real AzEl value
        az_grid[np.isnan(az_grid)] = -10000 
        el_grid[np.isnan(el_grid)] = -10000

        # Set up the KDtree.
        tree = scipy.spatial.KDTree(np.array(list(zip(az_grid, el_grid))))
        dist_to_neighbor, idx_neighbor = tree.query(azel, 
                                                    distance_upper_bound=deg_thresh)

        # The idx_neighbor indicies are for the flattened array. 
        # Az coordinate in the non-flattened array is idx % self.cal['az'].shape[1]
        # El coorinate in the non-flattened array is idx // self.cal['az'].shape[1]
        if n_dims == 2:
            self.asi_azel = np.empty((len(idx_neighbor), 2), dtype=np.uint8)
            self.asi_azel[:, 0] = np.remainder(idx_neighbor, self.cal['az'].shape[1])
            self.asi_azel[:, 1] = np.floor_divide(idx_neighbor, self.cal['az'].shape[1])
        elif n_dims == 1:
            self.asi_azel = np.empty(2, dtype=np.uint8)
            self.asi_azel[0] = np.remainder(idx_neighbor, self.cal['az'].shape[1])
            self.asi_azel[1] = np.floor_divide(idx_neighbor, self.cal['az'].shape[1])

        if debug:
            for azel_i, asi_idx, dist in zip(azel, idx_neighbor, dist_to_neighbor):
                print(f'The point {azel_i} is nearest the grid location '
                      f'({az_grid[asi_idx]}, {el_grid[asi_idx]}) and is {round(dist, 1)} degrees away.')
        return self.asi_azel

    def map_lla_to_sat_azel(self, lla):
        """
        Get the satellite's azimuth and elevation given the satellite's
        lat, lon, and alt_km coordinates.

        Parameters
        ----------
        lla : 1d or 2d array of floats
            The lat, lon, and alt_km values of the satellite. If 2d, 
            the rows correspond to time.

        Returns
        -------
        sat_azel : array
            An array of shape lla.shape[0] x 2 with the satellite azimuth 
            and elevation columns.
        """
        planets = skyfield.api.load('de421.bsp')
        earth = planets['earth']
        station = earth + skyfield.api.Topos(latitude_degrees=self.cal['lat'], 
                                longitude_degrees=self.cal['lon'], 
                                elevation_m=self.cal['alt_m'])
        ts = skyfield.api.load.timescale()
        t = ts.now()

        # Check if the user passed in one set of LLA values or a 2d array. 
        # Save the number of dimensions and if is 1D, turn into a 2D array of
        # shape 1 x 3. 
        n_dims = len(lla.shape)
        if n_dims == 1:
            lla = np.array([lla])

        sat_azel = np.zeros((lla.shape[0], 2))

        for i, (lat_i, lon_i, alt_km_i) in enumerate(lla):
            sat_i = earth + skyfield.api.Topos(
                                latitude_degrees=lat_i, 
                                longitude_degrees=lon_i, 
                                elevation_m=1E3*alt_km_i
                                )
            astro = station.at(t).observe(sat_i)
            app = astro.apparent()
            el_i, az_i, _ = app.altaz()
            sat_azel[i, 1], sat_azel[i, 0] = el_i.degrees, az_i.degrees
        # Remove the extra dimension if user fed in one set of LLA values.
        if n_dims == 1:
            sat_azel = sat_azel[0, :]
        return sat_azel

    def map_lla_to_footprint(self, lla, map_alt_km):
        """
        Use IRBEM to map the satelite's lat, lon, and alt_km coodinates 
        to map_alt_km and return the footprint's mapped_lat, mapped_lon, 
        and mapped_alt_km coordinates.

        Parameters
        ----------
        lla : array
            An nTime x 3 array of latitude, longitude, altitude_km values.
        map_alt_km : float
            The footprint altitude to map to.

        Returns
        -------
        mapped_lla : array
            Same shape as lla.
        """
        mapped_lla = np.zeros_like(lla)
        
        # Check if the user passed in one set of LLA values or a 2d array. 
        # Save the number of dimensions and if is 1D, turn into a 2D array of
        # shape 1 x 3. 
        n_dims = len(lla.shape)
        if n_dims == 1:
            lla = np.array([lla])

        for i, (lat_i, lon_i, alt_km_i) in enumerate(lla):
            model = IRBEM.MagFields(kext='OPQ77')
            model.find_foot_point(
                {'x1':alt_km_i, 'x2':lat_i, 'x3':lon_i, 'dateTime':self.time[0]},
                None, stopAlt=map_alt_km, hemiFlag=0
                )
            mapped_alt_km, mapped_lat, mapped_lon = model.find_foot_point_output['XFOOT']
            mapped_lla[i, :] = [mapped_lat, mapped_lon, mapped_alt_km]
        
        # Remove the extra dimension if user fed in one set of LLA values.
        if n_dims == 1:
            mapped_lla = mapped_lla[0, :]
        return mapped_lla
   
if __name__ == '__main__':
    ### TEST SCRIPT FOR THEMIS_ASI() CLASS ###
    # site = 'WHIT'
    # time = '2015-04-16T09:09:00'
    # l = THEMIS_ASI(site, time)
    # l.load_themis_cal()

    # fig, ax = plt.subplots(figsize=(6,8))
    # l.plot_themis_asi_frame(time, ax=ax)
    # l.plot_azel_contours(ax=ax)
    # plt.tight_layout()
    # plt.show()

    ### TEST SCRIPT FOR THEMIS_ASI_map_azel() CLASS ###
    site = 'WHIT'
    time = '2015-04-16T09:09:00'
    trajectory=[np.linspace(0, 90), np.linspace(0, 45)]
    l = THEMIS_ASI_map_azel(site, time)
    l.load_themis_cal()

    # lla = np.array([
    #     [55.01, -140, 500],
    #     [61.01, -135.22, 500]
    # ])
    print(l.cal)
    lla = np.array([61.01, -135.22, 500])
    asi_azel1 = l.map_lla_to_asiazel(lla)

    lla = np.array([61.01, -135, 500])
    asi_azel2 = l.map_lla_to_asiazel(lla)

    # fig, ax = plt.subplots(figsize=(6,8))
    # l.plot_themis_asi_frame(time, ax=ax)
    # l.plot_azel_contours(ax=ax)

    # ax.scatter(*asi_azel, c='g', marker='x')
    # # ax.plot(asi_azel[:, 0], asi_azel[:, 1], c='g')
    # plt.tight_layout()
    # plt.show()