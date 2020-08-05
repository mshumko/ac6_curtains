import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
from datetime import datetime
import dateutil.parser
import scipy.spatial
import pathlib

from skyfield.api import EarthSatellite, Topos, load
import cdflib # https://github.com/MAVENSDC/cdflib

from ac6_curtains import dirs
import IRBEM

class THEMIS_ASI:
    def __init__(self, site, time):
        """
        Loads the All-Sky Imager images and calibration.
        """
        self.asi_dir = dirs.ASI_DIR
        self.site = site.lower()
        self.time = time

        if isinstance(self.time, str):
            # Convert to datetime object if passes a tring time
            self.time = dateutil.parser.parse(self.time)
        self.load_themis_asf()
        return

    def load_themis_asf(self):
        """
        Load the THEMIS ASF data and convert the time keys to datetime objects
        """
        file_glob_str = f'*{self.site}_{self.time.strftime("%Y%m%d%H")}_v*.cdf'
        # file_name = f'thg_l1_asf_{self.site}_{self.time.strftime("%Y%m%d%H")}_v01.cdf'
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

    def load_themis_cal(self):
        """
        Loads the THEMIS calibration file.
        """
        file_name = f'thg_l2_asc_{self.site}_*.cdf'
        cdf_paths = sorted(pathlib.Path(self.asi_dir).glob(file_name))
        cdf_path = cdf_paths[-1] # Grab the most recent cal file.

        cal = cdflib.cdfread.CDF(cdf_path)
        az = cal[f"thg_asf_{self.site}_azim"][0]
        el = cal[f"thg_asf_{self.site}_elev"][0]
        lat = cal[f"thg_asc_{self.site}_glat"]
        lon = (cal[f"thg_asc_{self.site}_glon"] + 180) % 360 - 180  # [0,360] -> [-180,180]
        alt_m = cal[f"thg_asc_{self.site}_alti"]
        x = y = cal[f"thg_asf_{self.site}_c256"]
        time = datetime.utcfromtimestamp(cal[f"thg_asf_{self.site}_time"][-1])

        self.cal = {
            "az":az, "el":el, 'coords':x, "lat": lat, "lon": lon, "alt_m": alt_m, 
            "site": self.site, "calfilename": cdf_path.name, "caltime": time
                }
        return

    def plot_themis_asi_frame(self, t0, ax=None, max_tdiff_m=1, imshow_vmin=None, 
                            imshow_vmax=None, imshow_norm='log', colorbar=True):
        """
        Plot a ASI frame with a time nearest to t0.
        """
        if ax is None:
            _, self.ax = plt.subplots()
        else:
            self.ax = ax

        if isinstance(t0, str):
            # Convert to datetime object if passes a tring time
            t0 = dateutil.parser.parse(t0) 
        dt = self.time-t0
        dt_sec = np.abs([dt_i.total_seconds() for dt_i in dt])
        self.idt_nearest = np.argmin(dt_sec)
        t0_nearest = self.time[self.idt_nearest]

        if np.abs((t0_nearest - t0).total_seconds()) > 60*max_tdiff_m:
            raise ValueError(f'No THEMIS ASI image found within {max_tdiff_m} minutes.')

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
                     f'{np.abs(round(self.cal["lon"]))}{lon_label})\n{t0_nearest}')
        self.hi = self.ax.imshow(self.imgs[self.idt_nearest, :, :], cmap="gray", 
                                origin="lower", interpolation="none",
                                vmin=imshow_vmin, vmax=imshow_vmax, norm=norm)
        if colorbar:
            plt.colorbar(self.hi, ax=self.ax, orientation='horizontal')
        self.ht = self.ax.set_title(title_text, color="k")
        self.ax.axis('off')
        return t0_nearest

    def plot_azel_contours(self, ax=None):
        """ 
        Superpose the azimuth and elivation contours on the ASI frame 
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
        """
        super().__init__(site, time)
        return

    def find_nearest_azel(self, az, el, deg_thresh=1, debug=False):
        """
        Given the azimuth and elevation of the satellite, locate the THEMIS ASI 
        calibration pixel (value and index) that is closest to az and el. Use
        scipy.spatial.KDTree() to find the nearest values.

        deg_thresh is the starting degree threshold between az, el and the 
        calibration file. If no matches are found during the first pass,
        the deg_thresh is scaled by deg_thresh_scale_factor to expand the search
        to a wider range of calibration pixels. 
        """
        az_grid = self.cal['az'].copy().ravel()
        el_grid = self.cal['el'].copy().ravel()
        # An extreme dummy value that won't match any real AzEl value
        az_grid[np.isnan(az_grid)] = -10000 
        el_grid[np.isnan(el_grid)] = -10000

        # Set up the KDtree.
        print(np.array(list(zip(az_grid, el_grid))))
        tree = scipy.spatial.KDTree(np.array(list(zip(az_grid, el_grid))))
        dist_to_neighbor, idx_neighbor = tree.query(np.array([az, el]).T, 
                                                    distance_upper_bound=deg_thresh)

        # The idx_neighbor indicies are for the flattened array. 
        # Az coordinate in the non-flattened array is idx % self.cal['az'].shape[1]
        # El coorinate in the non-flattened array is idx // self.cal['az'].shape[1]
        self.asi_azel = np.empty((len(idx_neighbor), 2), dtype=np.uint8)
        self.asi_azel[:, 0] = np.remainder(idx_neighbor, self.cal['az'].shape[1])
        self.asi_azel[:, 1] = np.floor_divide(idx_neighbor, self.cal['az'].shape[1])

        if debug:
            for az_i, el_i, asi_idx, dist in zip(az, el, idx_neighbor, dist_to_neighbor):
                print(f'The point ({az_i}, {el_i}) is nearest the grid location '
                      f'({az_grid[asi_idx]}, {el_grid[asi_idx]}) and is {round(dist, 1)} degrees away.')
        return self.asi_azel

    def sat_lla2sat_azel(self, lat, lon, alt_km):
        """
        Get the satellite's azimuth and elevation given the satellite's
        lat, lon, and alt_km coordinates.
        """
        planets = load('de421.bsp')
        earth = planets['earth']
        station = earth + Topos(latitude_degrees=self.cal['lat'], 
                                longitude_degrees=self.cal['lon'], 
                                elevation_m=self.cal['alt_m'])
        ts = load.timescale()
        t = ts.now()

        # If the user did not provide LLA arrays turn them into arrays.
        if not hasattr(lat, '__len__'):
            lat = [lat]
            lon = [lon]
            alt_km = [alt_km]

        sat_azel = np.zeros((len(lat), 2))

        for i, lat_i, lon_i, alt_km_i in enumerate(zip(lat, lon, alt_km)):
            sat_i = earth + Topos(
                                latitude_degrees=lat_i, 
                                longitude_degrees=lon_i, 
                                elevation_m=1E3*alt_km_i
                                )
            astro = station.at(t).observe(sat_i)
            app = astro.apparent()
            sat_azel[i, 1], sat_azel[i, 0], _ = app.altaz()
        return sat_azel

    def map_lla_to_footprint(self, lat, lon, alt_km, map_alt_km):
        """
        Use IRBEM to map the sattelites lat, lon, and alt_km coodinates 
        to map_alt_km and return the footprint's mapped_lat, mapped_lon, 
        and mapped_alt_km coordinates.
        """
        model = IRBEM.MagFields(kext='OPQ77')
        model.find_foot_point(
            {'x1':alt_km, 'x2':lat, 'x3':lon, 'dateTime':self.time[0]},
            None, stopAlt=map_alt_km, hemiFlag=0
            )
        mapped_alt_km, mapped_lat, mapped_lon = model.find_foot_point_output['XFOOT']
        return mapped_lat, mapped_lon, mapped_alt_km
   
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

    asi_azel = l.find_nearest_azel(trajectory[0], trajectory[1])

    fig, ax = plt.subplots(figsize=(6,8))
    l.plot_themis_asi_frame(time, ax=ax)
    l.plot_azel_contours(ax=ax)

    ax.scatter(asi_azel[:, 0], asi_azel[:, 1])
    plt.tight_layout()
    plt.show()