import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
from datetime import datetime
import dateutil.parser
import pathlib

from skyfield.api import EarthSatellite, Topos, load
import cdflib # https://github.com/MAVENSDC/cdflib

from ac6_curtains import dirs

class Load_ASI:
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
        file_name = f'thg_l1_asf_{self.site}_{self.time.strftime("%Y%m%d%H")}_v01.cdf'
        cdf_path = pathlib.Path(self.asi_dir, file_name)

        self.asi = cdflib.cdfread.CDF(cdf_path)
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

    def plot_themis_asi_frame(self, t0, ax=None, max_tdiff_m=1, vmin=None, vmax=None):
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
        idt_nearest = np.argmin(dt_sec)
        t0_nearest = self.time[idt_nearest]

        if np.abs((t0_nearest - t0).total_seconds()) > 60*max_tdiff_m:
            raise ValueError(f'No THEMIS ASI image found within {max_tdiff_m} minutes.')

        if self.cal['lon'] < 0:
            lon_label='W'
        else:
            lon_label='E'

        title_text = (f'{self.site.upper()} ({round(self.cal["lat"])}N, '
                     f'{np.abs(round(self.cal["lon"]))}{lon_label})\n{t0_nearest}')
        self.hi = self.ax.imshow(self.imgs[idt_nearest, :, :], cmap="gray", 
                                origin="lower", interpolation="none",
                                vmin=vmin, vmax=vmax)
        # norm=matplotlib.colors.LogNorm()
        self.ht = self.ax.set_title(title_text, color="k")
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

    def find_nearest_azel(self, az, el):
        """
        Given the azimuth and elevation of the orbit track or satellite,
        find where in the calibration arrays that point is nearest to.
        """
        deg_thresh = 0.1
        deg_thresh_scale_factor = 2
        idx = ([], [])

        az_grid = self.cal['az'].copy()
        az_grid[np.isnan(az_grid)] = -1

        el_grid = self.cal['el'].copy()
        el_grid[np.isnan(el_grid)] = -1
        
        while len(idx[0]) == 0:
            # If no points were found, keep expanding the deg_thresh
            deg_thresh *= deg_thresh_scale_factor
            idx = np.where((np.abs(az_grid - az) < deg_thresh) & 
                            (np.abs(el_grid - el) < deg_thresh))
        # Return the first row/col that met the deg_thresh 
        # condition (other values will be close enough).
        return idx[0][0], idx[1][0] 

    def get_azel_from_lla(self, lat, lon, alt_km):
        """
        Get the ASI azimuth and elevation given the satellite
        location in lat, lon, alt_km subpoint coordinates.
        """
        planets = load('de421.bsp')
        earth = planets['earth']
        station = earth + Topos(latitude_degrees=self.cal['lat'], 
                                longitude_degrees=self.cal['lon'], 
                                elevation_m=self.cal['alt_m'])
        sat = earth + Topos(latitude_degrees=lat, longitude_degrees=lon, 
                        elevation_m=alt_km*1E3)

        ts = load.timescale()
        t = ts.now()

        astro = station.at(t).observe(sat)
        app = astro.apparent()
        alt, az, _ = app.altaz()
        return az, alt

    def keys(self):
        """
        Gets the variable names from the ASI data.
        """
        if hasattr(self, self.asi):
            return self.asi.cdf_info()['zVariables']
        else:
            raise AttributeError('The ASI CDF file is probably not loaded.')
        return

if __name__ == '__main__':
    site = 'WHIT'
    time = '2015-04-07T08'
    l = Load_ASI(site, time)
    l.load_themis_cal()

    fig, ax = plt.subplots()
    l.plot_themis_asi_frame(time, ax=ax)
    l.plot_azel_contours(ax=ax)
    plt.show()