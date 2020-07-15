# Make a gif of the ASI frames during curtain passes

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pathlib
import matplotlib.animation as animation
import numpy as np

from skyfield.api import EarthSatellite, Topos, load

import plot_themis_asi
from ac6_curtains import dirs


class ASI_Movie(plot_themis_asi.Load_ASI):
    def __init__(self, site, t0, pass_duration_min=1):
        self.t0 = t0#.copy() # plot_themis_asi.Load_ASI may modify this variable later so copy it.
        super().__init__(site, t0)
        self.load_themis_cal()

        # Calc time range to plot
        self.time_range = [t0-timedelta(minutes=pass_duration_min/2), 
                        t0+timedelta(minutes=pass_duration_min/2)]

        # Initialize the figure to plot the ASI and AC6 10 Hz 
        # spatial time series.
        self.fig = plt.figure(constrained_layout=True, figsize=(6,8))
        gs = self.fig.add_gridspec(4, 1)
        self.ax = self.fig.add_subplot(gs[:-1, 0])
        self.bx = self.fig.add_subplot(gs[-1, 0])

        # Filter the ASI and load the AC6 data
        self.filter_asi()
        self.load_ac6_10hz()
        return

    def filter_asi(self):
        """
        Filter the ASI time series to self.time_range.
        """
        asi_idt = np.where(
            (self.time >= self.time_range[0]) & 
            (self.time <= self.time_range[1])
            )[0]
        self.time = self.time[asi_idt]
        self.imgs = self.imgs[asi_idt, : , :]
        return

    def load_ac6_10hz(self):
        """ Load and time filter the AC6 data. """
        ac6_data_path = pathlib.Path(dirs.AC6_MERGED_DATA_DIR, 
                            f'AC6_{self.t0.strftime("%Y%m%d")}_L2_10Hz_V03_merged.csv')
        self.ac6_data = pd.read_csv(ac6_data_path, index_col=0, parse_dates=True)
        self.ac6_data['dateTime_shifted_B'] = pd.to_datetime(self.ac6_data['dateTime_shifted_B'])
        
        self.ac6_data = self.ac6_data[self.time_range[0]:self.time_range[1]]

        # Find AC6's AzEl coordinataes for the ASI
        self._get_azel_coords()
        return

    def plot_frame(self, i, azel_contours=False, imshow_vmax=None, 
                    imshow_norm='linear', individual_movie_dirs=True):
        """

        """
        # Clear the axes before plotting the current frame.
        self.ax.clear()
        self.bx.clear()

        t_i = self.time[i]

        # Plot the THEMIS ASI image and azel contours.
        self.plot_themis_asi_frame(t_i, ax=self.ax, imshow_vmin=None, 
                                    imshow_vmax=imshow_vmax, 
                                    imshow_norm=imshow_norm)
        if azel_contours: self.plot_azel_contours(ax=self.ax)

        # Plot the AC6 location
        self.ax.scatter(*self.azel_index_ac6[i, :], s=50, c='r', marker='x', 
                        label='AC6')
        self.ax.scatter(*self.azel_index_100km[i, :], s=50, c='g', marker='x', 
                        label='100 km footprint')

        # Plot the AC6 time series
        self.bx.plot(self.ac6_data.index, self.ac6_data.dos1rate, 'r', label='AC6A')
        self.bx.plot(self.ac6_data['dateTime_shifted_B'], self.ac6_data.dos1rate_B, 'b', 
                    label='AC6B')
        self.bx.axvline(t_i, c='k', ls='--')

        save_name = (f'{t_i.strftime("%Y%m%d")}_'
                    f'{t_i.strftime("%H%M%S")}_'
                    'themis_asi_frame.png')
        if individual_movie_dirs:
            save_dir = pathlib.Path(dirs.BASE_DIR, 'all_sky_imagers', 
                                    'movies', imshow_norm, 
                                    t_i.strftime("%Y%m%d"))
            # Make dir if does not exist
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / save_name
        else:
            save_path = pathlib.Path(dirs.BASE_DIR, 'all_sky_imagers', 
                                    'movies', imshow_norm, save_name)
        plt.savefig(save_path)
        print(f'Saved frame: {save_name}')
        return

    def make_animation(self, imshow_vmax=None, imshow_norm='linear', individual_movie_dirs=True):
        """ 
        Call plot_frame for all the times in the filtered 
        ASI time series. 
        """
        animation_frames = []
        for i in range(len(self.time)):
            self.plot_frame(i, imshow_vmax=imshow_vmax, imshow_norm=imshow_norm, 
                            individual_movie_dirs=individual_movie_dirs)

        return

    def _get_azel_coords(self, alt=False, down_sample=30):
        """
        Make a list of azimuth and elevation indicies for AC6 above the ASI. 
        If alt=False, the AC6 elevation will be used. Otherwise AC6's 100 km
        footprint will be calculated and the azel calculated from that.
        """
        self.azel_ac6 = np.nan*np.zeros((self.time.shape[0], 2), dtype=float)
        self.azel_index_ac6 = np.nan*np.zeros((self.time.shape[0], 2), dtype=int)
        self.azel_100km = np.nan*np.zeros((self.time.shape[0], 2), dtype=float)
        self.azel_index_100km = np.nan*np.zeros((self.time.shape[0], 2), dtype=int)

        
        for i, t_i in enumerate(self.time):
            # for each frame in self.time, find the nearest ac6_data (unit A) time 
            dt = self.ac6_data.index-t_i
            dt_sec = np.abs([dt_i.total_seconds() for dt_i in dt])
            ac6_ti_nearest = self.ac6_data.index[np.argmin(dt_sec)]
            nearest_lla = self.ac6_data.loc[ac6_ti_nearest, ['lat', 'lon', 'alt']]

            az, el = self.get_azel_from_lla(*nearest_lla, find_footpoint_alt_km=False)
            self.azel_ac6[i, :] = [az.degrees, el.degrees]
            self.azel_index_ac6[i, :] = self.find_nearest_azel(
                                                        self.azel_ac6[i, 0], 
                                                        self.azel_ac6[i, 1]
                                                        )
            az, el = self.get_azel_from_lla(*nearest_lla, find_footpoint_alt_km=100)
            self.azel_100km[i, :] = [az.degrees, el.degrees]
            self.azel_index_100km[i, :] = self.find_nearest_azel(
                                                        self.azel_100km[i, 0], 
                                                        self.azel_100km[i, 1]
                                                        )
        return

if __name__ == '__main__':
    # Load the curtain catalog
    catalog_name = 'AC6_curtains_themis_asi_15deg.csv'
    cat_path = dirs.CATALOG_DIR / catalog_name
    cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)

    # Only keep the dates in cat that are in the keep_dates array.
    keep_dates = pd.to_datetime([
        '2015-04-16', '2015-08-12', '2015-09-09', '2016-10-24',
        '2016-10-27', '2016-12-08', '2016-12-18', '2017-05-01'
    ])
    for t0, row in cat.iterrows():
        if not t0.date() in keep_dates:
            cat.drop(index=t0, inplace=True)

    # Loop over each curtain and try to open the ASI data and 
    # calibration from that date.
    for t0, row in cat.iterrows():
        # Pass time range to plot.

        for site in row['nearby_stations'].split():
            # Try to load the ASI station data from that file, if it exists.
            try:
                a = ASI_Movie(site, t0, pass_duration_min=3)
            except (FileNotFoundError, ValueError) as err:
                continue
            a.make_animation(imshow_vmax=1E4, imshow_norm='log')
            del(a)
            