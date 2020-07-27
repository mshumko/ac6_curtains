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
    def __init__(self, site, t0, pass_duration_min=1, footprint_alt=np.arange(100, 700, 100)):
        self.t0 = t0#.copy() # plot_themis_asi.Load_ASI may modify this variable later so copy it.
        self.footprint_alt = footprint_alt

        super().__init__(site, t0)
        self.load_themis_cal()

        # Calc time range to plot
        self.time_range = [t0-timedelta(minutes=pass_duration_min/2), 
                        t0+timedelta(minutes=pass_duration_min/2)]

        # Initialize the figure to plot the ASI and AC6 10 Hz 
        # spatial time series.
        self.fig = plt.figure(constrained_layout=True, figsize=(10,6))
        gs = self.fig.add_gridspec(2, 3)
        self.ax = self.fig.add_subplot(gs[:, 0:2])
        self.bx = [self.fig.add_subplot(gs[i, -1]) for i in range(2)]
        #self.bx = self.fig.add_subplot(gs[0, 1])

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
        self._get_azel_coords(footprint_alt=self.footprint_alt)
        return

    def plot_frame(self, i, azel_contours=False, imshow_vmax=None, 
                    imshow_norm='linear', individual_movie_dirs=True):
        """

        """
        # Clear the axes before plotting the current frame.
        self.ax.clear()
        for bx_i in self.bx:
            bx_i.clear()

        t_i = self.time[i]

        # Plot the THEMIS ASI image and azel contours.
        self.plot_themis_asi_frame(t_i, ax=self.ax, imshow_vmin=None, 
                                    imshow_vmax=imshow_vmax, 
                                    imshow_norm=imshow_norm)
        if azel_contours: self.plot_azel_contours(ax=self.ax)

        
        # Plot the footpoints for other mapping altitudes.
        for j_alt in range(self.azel_index.shape[1]-1):
            self.ax.scatter(*self.azel_index[i, j_alt, :], s=50, c='g', marker='x')

        self.ax.text(self.azel_index[i, 0, 0]+10, self.azel_index[i, 0, 1], 
                    f'{self.footprint_alt[0]} km', c='g', va='top')

        # Plot the AC6 location
        self.ax.scatter(*self.azel_index[i, -1, :], s=50, c='r', marker='x', 
                        label='AC6')
        self.ax.text(self.azel_index[i, -1, 0]+10, self.azel_index[i, -1, 1], 
                    f'AC6 alt', c='r', va='top')

        # Plot the AC6 time series
        self.bx[0].plot(self.ac6_data.index, self.ac6_data.dos1rate, 'r', label='AC6A')
        self.bx[0].plot(self.ac6_data['dateTime_shifted_B'], self.ac6_data.dos1rate_B, 'b', 
                    label='AC6B')
        self.bx[0].axvline(t_i, c='k', ls='--')

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

    def get_mean_asi_intensity(self, grid_width=10):
        """

        """
        current_img = self.imgs[self.idt_nearest, :, :]
        mean_intensity = np.nan*np.ones(self.azel_index.shape[1]) # Number of altitude points

        for alt_index in range(self.azel_index.shape[1]):
            start_x = self.azel_index[alt_index, 0]-grid_width//2
            end_x = self.azel_index[alt_index, 0]+grid_width//2
            start_y = self.azel_index[alt_index, 1]-grid_width//2
            end_y = self.azel_index[alt_index, 1]+grid_width//2
            mean_intensity[alt_index] = np.mean(self.imgs[self.idt_nearest, start_x:end_x, start_y:end_y])
        return mean_intensity
        

    def _get_azel_coords(self, ac6_alt=True, footprint_alt=np.arange(100, 700, 100), down_sample=30):
        """
        Make a list of azimuth and elevation indicies for AC6 above the ASI. 
        If alt=False, the AC6 elevation will be used. Otherwise AC6's 100 km
        footprint will be calculated and the azel calculated from that.
        """
        # If the ac6_alt is True and footprint_alt is an array, make the self.azel 
        # and self.azel_index arrays with the 2nd index of length len(footprint_alt)+1.
        if len(footprint_alt) and ac6_alt:
            self.azel = np.nan*np.zeros((self.time.shape[0], len(footprint_alt)+1, 2), dtype=float)
            self.azel_index = np.nan*np.zeros((self.time.shape[0], len(footprint_alt)+1, 2), dtype=int)
        # If the footprint_alt is an array, make the self.azel and self.azel_index
        # arrays with the 2nd index of length len(footprint_alt).
        elif len(footprint_alt):
            self.azel = np.nan*np.zeros((self.time.shape[0], len(footprint_alt), 2), dtype=float)
            self.azel_index = np.nan*np.zeros((self.time.shape[0], len(footprint_alt), 2), dtype=int)
        elif ac6_alt:
            self.azel = np.nan*np.zeros((self.time.shape[0], 1, 2), dtype=float)
            self.azel_index = np.nan*np.zeros((self.time.shape[0], 1, 2), dtype=int)
        else:
            raise ValueError('The ac6_alt needs to be true, or footprint_alt needs to be an array. Or both.')
        
        for i, t_i in enumerate(self.time):
            # for each frame in self.time, find the nearest ac6_data (unit A) time 
            dt = self.ac6_data.index-t_i
            dt_sec = np.abs([dt_i.total_seconds() for dt_i in dt])
            ac6_ti_nearest = self.ac6_data.index[np.argmin(dt_sec)]
            nearest_lla = self.ac6_data.loc[ac6_ti_nearest, ['lat', 'lon', 'alt']]

            # Find the AzEl location of AC6 and save it as the last index in the 2nd dim.
            if ac6_alt:
                az, el = self.get_azel_from_lla(*nearest_lla)
                self.azel[i, -1, :] = [az.degrees, el.degrees]
                self.azel_index[i, -1, :] = self.find_nearest_azel(
                                                            self.azel[i, -1, 0], 
                                                            self.azel[i, -1, 1]
                                                            )
            # Find the AzEl location of the footprints connected to AC6 at altitudes and 
            # save it as the index in the 2nd dim.
            if len(footprint_alt):
                for j, alt_j in enumerate(footprint_alt):
                    az, el = self.get_azel_from_lla(*nearest_lla, footpoint_alt_km=alt_j)
                    self.azel[i, j, :] = [az.degrees, el.degrees]
                    self.azel_index[i, j, :] = self.find_nearest_azel(
                                                                self.azel[i, j, 0], 
                                                                self.azel[i, j, 1]
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
            except (FileNotFoundError) as err: #ValueError
                if ('not found' in str(err)):
                    continue
            a.make_animation(imshow_vmax=1E4, imshow_norm='log')
            del(a)
            