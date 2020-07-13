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
                            f'AC6_{self.time[0].strftime("%Y%m%d")}_L2_10Hz_V03_merged.csv')
        self.ac6_data = pd.read_csv(ac6_data_path, index_col=0, parse_dates=True)
        self.ac6_data['dateTime_shifted_B'] = pd.to_datetime(self.ac6_data['dateTime_shifted_B'])
        
        self.ac6_data = self.ac6_data[self.time_range[0]:self.time_range[1]]

        # Find AC6's AzEl coordinataes for the ASI
        self._get_azel_coords()
        return

    def plot_frame(self, t_i, azel_contours=True):
        """

        """
        # Clear the axes before plotting the current frame.
        self.ax.clear()
        self.bx.clear()

        # Plot the THEMIS ASI image and azel contours.
        self.plot_themis_asi_frame(t_i.to_pydatetime(), ax=self.ax)
        if azel_contours: self.plot_azel_contours(ax=ax)

        

        return

    def make_animation(self):
        """ 
        Call plot_frame for all the times in the filtered 
        ASI time series. 
        """
        line_ani = animation.FuncAnimation(self.fig, self.plot_frame, 
                                        self.time, blit=True)
        return

    def _get_azel_coords(self, alt=False, down_sample=30):
        """
        Make a list of azimuth and elevation indicies for AC6 above the ASI. 
        If alt=False, the AC6 elevation will be used. Otherwise AC6's 100 km
        footprint will be calculated and the azel calculated from that.
        """
        self.azel = np.nan*np.zeros((self.time.shape[0], 2), dtype=float)
        self.azel_index = np.nan*np.zeros((self.time.shape[0], 2), dtype=int)

        
        for i, t_i in enumerate(self.time):
            # for each frame in self.time, find the nearest ac6_data (unit A) time 
            dt = self.ac6_data.index-t_i
            dt_sec = np.abs([dt_i.total_seconds() for dt_i in dt])
            ac6_ti_nearest = self.ac6_data.index[np.argmin(dt_sec)]
            nearest_lla = self.ac6_data.loc[ac6_ti_nearest, ['lat', 'lon', 'alt']]

            az, el = self.get_azel_from_lla(*nearest_lla)
            self.azel[i, :] = [az.degrees, el.degrees]
            self.azel_index[i, :] = self.find_nearest_azel(self.azel[i, 0], 
                                                        self.azel[i, 1])
        return

if __name__ == '__main__':
    # Load the curtain catalog
    catalog_name = 'AC6_curtains_themis_asi_15deg.csv'
    cat_path = dirs.CATALOG_DIR / catalog_name
    cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)

    # Loop over each curtain and try to open the ASI data and 
    # calibration from that date.
    for t0, row in cat.iterrows():
        # Pass time range to plot.

        for site in row['nearby_stations'].split():
            # Try to load the ASI station data from that file, if it exists.
            # try:
            #     l = plot_themis_asi.Load_ASI(site, t0)
            #     l.load_themis_cal()
            # except (FileNotFoundError, ValueError) as err:
            #     continue
            try:
                a = ASI_Movie(site, t0)
            except (FileNotFoundError, ValueError) as err:
                continue
            break # Remove when the animation code is done!



        # Plot the THEMIS ASI image and azel contours.
        # l.plot_themis_asi_frame(t0.to_pydatetime(), ax=ax)
        # l.plot_azel_contours(ax=ax)

        # # Plot the AC6 data in the bottom plot
        # bx.plot(ac6_data.index, ac6_data.dos1rate, 'r', label='AC6A')
        # bx.plot(ac6_data['dateTime_shifted_B'], ac6_data.dos1rate_B, 'b', 
        #         label='AC6B')
        # bx.axvline(t0, c='k', ls='--')
        # ac6_location_str = (f'Curtain:'
        #                     f'\ntime={t0.strftime("%H:%M:%S")}'
        #                     f'\nlat={round(row.lat, 1)}'
        #                     f'\nlon={round(row.lon, 1)}'
        #                     f'\nalt={round(row.alt, 1)} [km]')
        # if row.Lag_In_Track > 0:
        #     bx.text(0, 1, f'AC6A ahead by {round(row.Lag_In_Track, 1)} seconds\n' + ac6_location_str, 
        #             va='top', transform=bx.transAxes)
        # if row.Lag_In_Track < 0:
        #     bx.text(0, 1, f'AC6B ahead by {abs(round(row.Lag_In_Track, 1))} seconds\n' + ac6_location_str, 
        #             va='top', transform=bx.transAxes)       

        # # Plot the AC6 orbit track in the ASI
        # for ac6_data_time, ac6_data_row in ac6_data.iloc[::50].iterrows():
        #     az, el = l.get_azel_from_lla(*ac6_data_row[['lat', 'lon', 'alt']])
        #     idx = l.find_nearest_azel(az.degrees, el.degrees)
        #     ax.scatter(idx[1], idx[0], c='r', s=2)
        # # Plot the first point in the AC6 orbit track
        # az, el = l.get_azel_from_lla(*ac6_data.iloc[0][['lat', 'lon', 'alt']])
        # idx = l.find_nearest_azel(az.degrees, el.degrees)
        # ax.scatter(idx[1], idx[0], s=50, c='r', marker='x')

        # # Plot a star where AC6 observed the curtain in AzEl.
        # az, el = l.get_azel_from_lla(*row[['lat', 'lon', 'alt']])
        # idx = l.find_nearest_azel(az.degrees, el.degrees)
        # ax.scatter(idx[1], idx[0], s=50, c='r', marker='*')
        
        # bx.xaxis.set_major_locator(mdates.SecondLocator(interval=30))

        # plt.savefig(
        #     pathlib.Path(dirs.BASE_DIR, 'all_sky_imagers', 'plots', 
        #                 f'{t0.strftime("%Y%m%dT%H%M%S")}_themis_asi_frame.png'), 
        #             dpi=200)
        # ax.clear()
        # bx.clear()
