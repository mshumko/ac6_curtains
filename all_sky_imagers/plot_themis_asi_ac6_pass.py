# Make a gif of the ASI frames during curtain passes

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pathlib
import matplotlib.animation as animation
import numpy as np
import itertools

from skyfield.api import EarthSatellite, Topos, load

import plot_themis_asi
from ac6_curtains import dirs


class Plot_ASI_AC6_Pass_Frame(plot_themis_asi.Map_THEMIS_ASI):
    def __init__(self, site, t0, pass_duration_min=1, 
                footprint_altitudes=np.arange(100, 700, 100),
                debug=True):
        self.t0 = t0#.copy() # plot_themis_asi.Load_ASI may modify this variable later so copy it.
        self.footprint_altitudes = footprint_altitudes
        self.debug = debug

        super().__init__(site, t0)
        self.load_themis_cal()

        # Calc time range to plot
        self.time_range = [t0-timedelta(minutes=pass_duration_min/2), 
                        t0+timedelta(minutes=pass_duration_min/2)]

        # Initialize the figure to plot the ASI and AC6 10 Hz 
        # spatial time series.
        self.fig = plt.figure(constrained_layout=True, figsize=(10,6))
        n_rows = 3
        gs = self.fig.add_gridspec(n_rows, 2)
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.bx = n_rows*[None]
        self.bx[0] = self.fig.add_subplot(gs[0, -1])
        self.bx[1:] = [self.fig.add_subplot(gs[i, -1], sharex=self.bx[0]) for i in range(1, n_rows)]

        # Load the ASI frames and the AC6 data
        self.filter_asi_frames(self.time_range)
        self.load_ac6_10hz()
        self.get_ac6_track()
        return

    def filter_asi_frames(self, time_range):
        """
        Filteres the ASI time and image arrays to the time_range.
        """
        asi_idx = np.where(
                        (self.time > time_range[0]) & 
                        (self.time <= time_range[1])
                        )[0]
        self.time = self.time[asi_idx]
        self.imgs = self.imgs[asi_idx, :, :]
        return

    def load_ac6_10hz(self):
        """ 
        Load and filter the AC6 data by time. 
        """
        ac6_data_path = pathlib.Path(dirs.AC6_MERGED_DATA_DIR, 
                            f'AC6_{self.t0.strftime("%Y%m%d")}_L2_10Hz_V03_merged.csv')
        self.ac6_data = pd.read_csv(ac6_data_path, index_col=0, parse_dates=True)
        self.ac6_data['dateTime_shifted_B'] = pd.to_datetime(self.ac6_data['dateTime_shifted_B'])
        self.ac6_data = self.ac6_data.loc[self.time_range[0]:self.time_range[1], :]

        # Now find the nearest AC6 data points to the ASI times.
        asi_times_df = pd.DataFrame(index=self.time)
        self.ac6_data_downsampled = pd.merge_asof(asi_times_df, self.ac6_data, 
                                                left_index=True, right_index=True, 
                                                direction='nearest', 
                                                tolerance=pd.Timedelta(seconds=0.2))

        return self.ac6_data, self.ac6_data_downsampled

    def get_ac6_track(self):
        """

        """
        lla_ac6 = self.ac6_data_downsampled.loc[:, ('lat', 'lon', 'alt')].to_numpy()
        self.asi_azel_index_dict = {'ac6':self.map_lla_to_asiazel(lla_ac6)}

        for footprint_altitude in self.footprint_altitudes:  
            lla = self.map_lla_to_footprint(lla_ac6, footprint_altitude)
            self.asi_azel_index_dict[footprint_altitude] = self.map_lla_to_asiazel(lla)
        return

    def calc_asi_intensity(self, width_px=10):
        """
        Loops over the self.asi_azel_index_dict and calculates the mean 
        ASI intensity around the AC6 location +/- width_px//2
        """
        self.mean_asi_intensity = pd.DataFrame(
            data={key:np.nan*np.zeros(self.asi_azel_index_dict[key].shape[0]) 
                for key in self.asi_azel_index_dict.keys()},
            index=self.time
            )
        if self.debug:
            self.mean_asi_box_indicies = {key:pd.DataFrame(
                data=np.nan*np.ones((self.asi_azel_index_dict['ac6'].shape[0], 4)),
                columns=['start_x', 'end_x', 'start_y', 'end_y']
                ) for key in self.asi_azel_index_dict.keys()}
        # Loop over each altitude.
        for key, val in self.asi_azel_index_dict.items():
            #self.mean_asi_intensity[key] = np.nan*np.zeros(val.shape[0])
            # Loop over each time stamp.
            for i, (az, el) in enumerate(val):
                if any(np.isnan([az, el])):
                    continue
                start_x, end_x = int(az)-width_px//2, int(az)+width_px//2
                start_y, end_y = int(el)-width_px//2, int(el)+width_px//2
                if self.debug:
                    self.mean_asi_box_indicies[key].loc[i] = [start_x, end_x, start_y, end_y]

                img, _ = self.get_asi_frames(self.time[i])

                self.mean_asi_intensity.loc[self.time[i], key] = np.median(
                    img[start_x:end_x, start_y:end_y]
                    )
        return

    def plot_frame(self, i, azel_contours=False, imshow_vmax=None, 
                    imshow_norm='linear', individual_movie_dirs=True):
        """

        """
        # Clear the axes before plotting the current frame.
        self.ax.clear()
        for bx_i in self.bx:
            bx_i.clear()

        if self.debug: self.plot_azel_contours(ax=self.ax)

        t_i = self.time[i]

        # Plot the THEMIS ASI image and azel contours.
        if imshow_vmax is None:
            imshow_vmax = 0
            imshow_vmin = 65E3
            for key, vals in self.mean_asi_intensity.items():
                imshow_vmax = np.nanmax(np.append(vals, imshow_vmax))
                imshow_vmin = np.nanmin(np.append(vals, imshow_vmin))

        self.plot_themis_asi_frame(t_i, ax=self.ax, 
                                    imshow_vmax=imshow_vmax, 
                                    imshow_vmin=imshow_vmin,
                                    imshow_norm=imshow_norm)
        if azel_contours: self.plot_azel_contours(ax=self.ax)

        ac6_colors, asi_colors = itertools.tee(itertools.cycle(['r', 'g', 'b', 'c', 'y', 'grey']))

        # Plot the footprint and AC6 locations.
        for j, (key, val) in enumerate(self.asi_azel_index_dict.items()):
            color = next(ac6_colors)
            self.ax.plot(val[:, 0], val[:, 1], 
                        c=color, lw=1, label=f'{key} km')
            self.ax.scatter(val[i, 0], val[i, 1], 
                        c=color, marker='x', s=20)
            self.ax.text(val[-1, 0], val[-1, 1], key, color=color, va='top', ha='right')

            if self.debug:
                ### DRAW THE BOX WHERE THE MEAN ASI VALUE IS CALCULATED FROM.
                self.ax.plot(self.mean_asi_box_indicies[key].loc[i, ['start_x', 'end_x']],
                            self.mean_asi_box_indicies[key].loc[i, ['start_y', 'start_y']], c=color)
                self.ax.plot(self.mean_asi_box_indicies[key].loc[i, ['start_x', 'end_x']],
                            self.mean_asi_box_indicies[key].loc[i, ['end_y', 'end_y']], c=color)
                self.ax.plot(self.mean_asi_box_indicies[key].loc[i, ['start_x', 'start_x']],
                            self.mean_asi_box_indicies[key].loc[i, ['start_y', 'end_y']], c=color)
                self.ax.plot(self.mean_asi_box_indicies[key].loc[i, ['end_x', 'end_x']],
                            self.mean_asi_box_indicies[key].loc[i, ['start_y', 'end_y']], c=color)
        if self.debug:
            # Annotate the AC6 lat/lon in the lower-left corner. 
            ac6_lat_lon = self.ac6_data_downsampled.loc[t_i, ["lat", "lon"]].to_numpy(dtype=float)
            if not any(np.isnan(ac6_lat_lon)):
                ac6_lat_lon = np.around(ac6_lat_lon, 1)
            self.ax.text(0, 0, 
                        f'AC6 at:\nlat={ac6_lat_lon[[0]]}\nlon={ac6_lat_lon[[1]]}', 
                        color='r', va='bottom', ha='left', transform=self.ax.transAxes)

        ### Plot the AC6 time series
        self.bx[0].plot(self.ac6_data.index, self.ac6_data.dos1rate, 'r', label='AC6A')
        self.bx[0].plot(self.ac6_data['dateTime_shifted_B'], self.ac6_data.dos1rate_B, 'b', 
                    label='AC6B')
        self.bx[0].axvline(t_i, c='k', ls='--')

        ### Plot the ASI time series
        for (key, val) in self.mean_asi_intensity.items():
            color = next(asi_colors)
            self.bx[1].plot(self.time, val, label=key, color=color)
        self.bx[1].axvline(t_i, c='k', ls='--')

        # Plot the ASI time series in the pcolormesh format.
        # footprint_altitudes
        intensity = np.zeros((self.mean_asi_intensity['ac6'].shape[0], 
                            len(self.footprint_altitudes)))
        j=0
        for i, (key, val) in enumerate(self.mean_asi_intensity.items()):
            if key == 'ac6': 
                continue
            intensity[:, j] = val
            j+=1
        tt, aa = np.meshgrid(self.time, self.footprint_altitudes)
        self.bx[2].pcolormesh(tt, aa, intensity.T, shading='nearest')
        self.bx[2].axvline(t_i, c='k', ls='--')

        self.save_plot(t_i, individual_movie_dirs, imshow_norm)
        self.asi_colorbar.remove()
        return

    def make_animation(self, imshow_vmax=None, imshow_norm='linear', individual_movie_dirs=True):
        """ 
        Call plot_frame for all the times in the filtered 
        ASI time series. 
        """
        for i in range(len(self.time)):
            self.plot_frame(i, imshow_vmax=imshow_vmax, imshow_norm=imshow_norm, 
                            individual_movie_dirs=individual_movie_dirs)

        return      

    def save_plot(self, time, individual_movie_dirs, imshow_norm):
        """
        Handles the plot saving logic (location, diretories etc.)
        """
        save_name = (
                f'{self.site}_{time.strftime("%Y%m%d")}_'
                f'{time.strftime("%H%M%S")}_themis_asi_frame.png'
                )
        if individual_movie_dirs:
            save_dir = pathlib.Path(dirs.BASE_DIR, 'all_sky_imagers', 
                                    'movies', imshow_norm, 
                                    time.strftime("%Y%m%d"))
            # Make dir if does not exist
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / save_name
        else:
            save_path = pathlib.Path(dirs.BASE_DIR, 'all_sky_imagers', 
                                    'movies', imshow_norm, save_name)
        plt.savefig(save_path)
        print(f'Saved frame: {save_name}')
        return


if __name__ == '__main__':
    # Load the curtain catalog
    catalog_name = 'AC6_curtains_themis_asi_15deg.csv'
    cat_path = dirs.CATALOG_DIR / catalog_name
    cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)

    # Only keep the dates in cat that are in the keep_dates array.
    # keep_dates = pd.to_datetime([
    #     '2015-04-16', '2015-08-12', '2015-09-09', '2016-10-24',
    #     '2016-10-27', '2016-12-08', '2016-12-18', '2017-05-01'
    # ])
    # # keep_dates = pd.to_datetime([
    # #     '2016-12-18',
    # # ])
    # for t0, row in cat.iterrows():
    #     if not t0.date() in keep_dates:
    #         cat.drop(index=t0, inplace=True)

    # Loop over each curtain and try to open the ASI data and 
    # calibration from that date.
    for t0, row in cat.iterrows():
        # Pass time range to plot.

        for site in row['nearby_stations'].split():
            # Try to load the ASI station data from that file, if it exists.
            try:
                a = Plot_ASI_AC6_Pass_Frame(
                    site, t0.to_pydatetime(), pass_duration_min=3,
                    footprint_altitudes=[100, 200, 300, 400, 500, 600, 700]
                    )
            except (FileNotFoundError, AssertionError) as err: #ValueError
                if (('not found' in str(err)) or 
                    ('0 THEMIS ASI paths found for search string' in str(err))):
                    continue
            a.calc_asi_intensity(width_px=10)
            a.make_animation(imshow_vmax=None, imshow_norm='log')
            # del(a)
            