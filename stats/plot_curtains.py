# This program plots the catalog detections that are well-correlated. 
import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates

import cartopy
import cartopy.crs as ccrs

# Paths that are used everywhere in the class
from ac6_curtains import dirs

class PlotCurtains:
    def __init__(self, catalog_version, plot_width=5, catalog_name=None,
                plot_save_dir=None, plot_width_flag=False, 
                make_plt_dir_flag=True, load_sorted_catalog=False):
        """
        This class plots the detections from the coincident
        curtain catalog with a set of default filters.
        The user can supply other filter values in the 
        filterDict.
        """
        self.plot_width = timedelta(seconds=plot_width)
        self.catalog_name = catalog_name
        if plot_save_dir is None:
            self.plot_save_dir = dirs.PLOT_SAVE_DIR
        else:
            self.plot_save_dir = plot_save_dir

        if make_plt_dir_flag and (not os.path.exists(self.plot_save_dir)):
            os.mkdir(self.plot_save_dir)
            print('Made directory:', self.plot_save_dir)
        self.plot_width_flag = plot_width_flag
        self.load_sorted_catalog = load_sorted_catalog
        self.load_catalog(catalog_version)
        return

    def filter_catalog(self, defaultFilter=True, filterDict={}):
        """
        Apply filters to the catalog file. The default filter
        applies a filter that should always be used. The dafault
        filter filters out the events with a temporal_CC < 0.8,
        detections made outside of 4 < L < 8, filters out detections
        that have a spatial_CC-0.3 > temporal_CC. Lastly, the default
        filters removes detections made above the US and SAA to avoid
        noise from ground transmitters and saturation in the SAA.

        Whenever you want to filter other variables, supply them to
        the filterDict dictionary. filterDict's keys are the
        catalog keys and the values is a 2 element list with upper 
        and lower bounds. If you want a one-sided filter e.g. 
        Dist_Total greater than 50 km, pass in 
        filterDict={'Dist_Total':[50, 9999]}.
        """
        if defaultFilter:
            # High CC filter
            self.catalog = self.catalog[self.catalog['space_cc'] >= 0.8]

        for key, vals in filterDict.items():

            # If supplied bounds
            if hasattr(vals, '__len__'):
                self.catalog = self.catalog[
                    ((self.catalog[key] > min(vals)) & 
                    (self.catalog[key] < max(vals)))
                                ]
            # If one value - check for equality
            else:
                self.catalog = self.catalog[self.catalog[key] ==vals]
        print(f'Plotting {self.catalog.shape[0]} curtains')
        return

    def load_catalog(self, catalog_version):
        """
        Load the csv microburst catalog. Any key that has the word time in
        it is converted to am array of datetime objects. Time keys are:
        dateTime, time_spatial_A, and time_spatial_B.
        """
        # Load a catalog that has been sorted by a person or not.
        if self.catalog_name is None:
            if self.load_sorted_catalog: 
                catalog_name = f'AC6_coincident_microbursts_sorted_v{catalog_version}.txt'
            else:
                catalog_name = f'AC6_coincident_microbursts_v{catalog_version}.txt'
        else:
            catalog_name = self.catalog_name

        catalog_path = os.path.join(dirs.CATALOG_DIR, catalog_name)
        self.catalog = pd.read_csv(catalog_path)
        # Convert the catalog times to datetime objects
        for timeKey in ['dateTime', 'time_spatial_A', 'time_spatial_B']:
            self.catalog[timeKey] = pd.to_datetime(self.catalog[timeKey])
        return

    def loop(self, **kwargs):
        """
        Loops over the detections that made it through the filters and 
        make space-time plots.
        """
        current_date = date.min

        self.fig = plt.figure(figsize=(6, 8))
        self.gs = self.fig.add_gridspec(2, 1)
        self.ax = [self.fig.add_subplot(self.gs[0, 0]), self.fig.add_subplot(self.gs[1, 0])]
        #self.bx = self.fig.add_subplot(self.gs[2, 0], projection=ccrs.PlateCarree())

        # self.make_map(self.bx)

        for _, row in self.catalog.iterrows():
            if row.dateTime.date() != current_date:
                # Load current day AC-6 data if not loaded already
                self.load_ten_hz_data(row.dateTime.date())
                current_date = row.dateTime.date()
            self.make_plot(row, **kwargs)
            self.ax[0].clear()
            self.ax[1].clear()
            # self.bx.clear()
        plt.close()
        return

    def make_map(self, ax):
        """
        Make a map of the BLC region.
        """
        self.bx.set_extent([-60, 30, 40, 80], crs=ccrs.PlateCarree())
        self.bx.coastlines(resolution='50m', color='black', linewidth=1)
        self.bx.add_feature(cartopy.feature.LAND, zorder=0, 
                            edgecolor='black', facecolor='grey')

        # Overlay L shell contours
        L_lons = np.load('/home/mike/research/mission_tools'
                        '/misc/irbem_l_lons.npy')
        L_lats = np.load('/home/mike/research/mission_tools'
                        '/misc/irbem_l_lats.npy')
        L = np.load('/home/mike/research/mission_tools'
                        '/misc/irbem_l_l.npy')
        levels = [4,8]
        ax.contour(L_lons, L_lats, L, levels=levels, colors='k', linestyles='dotted')

        # Overlay BLC boundary
        mirror_point_df = pd.read_csv(os.path.join(dirs.BASE_DIR, 'data', 'lat_lon_mirror_alt.csv'),
                                    index_col=0, header=0)
        lons = np.array(mirror_point_df.columns, dtype=float)
        lats = mirror_point_df.index.values.astype(float)
        ax.contour(lons, lats, 
            mirror_point_df.values, transform=ccrs.PlateCarree(), 
            levels=[0, 100], colors=['b', 'b'], linestyles=['dashed', 'solid'])
        return

    def load_ten_hz_data(self, day):
        """
        Load the 10 Hz AC6 data from both spacecraft on date.
        """
        time_keys = ['year', 'month', 'day', 'hour', 'minute', 'second']
        dayStr = '{0:%Y%m%d}'.format(day)
        pathA = os.path.join(dirs.AC6_DATA_PATH('a'), 
                'AC6-A_{}_L2_10Hz_V03.csv'.format(dayStr))
        pathB = os.path.join(dirs.AC6_DATA_PATH('b'), 
                'AC6-B_{}_L2_10Hz_V03.csv'.format(dayStr))
        self.ac6a_data = pd.read_csv(pathA, na_values='-1e+31')
        self.ac6a_data['dateTime'] = pd.to_datetime(self.ac6a_data[time_keys])
        self.ac6b_data = pd.read_csv(pathB, na_values='-1e+31')
        self.ac6b_data['dateTime'] = pd.to_datetime(self.ac6b_data[time_keys])
        return

    def make_plot(self, row, **kwargs):
        """
        This method takes in a dataframe row from the catalog and makes a 
        space/time plot.
        """
        mean_subtracted = kwargs.get('mean_subtracted', False)
        savefig = kwargs.get('savefig', True)
        log_scale = kwargs.get('log_scale', False)
        plot_dos2_and_dos3 = kwargs.get('plot_dos2_and_dos3', False)
        plot_legend = kwargs.get('plot_legend', True)
        ax = kwargs.get('ax', self.ax)
        time_guide_flag = kwargs.get('time_guide_flag', False)

        df_time_a, df_time_b, df_space_a, df_space_b = self._get_filtered_plot_data(row)
        if mean_subtracted:
            df_time_a.loc[:, 'dos1rate'] -= df_time_a.loc[:, 'dos1rate'].mean()
            df_time_b.loc[:, 'dos1rate'] -= df_time_b.loc[:, 'dos1rate'].mean()
            df_space_a.loc[:, 'dos1rate'] -= df_space_a.loc[:, 'dos1rate'].mean()
            df_space_b.loc[:, 'dos1rate'] -= df_space_b.loc[:, 'dos1rate'].mean()

            if plot_dos2_and_dos3:
                df_time_a.loc[:, 'dos2rate'] -= df_time_a.loc[:, 'dos2rate'].mean()
                df_time_b.loc[:, 'dos2rate'] -= df_time_b.loc[:, 'dos2rate'].mean()
                #df_space_a.loc[:, 'dos2rate'] -= df_space_a.loc[:, 'dos2rate'].mean()
                df_time_a.loc[:, 'dos3rate'] -= df_time_a.loc[:, 'dos3rate'].mean()
                #df_space_a.loc[:, 'dos3rate'] -= df_space_a.loc[:, 'dos3rate'].mean()
                
        ax[0].plot(df_time_a['dateTime'], df_time_a['dos1rate'], 'r', label='AC6-A')
        if plot_dos2_and_dos3:
            ax[0].plot(df_time_a['dateTime'], df_time_a['dos2rate'], 'r:', label='AC6-A dos2')
            ax[0].plot(df_time_a['dateTime'], df_time_a['dos3rate'], 'r--', label='AC6-A dos3')
            
        ax[0].plot(df_time_b['dateTime'], df_time_b['dos1rate'], 'b', label='AC6-B')
        if plot_dos2_and_dos3:
            ax[0].plot(df_time_b['dateTime'], df_time_b['dos2rate'], 'b:', label='AC6-B dos2')
        if time_guide_flag:
            ax[0].axvline(row.at['dateTime'])
        if plot_legend:
            ax[0].legend(loc=1, fontsize=7
            )
        ax[1].plot(df_space_a['dateTime'], df_space_a['dos1rate'], 'r', label='AC6-A')
        ax[1].plot(df_space_b['dateTime'], df_space_b['dos1rate'], 'b', label='AC6-B')
        # ax[1].axvline(row.dateTime, c='k')

        #self.star = self.bx.scatter(row.lon, row.lat, marker='*', c='r', s=150)

        # self.bx.set_extent([row.lon-10, row.lon+10, row.lat-10, row.lat+10], crs=ccrs.PlateCarree())

        if log_scale:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        ax[0].set_ylabel('dos1rate [counts/s]')
        ax[1].set_ylabel('dos1rate [counts/s]')
        ax[1].set_xlabel('UTC')
        ax[0].set_title(f'AC6 Curtain Validation | {row.dateTime.date()}')
        ax[0].axes.get_xaxis().set_ticks([])

        # Print peak width if it exists in the catalog.
        if set(['peak_width_A', 'peak_width_B']).issubset(row.index) and self.plot_width_flag:
            s = (f'peak_width_A = {round(row["peak_width_A"], 2)} s\n'
                f'peak_width_B = {round(row["peak_width_B"], 2)} s\n'
                f'AE = {row["AE"]}, L = {round(row["Lm_OPQ"], 1)}, L = {round(row["MLT_OPQ"], 1)}')
            ax[0].text(0.02, 1, s, transform=ax[0].transAxes, va='top')

        if row.Lag_In_Track > 0:
            leading_unit = 'A'
        else:
            leading_unit = 'B'
            
        annotate_str = (f'L = {round(row.Lm_OPQ, 1)}\n'
                        f'MLT = {round(row.MLT_OPQ, 1)}\n'
                        f'lat = {round(row.lat, 1)} [deg]\n'
                        f'lon = {round(row.lon, 1)} [deg]\n'
                        f'alt = {round(row.alt, 1)} [km]\n'
                        f'dt = {round(row.Lag_In_Track, 1)} [s] (AC6-{leading_unit} ahead)\n'
                        f'AE = {row.AE} [nT]'
                        )
        ax[1].text(0.02, 0.98, annotate_str, 
                        transform=ax[1].transAxes, va='top')
        # # Print the time shift seconds
        # if row.Lag_In_Track > 0:
        #     ax[1].text(0.02, 0.98, f'AC6A ahead by = {round(row.Lag_In_Track, 1)} [s]', 
        #                 transform=ax[1].transAxes, va='top')
        # else:
        #     ax[1].text(0.02, 0.98, f'AC6B ahead by = {abs(round(row.Lag_In_Track, 1))} [s]', 
        #                 transform=ax[1].transAxes, va='top')

        # pos_str = (f'lat={round(row.lat)}, lon={round(row.lon)}, alt={round(row.alt)} [km]\n'
        #             f'Loss_Cone_Type={row.Loss_Cone_Type}\n'
        #             f'AE={row.AE} [nT]')
        # ax[1].text(0.02, 0.9, pos_str, transform=ax[1].transAxes, va='top')

        # ax[1].xaxis.set_major_locator(matplotlib.dates.SecondLocator(interval=3))

        ax[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%S'))
        ax[1].xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=1))
        formatted_start_time = datetime.strftime(row.dateTime-self.plot_width/2, "%Y/%m/%d %H:%M:00")
        xlabel = (f'AC6A seconds after\n{formatted_start_time}')
        ax[1].set_xlabel(xlabel)

        self.gs.tight_layout(self.fig)

        if savefig:
            save_name = '{0:%Y%m%d_%H%M%S}_ac6_curtain_validation.png'.format(
                        row['dateTime'])
            plt.savefig(os.path.join(self.plot_save_dir, save_name))
        
        # self.star.remove()
        return

    def _get_filtered_plot_data(self, row):
        df_time_a = self.ac6a_data[
                            (self.ac6a_data['dateTime'] > row['dateTime']-self.plot_width/2) & 
                            (self.ac6a_data['dateTime'] < row['dateTime']+self.plot_width/2)
                            ]
        df_time_b = self.ac6b_data[
                            (self.ac6b_data['dateTime'] > row['dateTime']-self.plot_width/2) & 
                            (self.ac6b_data['dateTime'] < row['dateTime']+self.plot_width/2)
                            ]
        if row.at['dateTime'] == row.at['time_spatial_A']:
            df_space_a = df_time_a
            df_space_b = self.ac6b_data[
                            (self.ac6b_data['dateTime'] > row['time_spatial_B']-self.plot_width/2) & 
                            (self.ac6b_data['dateTime'] < row['time_spatial_B']+self.plot_width/2)
                            ]
            df_space_b.loc[:, 'dateTime'] -= timedelta(seconds=row.at['Lag_In_Track'])
        elif row.at['dateTime'] == row.at['time_spatial_B']:
            df_space_a = self.ac6a_data[
                            (self.ac6a_data['dateTime'] > row['time_spatial_A']-self.plot_width/2) & 
                            (self.ac6a_data['dateTime'] < row['time_spatial_A']+self.plot_width/2)
                            ]
            df_space_a.loc[:, 'dateTime'] += timedelta(seconds=row.at['Lag_In_Track'])
            df_space_b = df_time_b
        else:
            raise(ValueError('No space matches found!'))
        return df_time_a, df_time_b, df_space_a, df_space_b

if __name__ == '__main__':
    version = 0
    plot_width = 20
    #catalog_name = f'AC6_curtains_themis_asi_5deg.csv'
    catalog_name = f'AC6_curtains_baseline_method_sorted_v0.csv'
    p = PlotCurtains(version, catalog_name=catalog_name, plot_width=plot_width)
    # p.filter_catalog(filterDict={'Loss_Cone_Type':2}, defaultFilter=False)
    #p.filter_catalog(filterDict={'Lag_In_Track':[30, 100]}, defaultFilter=False)
    p.loop(mean_subtracted=False, savefig=True)