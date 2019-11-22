# In this program we investigate the evolution of the 
# curtain flux (counts) as observed by the leading and
# trailing AC6 unit.

import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#import mission_tools.ac6.read_ac_data as read_ac_data

class CurtainAmplitude:
    """ 
    This class loops through the curtain dataset and 
    calculates the count rates in the curtain. The 
    """
    def __init__(self, catalog_name, debug=True):
        self.catalog_name = catalog_name
        self.save_name = catalog_name #f'{catalog_name.split(".")[0]}_integrated.txt' 
        self.debug = debug
        
        self.base_dir = '/home/mike/research/ac6_curtains/'
        self.ac6_data_dir = lambda sc_id: ('/home/mike/research/ac6/ac6{}/'
                                        'ascii/level2'.format(sc_id))
        self.load_catalog()
        return

    def load_catalog(self):
        """ 
        Loads a curtain catalog into a pd.DataFrame and 
        converts time stamps to datetime objects.
        """
        catalog_path = os.path.join(self.base_dir, 
                    'data/catalogs', self.catalog_name)
        self.cat = pd.read_csv(catalog_path)

        # If debugging, only use the first ten rows to speed 
        # up the loop and test the rest of the program.
        if self.debug: self.cat = self.cat.loc[:10, :]

        # Convert times
        for timeKey in ['dateTime', 'time_spatial_A', 'time_spatial_B']:
            self.cat[timeKey] = pd.to_datetime(self.cat[timeKey])
        return

    def loop(self, integration_width_s, baseline_subtract=False):
        """
        Loops over the curtain detections and calculates the 
        integrated curtain amplitude given the integration_width_s
        in seconds. 

        Optionally, a background subtraction can be applied before
        the integration by detrending or using the O'Brien baseline.
        """
        current_date = datetime.min
        self.integration_width_s = integration_width_s
        self.integration_width_td = timedelta(seconds=integration_width_s/2)
        self.baseline_subtract = baseline_subtract
        self.integrated_counts = np.nan*np.ones((self.cat.shape[0], 2), 
                                                dtype=int)

        for row_index, curtain in self.cat.iterrows():
            # Get current date
            date = curtain.dateTime.date()

            if self.debug: print(f'Index {row_index}/{self.cat.shape[0]}', 
                                curtain.dateTime, current_date)

            # If current date is not the same as last, load the 10 Hz data.
            if date != current_date: 
                self.load_ten_hz_data(date)
                # baseline subtract if args provided
                if baseline_subtract:
                    self._baseline_subtract(baseline_subtract)
                current_date = date

            # Get the integrated counts (baseline subtraction in there)
            self.integrated_counts[row_index, :] = \
                            self.get_integrated_counts(curtain, 
                                                        self.integration_width_td,
                                                        baseline_subtract)                    
        self.sort_leader_follower()
        self.save_catalog()
        return

    def get_integrated_counts(self, row, integration_width_td, baseline_subtract=False):
        """ 
        This method sums the 10 Hz counts from AC6A and AC6B data with a time width 
        integration_width_s, centered on time_spatial_A and time_spatial_B.

        baseline_subtract kwarg specifies which baseline subtraction to use. If
        """
        time_range_A = [row.time_spatial_A - integration_width_td, 
                        row.time_spatial_A + integration_width_td]
        time_range_B = [row.time_spatial_B - integration_width_td, 
                        row.time_spatial_B + integration_width_td]

        # df_a = self.ac6a_data.copy()
        # df_b = self.ac6b_data.copy()
        df_a = self.ac6a_cdata[(self.ac6a_data.dateTime > time_range_A[0]) & 
                              (self.ac6a_data.dateTime < time_range_A[1]) ]
        df_b = self.ac6b_data[(self.ac6b_data.dateTime > time_range_B[0]) & 
                              (self.ac6b_data.dateTime < time_range_B[1]) ]
        counts_A = df_a.dos1rate.sum()
        counts_B = df_b.dos1rate.sum()
        if self.debug:
            print(row.dateTime, time_range_A, time_range_B, counts_A, counts_B)
        return counts_A, counts_B

    def _baseline_subtract(self, baseline_subtract):
        """ Method that handles the baseline subtraction """
        mode, width, percentile = baseline_subtract # Unpack tuple

        # Paul's pecentile method described in O'Brien et al. 2004, GRL.
        if 'percent' in mode: 
            baseline_A = self.obrienBaseline(self.ac6a_data.dos1rate, timeWidth=width, PERCENTILE_THRESH=percentile)
            baseline_B = self.obrienBaseline(self.ac6b_data.dos1rate, timeWidth=width, PERCENTILE_THRESH=percentile)

            # df_A = self.ac6a_data.copy()
            # df_B = self.ac6b_data.copy()

            self.ac6a_data.dos1rate = self.ac6a_data.dos1rate-baseline_A 
            self.ac6b_data.dos1rate = self.ac6b_data.dos1rate-baseline_B 
            # baseline_width = self.integration_width_td + timedelta(seconds=width/2)
            # time_range_A = [row.time_spatial_A - baseline_width,
            #                 row.time_spatial_A + baseline_width]
            # time_range_B = [row.time_spatial_B - baseline_width,
            #                 row.time_spatial_B + baseline_width]
            # baseline_A = np.nan*np.ones(10*self.integration_width_s)
            # baseline_B = np.nan*np.ones(10*self.integration_width_s)
            # curtain_A = np.nan*np.ones(10*self.integration_width_s)
            # curtain_B = np.nan*np.ones(10*self.integration_width_s)

            # df_A = self.ac6a_data.set_index('dateTime')
            # df_B = self.ac6b_data.set_index('dateTime')

            # for i in range(10*self.integration_width_s):
            #     start_time_A = time_range_A[0] + timedelta(seconds=i/10)
            #     end_time_A = time_range_A[1] + timedelta(seconds=i/10)
            #     baseline_A[i] = np.percentile(df_A.loc[start_time_A:end_time_A, 'dos1rate'], percentile)
            #     curtain_A[i] = df_A.at[row.time_spatial_A-self.integration_width_td+timedelta(seconds=i/10), 'dos1rate'] - baseline_A[i]
            #     curtain_B[i] = df_B.at[row.time_spatial_B-self.integration_width_td+timedelta(seconds=i/10), 'dos1rate'] - baseline_B[i]

            #     start_time_B = time_range_B[0] + timedelta(seconds=i/10)
            #     end_time_B = time_range_B[1] + timedelta(seconds=i/10)
            #     baseline_B[i] = np.percentile(df_B.loc[start_time_B:end_time_B, 'dos1rate'], percentile)
        else:
            raise NotImplementedError

        return 

    def obrienBaseline(self, data, timeWidth=1.0, cadence=0.1, PERCENTILE_THRESH=10):
        """
        NAME:    obrienBaseline(data, timeWidth = 1.0, cadence = 18.75E-3, PERCENTILE_THRESH = 10)
        USE:     Implements the baseline algorithm described by O'Brien et al. 2004, GRL. 
                 The timeWidth parameter is the half of the range from which to calcualte the 
                 baseline from, units are seconds. Along the same lines, cadence is used
                 to determine the number of data points which will make the timeWidth
        RETURNS: A baseline array that represents the bottom 10th precentile of the data. 
        MOD:     2016-08-02
        """
        dataWidth = int(np.floor(timeWidth/cadence))

        assert 2*dataWidth < len(data), 'Data too short for timeWidth specified.'
        baseline= np.zeros_like(data, dtype=float)
        
        for i in range(int(len(baseline)-dataWidth)):
            dataSlice = data[i:2*dataWidth+i]
            baseline[dataWidth+i] = np.percentile(dataSlice, PERCENTILE_THRESH)        
        return baseline

    def sort_leader_follower(self):
        """ 
        Sorts the self.integrated_counts array (first column is counts from 
        AC6A and second is from AC6B). 
        """
        self.leader_follower_counts = self.integrated_counts.copy()
        for i, (counts, lag) in enumerate(zip(self.integrated_counts, self.cat.Lag_In_Track)):
            # If AC6B was ahead, swap the counts
            if lag < 0:
                self.leader_follower_counts[i, :] = self.leader_follower_counts[i, ::-1]
        return

    def plot_leader_follower(self, plot_integration_width_s, baseline=False):
        """ Makes a scatterplot of the leader and follower counts """
        save_catalog_path = os.path.join(self.base_dir, 
                    'data/catalogs', self.save_name)
        self.cat_integrated = pd.read_csv(save_catalog_path, na_values='-1e+31')

        n = int(10*plot_integration_width_s)
        if baseline:
            plot_keys = [f'leader_counts_{n:02d}_baseline', 
                        f'follower_counts_{n:02d}_baseline']
        else:
            plot_keys = [f'leader_counts_{n:02d}', 
                        f'follower_counts_{n:02d}']
        line = np.linspace(0, 1E5, 1000)
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15,6))
        s = self.ax[0].scatter(self.cat_integrated[plot_keys[0]], 
                        self.cat_integrated[plot_keys[1]], 
                        c=np.abs(self.cat_integrated.Lag_In_Track),
                        s=3, vmax=20, cmap='jet')
        self.ax[0].plot(line, line, 'k--')
        plt.colorbar(s, label='|In-track-lag| [seconds]', orientation='vertical')

        # Plot the ratio of the leader to follower counts.
        self.ax[1].scatter(self.cat_integrated[plot_keys[0]], 
            self.cat_integrated[plot_keys[0]]/self.cat_integrated[plot_keys[1]], 
            c=np.abs(self.cat_integrated.Lag_In_Track), s=3, vmax=20, cmap='jet')
        self.ax[1].plot(line, np.ones_like(line), 'k--')

        self.ax[-1].hist(self.cat_integrated[plot_keys[0]]/self.cat_integrated[plot_keys[1]], 
                        bins=np.arange(0, 3, 0.1), orientation='horizontal')
        self.ax[-1].plot(line, np.ones_like(line), 'k--', label='ratio = 1')
        median = np.median(self.cat_integrated[plot_keys[0]]/self.cat_integrated[plot_keys[1]])
        std = np.std(self.cat_integrated[plot_keys[0]]/self.cat_integrated[plot_keys[1]])
        self.ax[-1].plot(line, median*np.ones_like(line), 'r', label=f'median={round(median, 2)}')
        self.ax[-1].text(0.6, 0.85, f'std = {round(std, 2)}', transform=self.ax[-1].transAxes)

        # Plot settings
        self.ax[1].set_title(f'AC6 follower v. leader curtain counts\n'
                          f'integration_time = {plot_integration_width_s} s')
        self.ax[0].set_xlabel('Leader [counts]')
        self.ax[0].set_ylabel('Follower [counts]')
        self.ax[0].set(xlim=(0, 3E4), ylim=(0, 3E4))

        self.ax[1].set(xlim=(0, 3E4), ylim=(0, 3), ylabel='Leader/Follower', xlabel='Leader')
        self.ax[-1].set(ylim=(0, 3), xlim=(0, 200), xlabel='number of curtains', ylabel='Leader/Follower')

        self.ax[-1].legend()

        # Calculate and print the number of events where the leader or follower had higher counts.
        f = np.greater(self.cat_integrated[plot_keys[0]], 
                       self.cat_integrated[plot_keys[1]])
        print(f'Number of curtains with more leader counts {sum(f)}. '
              f'Number of curtains with less leader counts {sum(~f)}')
        return

    def save_catalog(self):
        """ Saves the catalog to a new file to the self.save_name csv file. """
        save_catalog_path = os.path.join(self.base_dir, 
                    'data/catalogs', self.save_name)
        df = self.cat.copy()
        self._get_keys()
        # Set the new keys with the 10 x integration time in the key string.
        df[self.leader_key] = self.leader_follower_counts[:, 0]
        df[self.follower_key] = self.leader_follower_counts[:, 1]
        df.to_csv(save_catalog_path, index=False)
        return

    def load_ten_hz_data(self, day):
        """
        Load the 10 Hz AC6 data from both spacecraft on date.
        """
        time_keys = ['year', 'month', 'day', 'hour', 'minute', 'second']
        dayStr = '{0:%Y%m%d}'.format(day)
        pathA = os.path.join(self.ac6_data_dir('a'), 
                'AC6-A_{}_L2_10Hz_V03.csv'.format(dayStr))
        pathB = os.path.join(self.ac6_data_dir('b'), 
                'AC6-B_{}_L2_10Hz_V03.csv'.format(dayStr))
        self.ac6a_data = pd.read_csv(pathA, na_values='-1e+31')
        self.ac6a_data['dateTime'] = pd.to_datetime(self.ac6a_data[time_keys])
        self.ac6b_data = pd.read_csv(pathB, na_values='-1e+31')
        self.ac6b_data['dateTime'] = pd.to_datetime(self.ac6b_data[time_keys])
        return

    def _get_keys(self):
        """ Get the keys for the integrated counts columns """
        if hasattr(self, 'baseline_subtract') and self.baseline_subtract:
            self.leader_key = self.leader_key + '_baseline'
            self.follower_key = self.follower_key + '_baseline'
        else:
            self.leader_key = f'leader_counts_{int(10*self.integration_width_s):02d}'
            self.follower_key = f'follower_counts_{int(10*self.integration_width_s):02d}'
        return

if __name__ == '__main__':
    a = CurtainAmplitude('AC6_curtains_sorted_v8_integrated.txt', debug=True)
    a.loop(0.5, baseline_subtract=('percentile', 10, 10))
    a.plot_leader_follower(0.5)
    plt.show()