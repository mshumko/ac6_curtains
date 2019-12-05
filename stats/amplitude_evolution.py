# In this program we investigate the evolution of the 
# curtain flux (counts) as observed by the leading and
# trailing AC6 unit.

import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time

import dirs

class CurtainAmplitude:
    """ 
    This class loops through the curtain dataset and 
    calculates the count rates in the curtain. The 
    """
    def __init__(self, catalog_name, debug=True):
        self.catalog_name = catalog_name
        self.save_name = catalog_name #f'{catalog_name.split(".")[0]}_integrated.txt' 
        self.debug = debug
        
        self.base_dir = dirs.BASE_DIR
        self.ac6_data_dir = dirs.AC6_DATA_PATH
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
        self.integration_width_s = integration_width_s/2
        self.integration_width_td = timedelta(seconds=integration_width_s/2)
        self.baseline_subtract = baseline_subtract
        self.integrated_counts = np.nan*np.ones((self.cat.shape[0], 2), 
                                                dtype=int)

        for row_index, current_row in self.cat.iterrows():
            # Get current date
            date = current_row.dateTime.date()

            if self.debug: print(f'Index {row_index}/{self.cat.shape[0]}', 
                                current_row.dateTime, current_date)

            # If current date is not the same as last, load the 10 Hz data.
            if date != current_date: 
                self.load_ten_hz_data(date)
                current_date = date

            # baseline subtract if args provided
            if baseline_subtract:
                    self._baseline_subtract(current_row, baseline_subtract)

            # Get the integrated counts (baseline subtraction in there)
            self.integrated_counts[row_index, :] = \
                            self.get_integrated_counts(current_row, 
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

        df_a = self.ac6a_data[(self.ac6a_data.index > time_range_A[0]) & 
                              (self.ac6a_data.index < time_range_A[1]) ]
        df_b = self.ac6b_data[(self.ac6b_data.index > time_range_B[0]) & 
                              (self.ac6b_data.index < time_range_B[1]) ]
        counts_A = df_a.dos1rate.sum()
        counts_B = df_b.dos1rate.sum()
        if self.debug:
            print(row.dateTime, time_range_A, time_range_B, counts_A, counts_B)
        return counts_A, counts_B

    def _baseline_subtract(self, row, baseline_subtract):
        """ Method that handles the baseline subtraction """
        mode, width, percentile = baseline_subtract # Unpack tuple

        # Paul's pecentile method described in O'Brien et al. 2004, GRL.
        if 'percent' in mode: 
            self.obrienBaseline(row, 'A',
                                time_width_s=width, percentile=percentile)
            self.obrienBaseline(row, 'B',
                                time_width_s=width, percentile=percentile)
        else:
            raise NotImplementedError

        return 

    def obrienBaseline(self, row, sc_id, time_width_s=30.0, cadence_s=0.1, percentile=10):
        """
        NAME:    obrienBaseline(data, timeWidth=1.0, cadence=0.1, PERCENTILE_THRESH=10)
        USE:     Implements the baseline algorithm described by O'Brien et al. 2004, GRL.
                 For each data point the baseline is defined as the percentile count level
                 over all counts observed within the time_width_s window.
        RETURNS: A baseline array that represents the bottom 10th precentile of the data. 
        MODIFIED:2019-11-22
        """
        # Start and end times for the baseline
        time_width_td = timedelta(seconds=time_width_s/2)
        if sc_id.upper() == 'A':
            df = self.ac6a_data
        else:
            df = self.ac6b_data

        t_key = f'time_spatial_{sc_id.upper()}'

        #baseline = np.nan*np.ones(10*2*self.integration_width_s)
        for i in range(int(10*2*self.integration_width_s)):
            # baseline_time is the time of the current point you want 
            # to estimate the baseline for. 
            baseline_time = (row[t_key] - 
                          self.integration_width_td + 
                          timedelta(seconds=i*cadence_s))
            idt_baseline = ((df.index >= baseline_time) & 
                            (df.index < baseline_time+timedelta(seconds=0.1)))
            # Start and end times are almost identical, except that
            # they are shifted by the baseline integration time, 
            # time_width_td (time_width_timedelta).
            start_time = (baseline_time - time_width_td)
            end_time = (baseline_time + time_width_td)
            idt_range = (df.index >= start_time) & (df.index <= end_time)
            baseline = np.percentile(df.loc[idt_range, 'dos1rate'], percentile)
            df.loc[idt_baseline, 'dos1rate'] -= baseline
        return        

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

    def plot_leader_follower(self, plot_integration_width_s, baseline=False, pa_thresh=0.1):
        """ Makes a scatterplot of the leader and follower counts """
        save_catalog_path = os.path.join(self.base_dir, 
                    'data/catalogs', self.save_name)
        self.cat_integrated = pd.read_csv(save_catalog_path, na_values='-1e+31')

        if pa_thresh:
            n_0 = self.cat_integrated.shape[0]
            self.cat_integrated = self.cat_integrated[
                                        np.abs(self.cat_integrated.alpha_a - self.cat_integrated.alpha_b) < pa_thresh
                                        ]
            print(n_0, self.cat_integrated.shape[0])

        n = int(10*plot_integration_width_s)
        if baseline:
            plot_keys = [f'leader_counts_{n:02d}_baseline', 
                        f'follower_counts_{n:02d}_baseline']
        else:
            plot_keys = [f'leader_counts_{n:02d}', 
                        f'follower_counts_{n:02d}']

        # Drop 0 count values
        # self.cat_integrated.loc[:, plot_keys][
        #     (self.cat_integrated.loc[:, plot_keys[0]] == 0) |
        #     (self.cat_integrated.loc[:, plot_keys[1]] == 0)
        #     ] = np.nan
        self.cat_integrated.loc[:, plot_keys[1]][
            (self.cat_integrated.loc[:, plot_keys[1]] == 0)
            ] = np.nan
        #self.cat_integrated[plot_keys][self.cat_integrated[plot_keys[1]] == 0] = np.nan
        
        line = np.linspace(0, 1E5, 1000)
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15,6))
        s = self.ax[0].scatter(self.cat_integrated[plot_keys[0]], 
                        self.cat_integrated[plot_keys[1]], 
                        c=np.abs(self.cat_integrated.Lag_In_Track),
                        s=3, vmax=20, cmap='jet')
        self.ax[0].plot(line, line, 'k--')
        plt.colorbar(s, label='|In-track-lag| [seconds]', orientation='vertical')

        # Plot the ratio of the leader to follower counts.
        r = self.cat_integrated[plot_keys[0]]/self.cat_integrated[plot_keys[1]]
        r[np.isinf(r)] = np.nan

        median = np.nanmedian(r)
        std = np.nanstd(r)
        percentiles = [25, 50, 75]
        quartiles = np.nanpercentile(r, percentiles)

        self.ax[1].scatter(self.cat_integrated[plot_keys[0]], r, 
            c=np.abs(self.cat_integrated.Lag_In_Track), s=3, vmax=20, cmap='jet')
        self.ax[1].plot(line, np.ones_like(line), 'k--')

        self.ax[-1].hist(r, bins=np.arange(0, 3, 0.1), orientation='horizontal')
        #self.ax[-1].plot(line, np.ones_like(line), 'k--', label='ratio = 1')
        self.ax[-1].axhline(1, c='k', ls='--', label='ratio = 1')

        colors = ['r', 'g', 'c']
        for q, p, c in zip(quartiles, percentiles, colors):
            self.ax[-1].axhline(q, c=c, label=f'{p} quartile')
        #self.ax[-1].axhline(median, c='r', label=f'median={round(median, 2)}')
        #self.ax[-1].text(0.6, 0.85, f'std = {round(std, 2)}', transform=self.ax[-1].transAxes)

        # Plot settings
        self.ax[1].set_title(f'AC6 follower v. leader curtain counts\n'
                          f'integration_time = {plot_integration_width_s} s | '
                          f'baseline_subtract={baseline} | PA_thresh={pa_thresh} [deg]')
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
        self.ac6a_data = self.ac6a_data.set_index('dateTime')
        self.ac6b_data = self.ac6b_data.set_index('dateTime')
        return

    def _get_keys(self):
        """ Get the keys for the integrated counts columns """
        self.leader_key = f'leader_counts_{int(20*self.integration_width_s):02d}'
        self.follower_key = f'follower_counts_{int(20*self.integration_width_s):02d}'

        if hasattr(self, 'baseline_subtract') and self.baseline_subtract:
            self.leader_key = self.leader_key + '_baseline'
            self.follower_key = self.follower_key + '_baseline'           
        return
        
class CurtainPitchAngle(CurtainAmplitude):
    def __init__(self, catalog_name, debug=False):
        """ 
        Child class of CurtainAmplitude to look up the 
        pitch angle of the instrument when each curtain
        was observed.
        """
        super().__init__(catalog_name, debug=debug)
        return

    def loop(self):
        """
        Loops over the curtain detections and find the pitch angle
        of the instruments for each curtain
        """
        current_date = datetime.min
        self.pa = np.nan*np.ones((self.cat.shape[0], 2), 
                                                dtype=int)

        for row_index, current_row in self.cat.iterrows():
            # Get current date
            date = current_row.dateTime.date()

            if self.debug: print(f'Index {row_index}/{self.cat.shape[0]}')

            # If current date is not the same as last, load the 10 Hz data.
            if date != current_date: 
                self.load_ten_hz_data(date)
                current_date = date

            self.pa[row_index, :] = self.get_pa(current_row)
        return

    def get_pa(self, current_row):
        """ 
        Given the catalog row, find the pitch angle that both AC6 units
        were scanning at the time when the curtain was observed.
        """
        idt_A = self.ac6a_data.index[np.where((self.ac6a_data.index >= current_row.time_spatial_A) & 
                (self.ac6a_data.index < current_row.time_spatial_A + 
                    timedelta(seconds=0.1)))[0][0]]
        idt_B = self.ac6b_data.index[np.where((self.ac6b_data.index >= current_row.time_spatial_B) & 
                (self.ac6b_data.index < current_row.time_spatial_B + 
                    timedelta(seconds=0.1)))[0][0]]
        return self.ac6a_data.loc[idt_A, 'Alpha'], self.ac6b_data.loc[idt_B, 'Alpha']


if __name__ == '__main__':
    if True:
        start_time = time.time()
        a = CurtainAmplitude('AC6_curtains_sorted_v8_integrated.txt', debug=False)
        #a.loop(1, baseline_subtract=('percentile', 30, 10))
        print(f'Loop time {round(time.time()-start_time)} s')
        a.plot_leader_follower(0.5, baseline=False, pa_thresh=10)
        plt.show()
    
    if False:
        c = CurtainPitchAngle('AC6_curtains_sorted_v8_integrated.txt')
        c.plot_leader_follower(1, baseline=False, pa_thresh=10)
        plt.show()
        # c.loop()

        # plt.scatter(c.pa[:, 0], c.pa[:, 1], c='k', s=3)
        # plt.plot(np.linspace(0, 180), np.linspace(0, 180), 'k--')
        # plt.show()