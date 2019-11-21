# In this program we investigate the evolution of the 
# curtain flux (counts) as observed by the leading and
# trailing AC6 unit.

import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import mission_tools.ac6.read_ac_data as read_ac_data

class CurtainAmplitude:
    """ 
    This class loops through the curtain dataset and 
    calculates the count rates in the curtain. The 
    """
    def __init__(self, catalog_name, debug=True):
        self.catalog_name = catalog_name
        self.save_name = f'{catalog_name.split(".")[0]}_integrated.txt' 
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
        self.integration_width_s = timedelta(seconds=integration_width_s/2)
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
                current_date = date
            self.integrated_counts[row_index, :] = self.get_integrated_counts(curtain, 
                                                                        self.integration_width_s,
                                                                        baseline_subtract)                    
        self.sort_leader_follower()
        self.save_catalog()
        return

    def get_integrated_counts(self, row, integration_width_s, baseline_subtract):
        """ 
        This method sums the 10 Hz counts from AC6A and AC6B data with a time width 
        integration_width_s, centered on time_spatial_A and time_spatial_B.
        """
        if baseline_subtract:
            raise NotImplementedError
        time_range_A = [row.time_spatial_A - integration_width_s, 
                        row.time_spatial_A + integration_width_s]
        time_range_B = [row.time_spatial_B - integration_width_s, 
                        row.time_spatial_B + integration_width_s]
        df_a = self.ac6a_data[(self.ac6a_data.dateTime > time_range_A[0]) & 
                              (self.ac6a_data.dateTime < time_range_A[1]) ]
        df_b = self.ac6b_data[(self.ac6b_data.dateTime > time_range_B[0]) & 
                              (self.ac6b_data.dateTime < time_range_B[1]) ]
        counts_A = df_a.dos1rate.sum()
        counts_B = df_b.dos1rate.sum()
        if self.debug:
            print(row.dateTime, time_range_A, time_range_B, counts_A, counts_B)
        return counts_A, counts_B

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

    def plot_leader_follower(self):
        """ Makes a scatterplot of the leader and follower counts """
        raise NotImplementedError
        return

    def save_catalog(self):
        """ Saves the catalog to a new file to the self.save_name csv file. """
        save_catalog_path = os.path.join(self.base_dir, 
                    'data/catalogs', self.save_name)
        df = self.cat.copy()
        df['leader_counts'] = self.leader_follower_counts[:, 0]
        df['follower_counts'] = self.leader_follower_counts[:, 1]
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

if __name__ == '__main__':
    a = CurtainAmplitude('AC6_curtains_sorted_v8.txt', debug=False)
    a.loop(0.5)