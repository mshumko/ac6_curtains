import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime, timedelta
#import dateutil.parser
#import csv
#import itertools
#import scipy.signal
#import sys
import os 
#import functools
import pandas as pd

import dirs


class CrossCalibrate:
    def __init__(self, percentiles=[25, 50, 75], debug=False):
        """
        NAME: CrossCalibrate
        USE:  Estimates the distribution of counts observed 
              by the two AC6 units during radiation belt 
              passes. A pass is defined by 4 < L < 8 
              (kwargs for the radBeltIntervals method.) Radiation belt 
              interval indicies are stored in 
              self.intervals array(nPasses x 2). Occurance 
              rate for each pass is saved in self.rates
              array. Occurance rate is calculated by adding 
              up all of the microbursts widths for each pass
              (assuming a width function for which two are
              defined and are chosen by the mode kwarg of 
              the occurance_rate method) and then divided 
              by the pass time.
        INPUT: sc_id: spacecraft id
               date:  data to load data from
               catV:  microburst catalog version
               catPath: microburst catalog path (if not
                        specied, the __init__ method 
                        will use a default path)
        AUTHOR: Mykhaylo Shumko
        RETURNS: self.intervals: array(nPasses x 2) of 
                    radiation belt pass start/end indicies
                 self.rates: array(nPasses) of the 
                    microburst duty cycle between 0 and 1 
                    for each pass.
        MOD:     2018-10-23
        """
        self.percentiles = np.array(percentiles)
        self.debug = debug
        return

    def loop(self, lbound=4, ubound=8):
        """
        Loop over each AC6 day when both spacecraft were taking 10 Hz
        data and esimated the self.percentiles for each radiation belt
        pass when they were taking data.
        """
        # Get dates to loop over
        self._get_loop_dates()
        self.passes_columns = ('start_A', 'end_A', 'start_B', 'end_B', '25p_A', 
                          '50p_A', '75p_A', '25p_B', '50p_B', '75p_B', 
                          'Lag_In_Track')
        self.passes = np.zeros((0, len(self.passes_columns)), dtype=object)

        if self.debug: i = 0 # Counter to halt exection after processing a few days

        for date in self.dates:
            # Load this day, if exists
            self._load_ten_hz_data(date)
            # If the 10 Hz data does not exist, or is empty on that day.
            if (self.ac6a_data is None) or (self.ac6b_data is None):
                continue

            if self.debug: 
                print(f"Processing data from {date}")
                i+=1
            if self.debug and i > 10:
                return
            self.get_belt_passes(lbound, ubound) 
            self.get_belt_pass_stats()
        return

    def get_belt_passes(self, lbound, ubound):
        """
        This method finds all of the radiation belt passes in a day 
        and separates it into start/end indicies.
        """
        # Get the absolute values of the L shell.
        L_A = self.ac6a_data.Lm_OPQ.abs()
        L_B = self.ac6b_data.Lm_OPQ.abs()

        df_A = self.ac6a_data[(L_A > lbound) & (L_A < ubound)]
        df_B = self.ac6b_data[(L_B > lbound) & (L_B < ubound)]

        if df_A.shape[0] == 0 or df_B.shape[0] == 0: 
            return

        start_time_A, end_time_A = self._get_pass_bounds(df_A)
        start_time_B, end_time_B = self._get_pass_bounds(df_B)
        self._match_passes(start_time_A, end_time_A, start_time_B, end_time_B)
        
        return 

    def get_belt_pass_stats(self):
        """ 
        Get the radiation belt statistics for the rows in 
        self.passes still left as NaN 
        """
        # Check which rows in the last column are NaN
        id_nan = np.where(pd.isnull(self.passes[:,-1]))[0]

        for i in id_nan:
            df_A = self.ac6a_data[
                (self.ac6a_data.index > self.passes[i, 0]) &
                (self.ac6a_data.index < self.passes[i, 1]) &
                (self.ac6a_data.flag == 0)
            ]
            df_B = self.ac6b_data[
                (self.ac6b_data.index > self.passes[i, 2]) &
                (self.ac6b_data.index < self.passes[i, 3]) &
                (self.ac6b_data.flag == 0)
            ]
            percentiles_A = df_A.dos1rate.quantile(self.percentiles/100)
            percentiles_B = df_B.dos1rate.quantile(self.percentiles/100)
            Lag_In_Track = df_A.Lag_In_Track.mean()
            print(percentiles_A.values, percentiles_B.values, Lag_In_Track)
            self.passes[i, 4:] = np.concatenate(
                (percentiles_A.values, percentiles_B.values, [Lag_In_Track])
                                                ) 
        return

    def save_pass_catalog(self, save_name):
        """ Saves the passes statistics catalog to a csv file. """
        df = pd.DataFrame(data=self.passes, columns=self.passes_columns)
        df.dropna() # Drop error values
        save_path = os.path.join(dirs.CATALOG_DIR, save_name)
        df.to_csv(save_path, index=False)
        return

    def _get_pass_bounds(self, df, thresh_sec=60):
        """ 
        Find the breaks in the df dataframe time stamps to 
        identify the start/end of a radiation belt pass. A
        break is defined as time gaps larger than 
        thresh_sec=60 seconds.
        """
        t = date2num(df.index)
        dt = t[1:] - t[:-1] # Change in time stamp
        # Find the breaks in the time series.
        breaks = np.where(dt > sec2day(thresh_sec))[0]

        start_ind = np.zeros(len(breaks)+1, dtype=int)
        end_ind = np.zeros(len(breaks)+1, dtype=int)
        end_ind[-1] = df.index.shape[0]-1

        for i, break_i in enumerate(breaks):
            end_ind[i] = break_i
            start_ind[i+1] = break_i+1
        
        ### TEST CODE ###
        if self.debug == 2: # higher level of debugging.
            plt.scatter(df.index, df.Lm_OPQ)
            plt.xlim(df.index[0]-timedelta(minutes=1), df.index[-1]+timedelta(minutes=1))

            for i, (s_i, e_i) in enumerate(zip(start_ind, end_ind)):
                plt.axvline(df.index[s_i], c='g')
                plt.axvline(df.index[e_i], c='r')
            plt.show()
        return df.index[start_ind], df.index[end_ind]

    def _match_passes(self, start_time_A, end_time_A, start_time_B, end_time_B, thresh_sec=70):
        """ 
        Use the start and end time arrays from the two AC6 units and find when they were 
        taking data for the same pass.

        thresh_sec=70 kwarg specifies the acceptable delay between the start and end times. 
        70 second is chosen since the furtherst distance AC6 were from one another and 
        taking 10 Hz data was about 65 seconds.
        """
        #shared_passes = np.zeros((0, 4), dtype=object)

        for start_A, end_A in zip(start_time_A, end_time_A):
            for start_B, end_B in zip(start_time_B, end_time_B):
                # Loop over all start and end times and check which ones are within thresh
                if ( (np.abs((start_A-start_B).total_seconds()) < thresh_sec) and
                     (np.abs((end_A-end_B).total_seconds()) < thresh_sec) ):
                    
                    append_row = np.concatenate(
                        (
                            [start_A, end_A, start_B, end_B], 
                            np.nan*np.zeros(self.passes.shape[1]-4)
                        ))
                    self.passes = np.vstack((self.passes, append_row))
        return #shared_passes


    def _load_ten_hz_data(self, day):
        """
        Load the 10 Hz AC6 data from both spacecraft on date.
        """
        time_keys = ['year', 'month', 'day', 'hour', 'minute', 'second']
        dayStr = '{0:%Y%m%d}'.format(day)
        pathA = os.path.join(dirs.AC6_DATA_PATH('a'), 
                'AC6-A_{}_L2_10Hz_V03.csv'.format(dayStr))
        pathB = os.path.join(dirs.AC6_DATA_PATH('b'), 
                'AC6-B_{}_L2_10Hz_V03.csv'.format(dayStr))
        try:
            self.ac6a_data = pd.read_csv(pathA, na_values='-1e+31')
            self.ac6b_data = pd.read_csv(pathB, na_values='-1e+31')
        except FileNotFoundError:
            self.ac6a_data, self.ac6b_data = None, None
            return

        if (self.ac6a_data.shape[0] <= 1) or (self.ac6b_data.shape[0] <= 1):
            self.ac6a_data, self.ac6b_data = None, None
            return

        self.ac6a_data['dateTime'] = pd.to_datetime(self.ac6a_data[time_keys])
        self.ac6a_data = self.ac6a_data.set_index('dateTime')
        
        self.ac6b_data['dateTime'] = pd.to_datetime(self.ac6b_data[time_keys])
        self.ac6b_data = self.ac6b_data.set_index('dateTime') 
        return

    def _get_loop_dates(self):
        """ Get dates to loop over """
        start_date = datetime(2014, 1, 1)
        end_date = datetime.now()
        self.dates = pd.date_range(start_date, end_date, freq='D')
        return


def sec2day(s):
    """ Convert seconds to fraction of a day."""
    return s/86400

if __name__ == '__main__':
    import time
    start_time = time.time()
    c = CrossCalibrate(debug=False)
    c.loop()
    c.save_pass_catalog('cross_calibrate_pass.csv')
    print(f'Run time = {round(time.time()-start_time)}')
