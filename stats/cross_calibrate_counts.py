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
        self.percentiles = percentiles
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
            print('No passes found.')
            return

        self._get_pass_bounds(df_A)
        self._get_pass_bounds(df_B)
        # self.passes = np.vstack((self.passes, np.zeros(11)))
        return 

    def save_pass_catalog(self, save_name):
        """ Saves the passes statistics catalog to a csv file. """
        df = pd.DataFrame(data=self.passes, columns=self.passes_columns)
        df.to_csv(save_name, index=False)
        return


    def _get_pass_bounds(self, df, thresh_min=10):
        """ 
        Find the breaks in the df dataframe time stamps to 
        identify the start/end of a radiation belt pass. A
        break is defined as time gaps larger than 
        thresh_min=10 minutes.
        """
        t = date2num(df.index)
        dt = t[1:] - t[:-1] # Change in time stamp
        # Find the breaks in the time series.
        breaks = np.where(dt > sec2day(60*thresh_min))[0]

        start_ind = np.zeros(len(breaks)+1, dtype=int)
        end_ind = np.zeros(len(breaks)+1, dtype=int)
        end_ind[-1] = df.index.shape[0]-1
        print(len(breaks))

        for i, break_i in enumerate(breaks):
            print(i, break_i)
            end_ind[i] = break_i
            start_ind[i+i] = break_i+1
        
        if self.debug:
            plt.scatter(df.index, np.zeros(df.index.shape[0]))
            plt.xlim(df.index[0]-timedelta(minutes=1), df.index[-1]+timedelta(minutes=1))

            for i, (s_i, e_i) in enumerate(zip(start_ind, end_ind)):
                plt.axvline(df.index[s_i], lw=5, c='g')
                plt.axvline(df.index[e_i], c='r')

            plt.show()

            # for i, (s_i, e_i) in enumerate(zip(start_ind, end_ind)):
            #     print(s_i, e_i)
            #     print(f'Pass {i} start_time: {df.index[s_i]}, end_time: {df.index[e_i]}')
        return start_ind, end_ind

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
    c = CrossCalibrate(debug=True)
    c.loop()