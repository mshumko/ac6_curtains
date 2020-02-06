import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import typing

import mission_tools.ac6.read_ac_data as read_ac_data


class SpatialAlign:
    def __init__(self, date : datetime) -> None:
        """
        This class
        """
        self.load_data(date)
        return

    def load_data(self, date : datetime) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        
        """
        self.df_a = read_ac_data.read_ac_data_wrapper('A', date)
        self.df_b = read_ac_data.read_ac_data_wrapper('B', date)
        return self.df_a, self.df_b

    def shift_time(self) -> None:
        """

        """
        # Add the time lag
        self.df_b['dateTime_shifted'] = (self.df_b['dateTime'] - 
                                        pd.to_timedelta(self.df_b.Lag_In_Track, unit='s'))
        # Round to nearest tenths second and strip timezone info.
        self.df_b['dateTime_shifted'] = self.df_b['dateTime_shifted'].dt.round('0.1S')
        self.df_b['dateTime_shifted'] = self.df_b['dateTime_shifted'].dt.tz_localize(None)
        return

    def align_space_time_stamps(self) -> None:
        """

        """
        idxa = np.in1d(self.df_a['dateTime'], self.df_b['dateTime_shifted'], 
                        assume_unique=True)
        idxb = np.in1d(self.df_b['dateTime_shifted'], self.df_a['dateTime'], 
                        assume_unique=True)

        self.df_a = self.df_a.iloc[idxa, :]
        self.df_b = self.df_b.iloc[idxb, :]
        return

    def rolling_correlation(self, window:int = 5) -> None:
        """
        Use pandas.rolling_corr to cross correlate the spatially aligned time series.
        """
        self.corr = self.df_a['dos1rate'].rolling(window).corr(self.df_b['dos1rate'])
        return

    def plot_time_and_space_aligned(self, ax=None) -> None:
        """

        """
        if ax is None:
            _, ax = plt.subplots(3, sharex=True)

        ax[0].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r', label='AC6A')
        ax[0].plot(self.df_b.dateTime, self.df_b.dos1rate, 'b', label='AC6B')

        ax[1].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r')
        ax[1].plot(self.df_b.dateTime_shifted, self.df_b.dos1rate, 'b')

        ax[2].plot(self.df_b.dateTime, self.corr, 'k')
        
        ax[0].legend(loc=1)
        plt.show()
        return

if __name__ == '__main__':
    #s = SpatialAlign(datetime(2016, 10, 14))
    s = SpatialAlign(datetime(2015, 7, 27))
    s.shift_time()
    s.align_space_time_stamps()
    s.rolling_correlation(5)
    s.plot_time_and_space_aligned()
