import pandas as pd
import numpy as np
import matplotlib
import matplotlib.dates 
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
        seconds_in_a_day = 60*60*24
        numeric_time_b = matplotlib.dates.date2num(self.df_b.dateTime)

        # Add the time lag
        numeric_time_b += self.df_b.Lag_In_Track/seconds_in_a_day
        self.df_b['dateTime_shifted'] = matplotlib.dates.num2date(numeric_time_b)
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

        ax[2].plot(self.df_b.dateTime, self.df_b.Lag_In_Track, 'k')
        
        ax[0].legend(loc=1)
        plt.show()
        return

if __name__ == '__main__':
    s = SpatialAlign(datetime(2016, 10, 14))
    s.shift_time()
    s.plot_time_and_space_aligned()
