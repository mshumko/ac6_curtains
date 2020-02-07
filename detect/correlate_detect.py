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
        dt = np.round(self.df_b.Lag_In_Track, 1)
        self.df_b['dateTime_shifted'] = (self.df_b['dateTime'] - pd.to_timedelta(dt, unit='s'))
        # Round to nearest tenths second and strip timezone info.
        #self.df_b['dateTime_shifted'] = self.df_b['dateTime_shifted'].dt.round('0.1S')
        self.df_b['dateTime_shifted'] = self.df_b['dateTime_shifted'].dt.tz_localize(None)
        # Drop duplicate times
        self.df_b = self.df_b.drop_duplicates(subset='dateTime_shifted')
        return

    def align_space_time_stamps(self) -> None:
        """

        """
        idxa = np.where(np.in1d(self.df_a['dateTime'], self.df_b['dateTime_shifted'], 
                        assume_unique=True))[0]
        idxb = np.where(np.in1d(self.df_b['dateTime_shifted'], self.df_a['dateTime'], 
                        assume_unique=True))[0]

        self.df_a = self.df_a.iloc[idxa, :]
        self.df_b = self.df_b.iloc[idxb, :]

        self.df_a.index = np.arange(self.df_a.shape[0])
        self.df_b.index = np.arange(self.df_b.shape[0])
        return

    def rolling_correlation(self, window:int=5) -> None:
        """
        Use df.rolling.corr to cross correlate the spatially aligned time series.
        """
        self.corr = self.df_b['dos1rate'].rolling(window).corr(self.df_a['dos1rate'])
        return

    def baseline_significance(self, widnow:int=5, significance:float=2) -> None:
        """
        Finds the data points that are significance number of standard deviations above
        a top hat rolling average window of width window.
        """
        rolling_average_a = self.df_a['dos1rate'].rolling(window=widnow).mean()
        rolling_average_b = self.df_b['dos1rate'].rolling(window=widnow).mean()

        self.n_std_a = (self.df_a['dos1rate']-rolling_average_a)/np.sqrt(rolling_average_a)
        self.n_std_b = (self.df_b['dos1rate']-rolling_average_b)/np.sqrt(rolling_average_b)
        return

    def plot_time_and_space_aligned(self, ax=None) -> None:
        """

        """
        if ax is None:
            _, ax = plt.subplots(3, sharex=True)

        ax[0].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r', label='AC6A')
        #ax[0].plot(self.df_b.dateTime, self.df_b.dos1rate, 'b', label='AC6B')

        ax[1].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r')
        #ax[1].plot(self.df_b.dateTime_shifted, self.df_b.dos1rate, 'b')

        #ax[2].plot(self.df_a.dateTime, self.corr, 'k')
        ax[2].plot(self.df_a.dateTime, self.n_std_a, 'r')
        #ax[2].plot(self.df_b.dateTime, self.n_std_b, 'b')

        idx_signif = np.where(self.n_std_a > 5)[0]
        ax[0].scatter(self.df_a.dateTime[idx_signif], self.df_a.dos1rate[idx_signif], c='g', s=10)
        
        ax[0].legend(loc=1)
        plt.show()
        return

if __name__ == '__main__':
    #s = SpatialAlign(datetime(2016, 10, 14))
    s = SpatialAlign(datetime(2015, 7, 27))
    s.shift_time()
    s.align_space_time_stamps()
    # s.rolling_correlation(10)
    s.baseline_significance(50)
    s.plot_time_and_space_aligned()
