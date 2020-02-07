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
        self.date = date
        self.load_data(self.date)
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
        self.corr_window = window
        self.corr = self.df_b['dos1rate'].rolling(self.corr_window).corr(self.df_a['dos1rate'])
        return

    def baseline_significance(self, baseline_window:int=5, significance:float=2) -> None:
        """
        Finds the data points that are significance number of standard deviations above
        a top hat rolling average window of width window.
        """
        self.baseline_window = baseline_window
        rolling_average_a = self.df_a['dos1rate'].rolling(window=baseline_window).mean()
        rolling_average_b = self.df_b['dos1rate'].rolling(window=baseline_window).mean()

        self.n_std_a = (self.df_a['dos1rate']-rolling_average_a)/np.sqrt(rolling_average_a)
        self.n_std_b = (self.df_b['dos1rate']-rolling_average_b)/np.sqrt(rolling_average_b)
        return

    def plot_time_and_space_aligned(self, ax=None, std_thresh:float=5, corr_thresh:float=0.8) -> None:
        """

        """
        if ax is None:
            _, ax = plt.subplots(4, sharex=True, figsize=(8, 8))

        ax[0].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r', label='AC6A')
        ax[0].plot(self.df_b.dateTime, self.df_b.dos1rate, 'b', label='AC6B')

        ax[1].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r')
        ax[1].plot(self.df_b.dateTime_shifted, self.df_b.dos1rate, 'b')

        ax[2].plot(self.df_a.dateTime, self.corr, 'k')
        ax[2].axhline(corr_thresh, c='k', ls='--')

        ax[3].plot(self.df_a.dateTime, self.n_std_a, 'r')
        ax[3].plot(self.df_b.dateTime_shifted, self.n_std_b, 'b')
        ax[3].axhline(std_thresh, c='k', ls='--')

        idx_signif = np.where(self.n_std_a > std_thresh)[0]
        ax[1].scatter(self.df_a.dateTime[idx_signif], self.df_a.dos1rate[idx_signif], c='g', s=10)

        ax[0].set(ylabel='dos1rate\ntime-aligned', 
                title=f'AC6 peak detection and curtain correlation\n{self.date.date()}')
        ax[1].set(ylabel='dos1rate\nspace-aligned')
        ax[2].set(ylabel=f'spatial correlation\nwindow={self.corr_window/10} s')
        ax[3].set(ylabel=f'std above {self.baseline_window/10} s\nmean baseline')
        ax[3].set_ylim(bottom=0, top=3*std_thresh)

        
        ax[0].legend(loc=1)
        plt.tight_layout()
        plt.show()
        return

if __name__ == '__main__':
    #s = SpatialAlign(datetime(2016, 10, 14))
    s = SpatialAlign(datetime(2015, 7, 27))
    s.shift_time()
    s.align_space_time_stamps()
    s.rolling_correlation(10)
    s.baseline_significance(baseline_window=50)
    s.plot_time_and_space_aligned()
