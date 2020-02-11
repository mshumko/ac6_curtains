import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime
import typing
from matplotlib.ticker import FuncFormatter

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

    def rolling_correlation(self, window:int=5, thresh:int=1) -> None:
        """
        Use df.rolling.corr to cross correlate the spatially aligned time series.
        """
        self.corr_window = window
        if thresh != 0:
            lags = np.arange(-thresh, thresh+1)
        else:
            lags = [0]
        # Correlation matrix of size n_samples x n_lags. At the end a 
        # max correlation at each sample will be taken.
        self.corr_matrix = np.zeros((len(self.df_b['dos1rate']), len(lags)))

        for i, lag in enumerate(lags):
            if lag < 0:
                r = self.df_a['dos1rate'][abs(lag):].rolling(self.corr_window)
                self.corr_matrix[:, i] = r.corr(self.df_b['dos1rate'][:lag])
            elif lag > 0:
                r = self.df_b['dos1rate'][abs(lag):].rolling(self.corr_window)
                self.corr_matrix[:, i] = r.corr(self.df_a['dos1rate'][:lag])
            else:
                r = self.df_b['dos1rate'].rolling(self.corr_window)
                self.corr_matrix[:, i] = r.corr(self.df_a['dos1rate'])
        self.corr = np.nanmax(self.corr_matrix, axis=1)

                
        
        #self.corr = self.df_b['dos1rate'].rolling(self.corr_window).corr(self.df_a['dos1rate'])
        # Now roll the self.corr to center the non-NaN values
        self.corr = np.roll(self.corr, -window//2)
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

    def valid_flag_data(self, flags=[1,2]):
        """
        Use bitwise operations to filter out data with invalid data flags.
        Returns a set.
        """
        idx = set(np.arange(self.df_a.shape[0]))

        for flag in flags:
            # Returns a non-zero value if that flag was found in there
            a_and_flags = np.bitwise_and(self.df_a.flag, (1 << flag-1)).astype(bool)
            b_and_flags = np.bitwise_and(self.df_b.flag, (1 << flag-1)).astype(bool)

            idx_i = set(np.where((~a_and_flags) & (~b_and_flags))[0])
            idx.intersection_update(idx_i)
        return idx


    def plot_time_and_space_aligned(self, ax=None, std_thresh:float=5, corr_thresh:float=0.7) -> None:
        """

        """
        self._plotLabels()
        
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

        # Find indicies in the AC6A and B data that are significant 
        # above the background
        idx_signif = np.where((self.n_std_a > std_thresh) | 
                            (self.n_std_b > std_thresh))[0]
        # Find indicies where the temporal time series was 
        # highly correlated
        idx_corr = np.where(self.corr > corr_thresh)[0]
        # Find where the two above conditions are true and good data
        idx_detect = np.where((self.n_std_a > std_thresh) & 
                              (self.corr > corr_thresh))[0]
        valid_flag_set = self.valid_flag_data()
        idx_detect_valid_flag = list(set(idx_detect).intersection(valid_flag_set))

        # Plot where the above conditions are true.
        ax[1].scatter(self.df_a.dateTime[idx_signif], self.df_a.dos1rate[idx_signif], 
                    c='g', s=40, label='std signif')
        ax[1].scatter(self.df_a.dateTime[idx_corr], 1.1*self.df_a.dos1rate[idx_corr], 
                    c='g', s=20, marker='s', label='correlation signif')
        ax[1].scatter(self.df_a.dateTime[idx_detect_valid_flag], 1.2*self.df_a.dos1rate[idx_detect_valid_flag],
                    c='k', s=50, marker='X', label='detection')

        ax[0].set(ylabel='dos1rate\ntime-aligned', 
                title=f'AC6 peak detection and curtain correlation\n{self.date.date()}')
        ax[1].set(ylabel='dos1rate\nspace-aligned')
        ax[2].set(ylabel=f'spatial correlation\nwindow={self.corr_window/10} s')
        ax[3].set(ylabel=f'std above {self.baseline_window/10} s\nmean baseline')
        ax[3].set_ylim(bottom=0, top=3*std_thresh)

        ax[1].legend(loc=1)
        ax[0].legend(loc=1)
        ax[-1].xaxis.set_major_formatter(FuncFormatter(self.format_fn))
        ax[-1].set_xlabel('time\nL\nMLT\nlat\nlon\nflag\nin-track lag')
        ax[-1].xaxis.set_label_coords(-0.1,-0.03)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.17)
        # plt.tight_layout()
        for a in ax:
            a.format_coord = lambda x, y: f'{matplotlib.dates.num2date(x).replace(tzinfo=None).isoformat()}'        
        plt.show()
        return

    def _plotLabels(self, skip_n:int=5):
        ### FORMAT X-AXIS to show more information ###
        self.df_a['Lm_OPQ'] = np.round(self.df_a['Lm_OPQ'], decimals=1)
        L = pd.DataFrame(self.df_a['Lm_OPQ'])#.astype(object))
        L = L.replace(np.nan, '', regex=True)
        time = self.df_a['dateTime']
        # This code is a nifty way to format the x-ticks to my liking.
        self.labels = [f'{t.replace(microsecond=0).time()}\n'
                       f'{L}\n{round(MLT,1)}\n{round(lat,1)}\n'
                       f'{round(lon,1)}\n{flag}\n{round(lag, 1)}' for 
                        (t, L, MLT, lat, lon, flag, lag) in zip(
                        time[::skip_n], L.loc[::skip_n, 'Lm_OPQ'], self.df_a['MLT_OPQ'][::skip_n], 
                        self.df_a['lat'][::skip_n], self.df_a['lon'][::skip_n], 
                        self.df_a['flag'][::skip_n], self.df_a['Lag_In_Track'][::skip_n])]  
        self.numeric_time = matplotlib.dates.date2num(time[::skip_n])
        return time, self.numeric_time, self.labels 

    def format_fn(self, tick_val, tick_pos):
        """
        The tick magic happens here. pyplot gives it a tick time, and this function 
        returns the closest label to that time. Read docs for FuncFormatter().
        """
        dt = self.numeric_time-tick_val
        # If time difference between matplotlib's tick and HiRes time stamp 
        # is larger than 30 minutes, skip that tick label.
        if np.min(np.abs(dt)) > 30/1440:
            return ''
        else:
            idx = np.argmin(np.abs(dt))
            return self.labels[idx]

if __name__ == '__main__':
    s = SpatialAlign(datetime(2016, 10, 14))
    # s = SpatialAlign(datetime(2015, 7, 27))
    s = SpatialAlign(datetime(2015, 4, 9))
    s.shift_time()
    s.align_space_time_stamps()
    s.rolling_correlation(window=10)
    s.baseline_significance(baseline_window=50)
    s.plot_time_and_space_aligned()
