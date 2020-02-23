import pandas as pd
import numpy as np
from datetime import datetime
import typing

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib.ticker import FuncFormatter

# Set up enviromental variables in the directory that contains the 
# mission_tools repo.
import mission_tools.ac6.read_ac_data as read_ac_data
import mission_tools.misc.locate_consecutive_numbers as locate_consecutive_numbers

class DetectDailyCurtains:
    def __init__(self, date : datetime, bad_flags:typing.List[int]=[1,2], 
                detect_script=True) -> None:
        """
        This class detects curtains for one day. Put this in a loop over days 
        to find all curtains in the AC6 data.
        """
        self.date = date
        self.bad_flags = bad_flags
        return

    def load_data(self, date : datetime) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Attempts to load the AC6 data into two DataFrames.
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

    def rolling_correlation(self, window:int=20) -> None:
        """
        Use df.rolling.corr to cross correlate the spatially-aligned time series.
        """
        self.corr_window = window
        self.corr = self.df_b['dos1rate'].rolling(self.corr_window).corr(self.df_a['dos1rate'])
        # Mark bad correlations with np.nan
        self.corr[self.corr > 1] = np.nan
        # Now roll the self.corr to center the non-NaN values
        self.corr = np.roll(self.corr, -window//2)
        return

    def baseline_significance(self, baseline_window:int=100) -> None:
        """
        Calculates the number of standard deviations, assuming Poisson statistics, that a
        a count value is above a rolling average baseline of length baseline_window.
        """ 
        self.baseline_window = baseline_window
        rolling_average_a = self.df_a['dos1rate'].rolling(window=baseline_window).mean()/10
        rolling_average_b = self.df_b['dos1rate'].rolling(window=baseline_window).mean()/10

        self.n_std_a = (self.df_a['dos1rate']/10-rolling_average_a)/np.sqrt(rolling_average_a+1)
        self.n_std_b = (self.df_b['dos1rate']/10-rolling_average_b)/np.sqrt(rolling_average_b+1)
        return

    def valid_data_flag(self) -> typing.Set:
        """
        Use bitwise operations to filter out data with invalid data flags 
        using self.bad_flags attribute. Returns a set.
        """
        idx = set(np.arange(self.df_a.shape[0]))

        for flag in self.bad_flags:
            # Returns a non-zero value if that flag was found in there
            a_and_flags = np.bitwise_and(self.df_a.flag, (1 << flag-1)).astype(bool)
            b_and_flags = np.bitwise_and(self.df_b.flag, (1 << flag-1)).astype(bool)

            idx_i = set(np.where((~a_and_flags) & (~b_and_flags))[0])
            idx.intersection_update(idx_i)
        return idx

    def detect(self, std_thresh:float=2, corr_thresh:float=None) -> typing.List[int]:
        """
        After running the baseline_significance() and/or rolling_correlation() methods,
        use this method to organize the detections. 
        """
        self.std_thresh = std_thresh
        self.corr_thresh = corr_thresh

        # Find indicies in the AC6A and B data that are significant 
        # above the background
        idx_signif = np.where((self.n_std_a > std_thresh) & 
                            (self.n_std_b > std_thresh))[0]
        if corr_thresh is not None:
            # Find indicies where the temporal time series was 
            # highly correlated
            idx_corr = np.where(self.corr > corr_thresh)[0]
            idx_signif = list(set(idx_signif).intersection(idx_corr))
        # Now find good quality data.
        valid_data = self.valid_data_flag()
        self.detections = np.array(list(set(idx_signif).intersection(valid_data)))
        self.detections = np.sort(self.detections)
        return self.detections

    def find_peaks(self):
        """
        Given the self.detections array, find each continious index interval and 
        find the index with highest count rates for each interval.
        """
        startInd, endInd = locate_consecutive_numbers.locateConsecutiveNumbers(
            self.detections) # Find consecutive numbers to get a max of first
        self.peaks_A = np.nan*np.ones(len(startInd), dtype=int)
        self.peaks_B = np.nan*np.ones(len(startInd), dtype=int)
        # Loop over every microburst detection region (consecutive microburst indicies)
        for i, (st, et) in enumerate(zip(startInd, endInd)):
            if st == et: 
                # If the interval is just one point 
                et += 1
            # Find the max and reindex.
            offset = self.df_a.index[self.detections[st]]
            self.peaks_A[i] = np.argmax(
                    self.df_a.loc[self.detections[st:et], 'dos1rate']) + offset
            self.peaks_B[i] = np.argmax(
                    self.df_b.loc[self.detections[st:et], 'dos1rate']) + offset
        return

    def catalog_detections(self, aux_columns='default'):
        """
        This method creates a pandas DataFrame that has the curtain time from the 
        AC6A, and shifted AC6B data. Auxiliary data columns such as L, MLT, lat, 
        lon, alt, In_Track_Lag, etc are specified by the aux_columns kwarg. If 
        aux_columns='default', a default set of AC6 data keys will be used, 
        otherwise provide a list of aux_columns that exist in the AC6 10 Hz data.
        """
        if aux_columns=='default':
            aux_columns = [
                'Lm_OPQ','MLT_OPQ','lat','lon','alt','Dist_In_Track',
                'Lag_In_Track','Dist_Total', 'Loss_Cone_Type','flag'
                ]
        # The dateTime column redundancy in times_df is to make the 
        # DataFrame backwards compatable with the legacy code. 
        times_df = pd.DataFrame(
            data=np.array([
                self.df_a.loc[self.peaks_A, 'dateTime'], 
                self.df_a.loc[self.peaks_A, 'dateTime'],
                self.df_b.loc[self.peaks_B, 'dateTime']]).T,
            index=np.arange(len(self.peaks_A)), 
            columns=['dateTime', 'time_spatial_A', 'time_spatial_B']
            )
        aux_df = self.df_a.loc[self.peaks_A, aux_columns]
        aux_df.index = np.arange(len(self.peaks_A))
        self.detections_df = times_df.merge(aux_df, left_index=True, right_index=True)
        return

class Validate_Detections(DetectDailyCurtains):
    def __init__(self, date:datetime, bad_flags:typing.List[int]=[1,2], 
                std_thresh:float=2, corr_thresh:float=None) -> None:
        """
        This class validates the detections made by the DetectDailyCurtains
        class.
        """
        super().__init__(date, bad_flags=bad_flags)
        self.std_thresh = std_thresh
        self.corr_thresh = corr_thresh
        return

    def validate(self):
        """
        A wrapper to make the curtain detections.
        """
        self.load_data(self.date)
        self.shift_time()
        self.align_space_time_stamps()
        self.rolling_correlation()
        self.baseline_significance()
        self.detect(std_thresh=self.std_thresh, corr_thresh=self.corr_thresh)
        self.find_peaks()
        self.catalog_detections()
        self.plot_validation()
        return

    def plot_validation(self, ax=None) -> None:
        """
        Makes a validation plot of the detections. By default it plots:
        the time-aligned data, space-aligned data, the rolling correlation,
        and the std significance.
        """
        self._plotLabels()
        
        if ax is None:
            _, ax = plt.subplots(4, sharex=True, figsize=(8, 8))

        ax[0].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r', label='AC6A')
        ax[0].plot(self.df_b.dateTime, self.df_b.dos1rate, 'b', label='AC6B')

        ax[1].plot(self.df_a.dateTime, self.df_a.dos1rate, 'r')
        ax[1].plot(self.df_b.dateTime_shifted, self.df_b.dos1rate, 'b')

        ax[2].plot(self.df_a.dateTime, self.corr, 'k')
        if self.corr_thresh is not None:
            ax[2].axhline(self.corr_thresh, c='k', ls='--')

        ax[3].plot(self.df_a.dateTime, self.n_std_a, 'r')
        ax[3].plot(self.df_b.dateTime_shifted, self.n_std_b, 'b')
        if self.std_thresh is not None:
            ax[3].axhline(self.std_thresh, c='k', ls='--')

        # Plot where the above conditions are true.
        ax[1].scatter(self.df_a.dateTime[self.detections], self.df_a.dos1rate[self.detections], 
                    c='g', s=40, label='Significant std A&B')
        if hasattr(self, 'peaks_A'):
            ax[1].scatter(self.df_a.dateTime[self.peaks_A], self.df_a.dos1rate[self.peaks_A], 
                    marker='x', c='r', s=60, label='A peaks')
            ax[1].scatter(self.df_b.dateTime_shifted[self.peaks_B], self.df_b.dos1rate[self.peaks_B], 
                    marker='x', c='b', s=60, label='B peaks')
        # ax[1].scatter(self.df_a.dateTime[idx_corr], 1.1*self.df_a.dos1rate[idx_corr], 
        #             c='g', s=20, marker='s', label='correlation signif')
        # ax[1].scatter(self.df_a.dateTime[idx_detect_valid_flag], 1.2*self.df_a.dos1rate[idx_detect_valid_flag],
        #             c='k', s=50, marker='X', label='detection')

        self._adjust_plots(ax)     
        plt.show()
        return

    def _adjust_plots(self, ax):
        """
        Makes small adjustments to the subplots.
        """
        ax[0].set(ylabel='dos1rate\ntime-aligned', 
                title=f'AC6 peak detection and curtain correlation\n{self.date.date()}')
        ax[1].set(ylabel='dos1rate\nspace-aligned')
        ax[2].set(ylabel=f'spatial correlation\nwindow={self.corr_window/10} s')
        ax[3].set(ylabel=f'std above {self.baseline_window/10} s\nmean baseline')
        ax[3].set_ylim(bottom=0, top=3*self.std_thresh)

        ax[1].legend(loc=1)
        ax[0].legend(loc=1)
        ax[-1].xaxis.set_major_formatter(FuncFormatter(self.format_fn))
        ax[-1].set_xlabel('time\nL\nMLT\nlat\nlon\nflag\nin-track lag')
        ax[-1].xaxis.set_label_coords(-0.1,-0.03)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.17)
        for a in ax:
            a.format_coord = lambda x, y: f'{matplotlib.dates.num2date(x).replace(tzinfo=None).isoformat()}' 
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
    # A few possible dates to play around with:
    # datetime(2016, 10, 14), datetime(2015, 7, 27), datetime(2015, 4, 9)
    v = Validate_Detections(datetime(2015, 7, 27))
    v.validate()