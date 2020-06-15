import numpy as np
import pandas as pd
import scipy.signal
import os
import matplotlib.pyplot as plt

import dirs
import detect_daily

class Curtain_Width(detect_daily.DetectDailyCurtains):
    def __init__(self, catalog_name, rel_height=0.5):
        """

        """
        self.catalog_name = catalog_name
        self.catalog_path = os.path.join(dirs.CATALOG_DIR, self.catalog_name)
        self.rel_height = rel_height

        super().__init__(None)

        self.load_catalog()
        return

    def load_catalog(self):
        """
        Loads the catalog
        """
        
        self.cat = pd.read_csv(self.catalog_path, index_col=0)
        time_keys = [key for key in self.cat.columns if 'time' in key.lower()]

        for time_key in time_keys:
            self.cat[time_key] = pd.to_datetime(self.cat[time_key])
        self.cat.index = pd.to_datetime(self.cat.index)
        return

    def loop(self, test_plots=False):
        """
        Loop over all of the days and find the peak width for each curtain.
        """
        width_df = pd.DataFrame(data=np.nan*np.ones((self.cat.shape[0], 2)),
                                index=self.cat.index,
                                columns=['width_A', 'width_B'])
        self.date = pd.to_datetime('2014-01-01')

        #for t0, row in self.cat.iterrows():
        for t0 in self.cat.index:
            # Load the data from this day if it has not already.
            if self.date != t0.date():
                print(f'Loading data from {t0.date()}')
                #daily_detections = detect_daily.DetectDailyCurtains(t0)
                self.date = t0.date()
                self.load_data(t0)
                self.shift_time()
                self.align_space_time_stamps()
                #current_date = t0
                
            center_time_A = np.where(self.df_a['dateTime'] == t0)[0]
            center_time_B = np.where(self.df_b['dateTime_shifted'] == t0)[0]
            assert ((len(center_time_A) == 1) and (len(center_time_B) == 1)), 'No matches found!'
            peak_id_window_dp = 10    
            peak_A = np.argmax(self.df_a['dos1rate'][center_time_A[0]-peak_id_window_dp:
                                center_time_A[0]+peak_id_window_dp])
            peak_B = np.argmax(self.df_b['dos1rate'][center_time_B[0]-peak_id_window_dp:
                                center_time_B[0]+peak_id_window_dp])    
            # peak_A = self.df_a['dos1rate'][center_time_A[0]-peak_id_window_dp:
            #                             center_time_A[0]+peak_id_window_dp].argmax()+\
            #                                 center_time_A[0]-peak_id_window_dp
            # peak_B = self.df_b['dos1rate'][center_time_B[0]-peak_id_window_dp:
            #                             center_time_B[0]+peak_id_window_dp].argmax()+\
            #                                 center_time_B[0]-peak_id_window_dp
            try:
                widths_A, widths_B = self.calc_peak_width(peak_A, peak_B)
            except ValueError as err:
                if 'is not a valid peak' in str(err):
                    print(err)
                    continue
                else:
                    raise

            width_df.loc[t0, 'width_A'] = widths_A[0]*0.1
            width_df.loc[t0, 'width_B'] = widths_B[0]*0.1

            if test_plots:
                self.make_test_plot([peak_A, peak_B], [widths_A, widths_B])
        
        # Merge data frames
        self.cat = self.cat.merge(width_df, left_index=True, right_index=True)

        return

    def calc_peak_width(self, peak_A, peak_B):
        """
        Use scipy's peak_widths function to estimate the peak width for 
        the peaks in AC6A and AC6B time series using the topographic 
        prominence.
        """
        widths_A = scipy.signal.peak_widths(self.df_a['dos1rate'], [peak_A], 
                                            rel_height=self.rel_height)
        widths_B = scipy.signal.peak_widths(self.df_b['dos1rate'], [peak_B], 
                                            rel_height=self.rel_height)
        return widths_A, widths_B

    def make_test_plot(self, peak_idx, widths, plot_width_s=10):
        """
        Make test plots to varify the peak width calculation.
        peak_idx is an array of two elements representing the 
        peaks from AC6A and B. widths argument is a two element
        tuple containing the output from scipy.signal.find_peaks().
        """
        plot_width_dp = plot_width_s//2*10
        _, ax = plt.subplots(2, sharex=True)
        ax[0].plot(self.df_a['dateTime'][peak_idx[0]-plot_width_dp:peak_idx[0]+plot_width_dp], 
                self.df_a['dos1rate'][peak_idx[0]-plot_width_dp:peak_idx[0]+plot_width_dp], 'r')
        ax[1].plot(self.df_b['dateTime_shifted'][peak_idx[1]-plot_width_dp:peak_idx[1]+plot_width_dp], 
                self.df_b['dos1rate'][peak_idx[1]-plot_width_dp:peak_idx[1]+plot_width_dp], 'b')
        ax[0].axvline(self.df_a['dateTime'][peak_idx[0]], c='r')
        ax[1].axvline(self.df_b['dateTime_shifted'][peak_idx[1]], c='b')

        ax[0].hlines(widths[0][1], 
                    self.df_a['dateTime'][int(widths[0][2])], 
                    self.df_a['dateTime'][int(widths[0][3])], 
                    color="r", lw=3)
        ax[1].hlines(widths[1][1], 
                    self.df_b['dateTime_shifted'][int(widths[1][2])], 
                    self.df_b['dateTime_shifted'][int(widths[1][3])], 
                    color="b", lw=3)
        plt.show()
        return

    def save_catalog(self, catalog_name='same'):
        if catalog_name == 'same':
            catalog_name = self.catalog_name
        catalog_path = os.path.join(dirs.CATALOG_DIR, catalog_name)
        self.cat.to_csv(catalog_path)


if __name__ == '__main__':
    c = Curtain_Width('AC6_curtains_baseline_method_sorted_v0.csv')
    c.loop(test_plots=False)
    c.save_catalog()