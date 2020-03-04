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

    def loop(self):
        """
        Loop over all of the days and find the peak width for each curtain.
        """
        width_df = pd.DataFrame(data=np.nan*np.ones((self.cat.shape[0], 2)),
                                index=self.cat.index,
                                columns=['width_A', 'width_B'])
        current_date = pd.to_datetime('2014-01-01')

        for i, (t0, row) in enumerate(self.cat.iterrows()):
            # Load the data from this day if it has not already.
            if current_date != t0.date():
                #daily_detections = detect_daily.DetectDailyCurtains(t0)
                self.date = t0.date()
                self.load_data(t0)
                self.shift_time()
                self.align_space_time_stamps()
                current_date = t0

            center_time_A = np.where(self.df_a['dateTime'] == t0)[0]
            center_time_B = np.where(self.df_b['dateTime_shifted'] == t0)[0]
            assert ((len(center_time_A) == 1) and (len(center_time_B) == 1)), 'No matches found!'
            peak_id_window_dp = 10    
            peak_A = np.argmax(self.df_a['dos1rate'][center_time_A[0]-peak_id_window_dp:center_time_A[0]+peak_id_window_dp])
            peak_B = np.argmax(self.df_b['dos1rate'][center_time_B[0]-peak_id_window_dp:center_time_B[0]+peak_id_window_dp])    

            widths_A, widths_B = self.calc_peak_width(peak_A, peak_B)

            plot_width = 10
            plot_width_dp = plot_width//2*10
            if plot_width:
                plt.plot(self.df_a['dateTime'][center_time_A[0]-plot_width_dp:center_time_A[0]+plot_width_dp], 
                        self.df_a['dos1rate'][center_time_A[0]-plot_width_dp:center_time_A[0]+plot_width_dp], 'r')
                plt.plot(self.df_b['dateTime_shifted'][center_time_B[0]-plot_width_dp:center_time_B[0]+plot_width_dp], 
                        self.df_b['dos1rate'][center_time_B[0]-plot_width_dp:center_time_B[0]+plot_width_dp], 'b')
                plt.axvline(self.df_a['dateTime'][peak_A], c='r')
                plt.axvline(self.df_b['dateTime_shifted'][peak_B], c='b')

                plt.hlines(widths_A[1], 
                            self.df_a['dateTime'][int(widths_A[2])], 
                            self.df_a['dateTime'][int(widths_A[3])], 
                            color="r", lw=3)
                plt.hlines(widths_B[1], 
                            self.df_b['dateTime_shifted'][int(widths_B[2])], 
                            self.df_b['dateTime_shifted'][int(widths_B[3])], 
                            color="b", lw=3)
                # plt.hlines(*widths_B[1:], color="b")

                plt.title(row.Lag_In_Track)
                plt.show()
            print(center_time_A, peak_A, center_time_B, peak_B)
            #print(self.calc_peak_width(peak_A, peak_B))

        return

    def calc_peak_width(self, peak_A, peak_B):
        """

        """
        widths_A = scipy.signal.peak_widths(self.df_a['dos1rate'], [peak_A], rel_height=self.rel_height)
        widths_B = scipy.signal.peak_widths(self.df_b['dos1rate'], [peak_B], rel_height=self.rel_height)
        return widths_A, widths_B

if __name__ == '__main__':
    c = Curtain_Width('AC6_curtains_baseline_method_sorted_v0.txt')
    c.loop()