import numpy as np
import pandas as pd
import scipy.signal
import os
import matplotlib.pyplot as plt

import dirs
import detect_daily

class Curtain_Width:
    def __init__(self, catalog_name, rel_height=0.5):
        """

        """
        self.catalog_name = catalog_name
        self.catalog_path = os.path.join(dirs.CATALOG_DIR, self.catalog_name)
        self.rel_height = rel_height

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
        width_df = pd.DataFrame(data=np.nan*np.ones((self.cat.shape[0], 2)),
                                index=self.cat.index,
                                columns=['width_A', 'width_B'])
        print(width_df)
        current_date = pd.to_datetime('2014-01-01')

        for i, (date, row) in enumerate(self.cat.iterrows()):
            # date = date.date()

            # Load the data from this day if it has not already.
            if current_date != date.date():
                daily_detections = detect_daily.DetectDailyCurtains(date)
                daily_detections.load_data(date)
                daily_detections.shift_time()
                daily_detections.align_space_time_stamps()
                current_date = date

        return

    def calc_peak_width(self):
        """

        """
        widths = scipy.signal.peak_widths(x, peaks, rel_height=self.rel_height)
        return

if __name__ == '__main__':
    c = Curtain_Width('AC6_curtains_baseline_method_sorted_v0.txt')
    c.loop()