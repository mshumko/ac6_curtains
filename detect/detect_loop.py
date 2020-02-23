import pandas as pd
import numpy as np
import os

import detect_daily
import dirs

class DetectCurtainLoop():
    def __init__(self, save_name, save_dir=None, std_thresh=2):
        if save_dir is None:
            save_dir = dirs.CATALOG_DIR
        else:
            save_dir = save_dir
        self.save_path = os.path.join(save_dir, save_name)
        self.std_thresh = std_thresh
        return

    def loop(self, overwrite=True):
        """ Loop over all the AC6 days and find curtains."""
        if overwrite:
            start_date = '2014-01-01'
        else:
            df = pd.read_csv(self.save_path)
            df['dateTime_A'] = pd.to_datetime(df['dateTime_A'])
            print(df['dateTime_A'].dt.date)
            start_date = df.loc[-1, 'dateTime_A'].date

        dates = pd.date_range(start=start_date, end='2017-07-01', freq='1D')

        for date in dates:
            # Make the detections using the detect_daily.DetectDailyCurtains
            # class.
            daily_detections = detect_daily.DetectDailyCurtains(date)
            daily_detections.load_data(date)
            daily_detections.shift_time()
            daily_detections.align_space_time_stamps()
            #daily_detections.rolling_correlation()
            daily_detections.baseline_significance()
            daily_detections.detect(std_thresh=self.std_thresh)
            daily_detections.find_peaks()

            # Now save the detections to a file.
            

        return