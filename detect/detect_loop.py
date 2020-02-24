import pandas as pd
import numpy as np
import os
import time
import progressbar

import detect_daily
import dirs

class DetectCurtainLoop():
    def __init__(self, save_name, save_dir=None, std_thresh=2, corr_thresh=0.8):
        if save_dir is None:
            save_dir = dirs.CATALOG_DIR
        else:
            save_dir = save_dir
        self.save_path = os.path.join(save_dir, save_name)
        self.std_thresh = std_thresh
        self.corr_thresh = corr_thresh
        return

    def loop(self, overwrite=True):
        """ 
        Loop over all the AC6 days, find curtains, and save to a file.
        """
        if overwrite:
            start_date = '2014-01-01'
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
                print(f'Warning: deleted the curtain catalog '
                        f'in {self.save_path}')
            write_header = True
        else:
            if os.path.exists(self.save_path): 
                df = pd.read_csv(self.save_path)
                df['dateTime_A'] = pd.to_datetime(df['dateTime_A'])
                print(df['dateTime_A'].dt.date)
                start_date = df.loc[-1, 'dateTime_A'].date
                write_header = False
            else:
                start_date = '2014-01-01'
                write_header = True

        dates = pd.date_range(start=start_date, end='2017-07-01', freq='1D')

        for date in progressbar.progressbar(dates):
            # Make the detections using the detect_daily.DetectDailyCurtains
            # class.
            daily_detections = detect_daily.DetectDailyCurtains(date)
            try:
                daily_detections.load_data(date)
            except AssertionError as err: 
                if (('None or > 1 AC6 files found in' in str(err)) or
                    'File is empty!' in str(err)):
                    del(daily_detections)
                    continue
                else:
                    raise
            daily_detections.shift_time()
            daily_detections.align_space_time_stamps()
            daily_detections.rolling_correlation()
            daily_detections.baseline_significance()
            try:
                daily_detections.detect(
                    std_thresh=self.std_thresh, 
                    corr_thresh=self.corr_thresh
                    )
                daily_detections.find_peaks()
            except ValueError as err:
                if 'No detections were found on this day.' in str(err):
                    continue
                else:
                    raise
            daily_detections.catalog_detections()

            # Now save the detections to a file.
            daily_detections.detections_df.to_csv(
                self.save_path, index=False, header=write_header, mode='a'
                )
            write_header = False
            del(daily_detections)
        return

if __name__ == '__main__':
    start_time = time.time()
    d = DetectCurtainLoop(
        'ac6_curtains_baseline_method_v0.csv',
        std_thresh=2,
        corr_thresh=0.8
    )
    d.loop(overwrite=True)
    print(f'Loop run in {round(time.time() - start_time)} seconds.')