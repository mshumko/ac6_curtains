
import pandas as pd
import progressbar
from datetime import datetime
import os
import numpy as np

import detect_daily
import dirs

# Before: 22.1 GB for all files.
class CombineData():
    def __init__(self, save_dir=None):
        """
        This program combines 10 Hz data from both spacecraft and 
        preserves only the data taken at the same time.
        """
        if save_dir is None:
            self.save_dir = dirs.AC6_MERGED_DATA_DIR
        else:
            self.save_dir = save_dir
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f'Made {self.save_dir} directory.')
        return

    def loop(self):
        """ 
        Loop over all the AC6 days, find curtains, and save to a file.
        """
        dates = pd.date_range(start='2014-01-01', end='2017-07-01', freq='1D')

        for date in progressbar.progressbar(dates):
            # Make the detections using the detect_daily.DetectDailyCurtains
            # class.
            self.daily_detections = detect_daily.DetectDailyCurtains(date)
            try:
                self.daily_detections.load_data(date)
            except AssertionError as err: 
                if (('None or > 1 AC6 files found in' in str(err)) or
                    'File is empty!' in str(err)):
                    del(self.daily_detections)
                    continue
                else:
                    raise
            self.daily_detections.shift_time()
            self.daily_detections.align_space_time_stamps()
            self.drop_columns()
            self.merge_data(self.daily_detections.df_a, 
                            self.daily_detections.df_b)
            if self.df_merged.shape[0] > 0:
                self.save_data()
            del(self.daily_detections)
        return

    def merge_data(self, df_a, df_b, move_columns=['dateTime', 'dateTime_B', 'dateTime_shifted_B']):
        """
        This method handles the merging between the two data frames.
        """
        new_columns = [column + '_B' for column in df_b.columns]
        df_b.columns = new_columns
        self.df_merged = df_a.merge(df_b, left_index=True, right_index=True)

        # Rearange the columns to move the datetime columns to the beginning.
        merged_columns = self.df_merged.columns.tolist()
        for i, c in enumerate(move_columns):
            merged_columns.remove(c)
            merged_columns.insert(i, c)
        self.df_merged = self.df_merged[merged_columns]
        return

    def save_data(self):
        """
        Saves data to a csv file in self.save_dir.
        """
        day = datetime.strftime(
                        self.daily_detections.df_a.loc[0, 'dateTime'], "%Y%m%d"
                        )
        save_name = f'AC6_{day}_L2_10Hz_V03_merged.csv'
        save_path = os.path.join(self.save_dir, save_name)
        self.df_merged.to_csv(save_path, index=False)
        return

    def drop_columns(self, columns='default'):
        """
        Drop a few columns that are no longer necessary.
        """
        if columns=='default':
            columns=[
                'year','month','day','hour','minute','second','dos1l',
                'dos1m', 'dos2l','dos2m','dos3l','dos3m'
                    ]

        self.daily_detections.df_a.drop(columns=columns, inplace=True)
        self.daily_detections.df_b.drop(columns=columns, inplace=True)
        return
if __name__ == '__main__':
    c = CombineData()
    c.loop()