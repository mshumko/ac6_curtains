
import pandas as pd
import progressbar

import detect_daily
import dirs

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
        return

    def loop(self):
        """ 
        Loop over all the AC6 days, find curtains, and save to a file.
        """
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
            
            # del(daily_detections)
        return