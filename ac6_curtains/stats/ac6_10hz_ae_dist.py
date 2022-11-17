# This script calculates the distribution of the AE index
# when AC6 took 10 Hz data togeather with flag=0.

import numpy as np
import pandas as pd
import pathlib

import dirs

class AE_dist:
    def __init__(self, AE_bins):
        """
        Loop over all of the AC6 merged data and calculate the distribution
        of AE.
        """
        self.AE_bins = AE_bins
        self.H = np.zeros(len(AE_bins)-1)
        self._get_merged_file_list()
        self._load_AE()
        return

    def loop(self):
        """
        Loop over every merged file, filter out rows where flag or flag_B != 0, 
        then match the remaining 10 Hz data to self.AE and lastly histogram the AE
        values.
        """
        for merged_file in self.merged_files:
            print(f'Processing {merged_file.name}')
            # Load the merged files but only the dateTime column
            self.load_filter_10hz(merged_file)
            # If there are any merged data with both flags == 0.
            if self.merged_data.shape[0]:
                self.merge()
                self.hist()

        # Now package the results into the AE histogram DataFrame
        self.df_H = pd.DataFrame(data={'10Hz_samples':self.H.astype(int)}, index=self.AE_bins[:-1])
        return

    def load_filter_10hz(self, f):
        """
        Load the 10Hz data, parse the datetime column, and filter times when either 
        flag or flag_B are !=0 
        """
        data = pd.read_csv(f, index_col=0, parse_dates=True)
        self.merged_data = data[(data.flag == 0) & 
                                (data.flag_B == 0)]
        return

    def merge(self):
        """ 
        Applies the merge_asof opperation to self.merged_data and finds
        the nearest AE value within a minute (the AE data cadence).
        """
        self.df_merged_asof = pd.merge_asof(
                                self.merged_data, self.AE, 
                                left_index=True, right_index=True,
                                tolerance=pd.Timedelta(minutes=1)
                                )        
        return

    def hist(self):
        """
        Histogram the filtered AE data.
        """
        H_temp, _ = np.histogram(self.df_merged_asof.AE, bins=self.AE_bins)
        self.H += H_temp
        return

    def save_hist(self, file_name):
        """
        Saves df_H dataframe to a csv file in dirs.NORM_DIR
        """
        save_path = pathlib.Path(dirs.NORM_DIR, file_name)
        self.df_H.to_csv(save_path, index_label='AE')
    
    def _get_merged_file_list(self):
        """ Find all of the merged files and sort them by time. """
        file_generator = pathlib.Path(dirs.AC6_MERGED_DATA_DIR).rglob('*merged.csv')
        self.merged_files = sorted(list(file_generator))
        assert len(self.merged_files), f'No merged files found in {dirs.AC6_MERGED_DATA_DIR}'
        return

    def _load_AE(self):
        # Load the AE index for all of the years that curtains 
        # were observed
        ae_dir = pathlib.Path(dirs.BASE_DIR, 'data', 'ae')
        years = np.arange(2014, 2018)
        self.AE = pd.DataFrame(data=np.zeros((0, 1)), columns=['AE'])

        for year in years:
            ae_path = pathlib.Path(ae_dir, f'{year}_ae.txt')
            year_ae = pd.read_csv(ae_path, delim_whitespace=True, 
                            usecols=[0, 1, 3], skiprows=14, 
                            parse_dates=[['DATE', 'TIME']])
            year_ae.index=year_ae.DATE_TIME
            del year_ae['DATE_TIME']
            self.AE = self.AE.append(year_ae)
        # There is one overlapping day in each file so drop duplicate 
        # indices
        self.AE = self.AE.groupby(self.AE.index).first()
        return

    
if __name__ == '__main__':
    a = AE_dist(np.arange(0, 2001, 10))
    a.loop()
    a.save_hist('AE_10Hz_AE.csv')