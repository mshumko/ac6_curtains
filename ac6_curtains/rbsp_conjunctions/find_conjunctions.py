import pathlib
import pandas as pd
import numpy as np
from datetime import datetime, date

import spacepy.pycdf

import dirs

class RBSP_AC6_Conjunctions:
    def __init__(self, catalog_name):
        self.catalog_name = catalog_name
        self.load_curtain_catalog()
        return

    def load_curtain_catalog(self):
        """
        Loads the AC6 curtain list and converts the dateTime 
        column into datetime objects. 
        """    
        cat_path = dirs.CATALOG_DIR / self.catalog_name
        self.cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)
        self.cat['date'] = self.cat.index.date
        return

    def loop(self):
        """
        Loops over the dates in 
        """
        current_date = date.min

        # Create arrays that will be filled.
        self.cat['rbspa_L'] = np.nan*np.ones(self.cat.shape[0])
        self.cat['rbspa_MLT'] = np.nan*np.ones(self.cat.shape[0])
        self.cat['rbspb_L'] = np.nan*np.ones(self.cat.shape[0])
        self.cat['rbspb_MLT'] = np.nan*np.ones(self.cat.shape[0])

        for t_i, row in self.cat.iterrows():
            # Load the current_date RBSP MagEIS data 
            if current_date != row.date:
                current_date = row.date
                self.mageis_a = self._load_mageis('a', row.date)
                self.mageis_b = self._load_mageis('b', row.date)

            # Find the closest MagEIS time to the curtain time.
            mageis_a_idx = self._closest_time_index(t_i, self.mageis_a['Epoch'][...])
            mageis_b_idx = self._closest_time_index(t_i, self.mageis_b['Epoch'][...])
            # Load the values into the self.cat DataFrame
            self.cat.loc[t_i, 'rbspa_L'] = self.mageis_a['L'][mageis_a_idx]
            self.cat.loc[t_i, 'rbspb_L'] = self.mageis_b['L'][mageis_b_idx]
            self.cat.loc[t_i, 'rbspa_MLT'] = self.mageis_a['MLT'][mageis_a_idx]
            self.cat.loc[t_i, 'rbspb_MLT'] = self.mageis_b['MLT'][mageis_b_idx]
        return

    def calc_dL_dMLT(self, L_thresh=1, MLT_thresh=2):
        """

        """
        self.dL_A = np.abs(self.cat['Lm_OPQ'] - self.cat['rbspa_L'])
        self.dL_B = np.abs(self.cat['Lm_OPQ'] - self.cat['rbspb_L'])
        self.dMLT_A = self._dmlt(self.cat['Lm_OPQ'], self.cat['rbspa_MLT'])
        self.dMLT_B = self._dmlt(self.cat['Lm_OPQ'], self.cat['rbspb_MLT'])

        self.rbspa_close_conjunctions = np.where((self.dL_A < L_thresh) & 
                                                (self.dMLT_A < MLT_thresh))[0]
        self.rbspb_close_conjunctions = np.where((self.dL_B < L_thresh) & 
                                                (self.dMLT_B < MLT_thresh))[0]
        return

    def _load_mageis(self, sc_id, current_date):
        """
        Loads the MagEIS data for RBSP-{sc_id} on date.
        """
        mageis_dir = dirs.RBSP_DIR('mageis', sc_id)

        date_str = current_date.strftime("%Y%m%d")
        glob_str = f'rbsp{sc_id.lower()}_rel04_ect-mageis-L3_{date_str}_v*.cdf'

        mageis_paths = list(pathlib.Path(mageis_dir).glob(glob_str))
        
        # Check that only one file was found
        if len(mageis_paths) == 0:
            raise FileNotFoundError(f'No MagEIS-{sc_id} files found at {mageis_dir} '
                                    f'for glob str {glob_str}.')
        elif len(mageis_paths) > 1:
            raise FileNotFoundError(f'Multiple MagEIS-{sc_id} files found: {mageis_paths}.')
        else:
            mageis_path = str(mageis_paths[0])
            mageis = spacepy.pycdf.CDF(mageis_path)
        return mageis

    def _closest_time_index(self, curtain_time, mageis_times, thresh_minute=1):
        """
        Given the curtain time and mageis_times array, find the closest time
        within thresh_minute. If no close time found, return np.nan.
        """
        dt = np.abs(curtain_time-mageis_times)
        dt_sec = [dt_i.seconds for dt_i in dt]
        idx_min = np.argmin(dt_sec)
        min_dt_min = np.abs((curtain_time - mageis_times[idx_min]).total_seconds())/60 

        if min_dt_min > thresh_minute:
            return np.nan
        else:
            return idx_min

    def save_curtain_catalog(self, save_name):
        """
        Saves the catalog with the RBSP-A/B L shells and MLT annotated.
        """
        cat_path = dirs.CATALOG_DIR / save_name
        self.cat.to_csv(cat_path, index_label='dateTime')
        return

    def _dmlt(self, a, b):
        """
        NAME:    dmlt(a, b)
        USE:     Finds the absolute value of the difference
                of two MLT arrays a and b. This function
                correctly maps the differences over midnight
                Example: MLTa = 23, MLTb = 2 => dMLT = 3, NOT
                21!
        INPUT:   Two integers, or arrays to be differenced
        RETURNS: difference in MLT. 
        AUTHOR:  Mykhaylo Shumko
        MOD:     2017-04-30
        """
        # Convert the difference to spduo-angle.
        arg = 2*np.pi*np.abs(a - b)/24 
        # Utilize the even symmetry of cos to get 
        # the correct dmlt.
        return 24/(2*np.pi)*np.arccos(np.cos(arg))

if __name__ == "__main__":
    # load_catalog_name = 'AC6_curtains_baseline_method_sorted_v0.csv'
    load_catalog_name = 'AC6_curtains_baseline_method_conjunctions_v0.csv'
    save_catalog_name = 'AC6_curtains_baseline_method_conjunctions_v0.csv'

    con = RBSP_AC6_Conjunctions(load_catalog_name)
    # try:
    #     con.loop()
    # finally:
    #     con.save_curtain_catalog(save_catalog_name)
    con.calc_dL_dMLT()