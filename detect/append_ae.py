import numpy as np
import pandas as pd
import os
import glob

import dirs

class Append_AE:
    def __init__(self, catalog_name, ae_dir=None, save_name=None):
        """
        This class uses AE data downloaded from 
        http://wdc.kugi.kyoto-u.ac.jp/aeasy/index.html
        and finds the appropriate values to the curtain catalog.
        The appropriate AE values are found within one 
        minute of the curtain observation.
        """
        self.catalog_name = catalog_name
        self.cat_path = os.path.join(dirs.CATALOG_DIR, self.catalog_name)
        self.load_catalog()

        if ae_dir is None:
            self.ae_dir = os.path.join(dirs.BASE_DIR, 'data', 'ae')
        else:
            self.ae_dir = ae_dir
        self.load_ae()
        self.merge()
        self.save_catalog(save_name=save_name)
        return

    def load_catalog(self):
        self.cat = pd.read_csv(self.cat_path, index_col=0)
        self.cat.index = pd.to_datetime(self.cat.index)
        return

    def load_ae(self):
        """
        Loads the AE index data for all of the years in self.cat.
        """
        years = sorted(set(self.cat.index.year))

        self.ae = pd.DataFrame(data=np.zeros((0, 1)), columns=['AE'])

        for year in years:
            ae_path = os.path.join(self.ae_dir, f'{year}_ae.txt')
            year_ae = pd.read_csv(ae_path, delim_whitespace=True, 
                            usecols=[0, 1, 3], skiprows=14, 
                            parse_dates=[['DATE', 'TIME']])
            year_ae.index=year_ae.DATE_TIME
            del year_ae['DATE_TIME']
            self.ae = self.ae.append(year_ae)
        
        # Drop duplicate indicies
        self.ae = self.ae.groupby(self.ae.index).first()
        return

    def merge(self, tolerance_min=1):

        self.cat = pd.merge_asof(self.cat, self.ae, left_index=True, 
                    right_index=True, direction='nearest', 
                    tolerance=pd.Timedelta(minutes=tolerance_min))
        return

    def save_catalog(self, save_name=None):
        """
        Saves the edited catalog to save_name which is the same as 
        self.catalog_name by default.
        """
        self.cat.to_csv(self.cat_path, index=False)
        return

if __name__ == '__main__':
    a = Append_AE('AC6_curtains_baseline_method_sorted_v0.txt')