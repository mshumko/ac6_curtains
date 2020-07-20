import pathlib
import pandas as pd
import numpy as np

import dirs

class RBSP_AC6_Conjunctions:
    def __init__(self, catalog_name):
        self.catalog_name = catalog_name

        return

    def load_curtain_catalog(self):
        """
        Loads the AC6 curtain list and converts the dateTime 
        column into datetime objects. 
        """    
        cat_path = dirs.CATALOG_DIR / self.catalog_name
        self.cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)
        #self.cat['dateTime'] = pd.to_datetime(self.cat['dateTime'])
        return

if __name__ == "__main__":
    catalog_name = 'ac6_curtains_baseline_method_sorted_v0.csv'
    con = RBSP_AC6_Conjunctions(catalog_name)