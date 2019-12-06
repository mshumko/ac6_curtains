# Look for curtains in the BLC
import numpy as np
import pandas as pd
import os

BASE_DIR = '/home/mike/research/ac6_curtains/'
CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
CATALOG_PATH = os.path.join(BASE_DIR, 'data/catalogs', CATALOG_NAME)
cat = pd.read_csv(CATALOG_PATH)

def filter_blc(cat, lat_edges=[30, 75], lon_edges=[-45, 30]):
    """ 
    Takes in a catalog and filters out events outside of 
    the lat_edges and lon_edges. 
    """
    cat = cat[(cat.loc[:, 'lat'] > lat_edges[0]) & 
              (cat.loc[:, 'lat'] < lat_edges[1])]
    cat = cat[(cat.loc[:, 'lon'] > lon_edges[0]) & 
              (cat.loc[:, 'lon'] < lon_edges[1])]
    return cat

print(f'Total number of curtains = {cat.shape[0]}')
print(f'Number of curtains in the BLC = {filter_blc(cat).shape[0]}')