# This script removes duplicate entries from the curtain dataset by
# looking for similar spatial times - within a few tenths of a second -
# using time_spatial_A/B columns, across multiple rows.

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

BASE_DIR = '/home/mike/research/ac6_curtains/'
CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
CATALOG_PATH = os.path.join(BASE_DIR, 'data/catalogs', CATALOG_NAME)
dt = timedelta(seconds=0.2) # time threshold

cat = pd.read_csv(CATALOG_PATH)
for timeKey in ['dateTime', 'time_spatial_A', 'time_spatial_B']:
    cat[timeKey] = pd.to_datetime(cat[timeKey])

# Double for loop to loop over each time, and look for occurances
# of the same spatial time (or similar time, within dt)
for i, row_i in cat.iterrows():
    for j, row_j in cat.iterrows():
        # If the two times in time_spatial_A and time_spatial_B are similar
        if (np.abs(row_i.time_spatial_A - row_j.time_spatial_B) <= dt) and (i != j):
            print('A,B:', i, j, row_i.time_spatial_A)
            cat = cat.drop(index=j)

cat.to_csv(CATALOG_PATH, index=False)
