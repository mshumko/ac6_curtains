"""
Look at curtains observed in the dusk (21-1) MLTs and see how the 
number of curtains changes as a function of local time, hence the
position of the SAA.
"""
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import dateutil.parser

### Load curtain dataset ###
BASE_DIR = '/home/mike/research/ac6_curtains/'
CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
CATALOG_PATH = os.path.join(BASE_DIR, 'data/catalogs', CATALOG_NAME)
cat = pd.read_csv(CATALOG_PATH)

# Filter dusk events
START_MLT = 20
END_MLT = 2
MLT_COLS = np.concatenate((np.arange(START_MLT, 24), np.arange(END_MLT)))
cat = cat[(cat.MLT_OPQ > START_MLT) | (cat.MLT_OPQ < END_MLT)]
print(f"Number of duskside events {cat.shape[0]}")

### Load the MLT-lon normalization files.
with open('/home/mike/research/ac6_curtains/data/norm/ac6_MLT_lon_bins_same_loc.csv') as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = np.array(list(map(float, bins[key])))
with open('/home/mike/research/ac6_curtains/data/norm/ac6_MLT_lon_norm_same_loc.csv') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    norm = 10*np.array(list(reader)).astype(float) # Convert to number of samples.

# Downsample the MLT into the correct MLT and lon bins. Norm shape is nMLT, nLon
norm = pd.DataFrame(norm, index=bins['MLT_OPQ'][:-1], columns=bins['lon'][:-1])

# Resample to every n longitude samples (nominal lon bin width is 10 degrees)
n = 4
if n > 1:
    norm = norm.groupby(np.arange(len(norm.columns))//n, axis=1).sum()
    # Rename the columns with a column mapper dict
    column_mapper = {i:j for i, j in zip(
                                    np.arange(len(norm.columns)), 
                                    -180+10*n*np.arange(len(norm.columns)) 
                                    ) }
    norm.rename(columns=column_mapper, inplace=True)

# Now sum over the MLT ranges.
norm_mlt_sum = norm.loc[MLT_COLS].sum(axis=0)
scaling_factors = np.max(norm_mlt_sum)/norm_mlt_sum

# Bin the curtain detections.
curtain_bins = np.append(norm.columns, 
                norm.columns[-1] + (norm.columns[1]-norm.columns[0]))
binned_curtains, _ = np.histogram(cat.lon, bins=curtain_bins)
binned_curtains[binned_curtains <= 1] = 0
scaled_curtains = binned_curtains*scaling_factors


_, ax = plt.subplots(3, sharex=True, figsize=(8, 8))
ax[0].step(norm.columns, binned_curtains, where='post')
ax[0].set_title(f'Curtain distribution in longitude\n{START_MLT} < MLT < 24')
ax[0].set_ylabel('Unnormalized\nnumber of curtains')

ax[1].step(norm.columns, scaled_curtains, where='post')
ax[1].set_ylabel('Normalized\nnumber of curtains')

ax[-1].step(norm.columns, norm_mlt_sum/1E5, where='post')
ax[-1].set_ylabel(r'Number of 10 Hz samples x $10^5$')
ax[-1].set_xlabel('Longitude [degrees]')

for a in ax:
    a.axvline(-60, c='k')
    a.axvline(30, c='k')
plt.show()