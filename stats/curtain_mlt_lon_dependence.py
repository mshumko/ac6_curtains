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
START_MLT = 21
cat = cat[(cat.MLT_OPQ > START_MLT)]
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

norm = pd.DataFrame(norm, index=bins['MLT_OPQ'][:-1], columns=bins['lon'][:-1])
if True:
    # Resample to every n MLT
    n=2
    norm = norm.groupby(norm.index//n).sum()
    norm = norm.set_index(np.arange(0, 24, n))

# Rebin the normalization. Norm shape is nMLT, nLon
idx = np.where(norm.index >= START_MLT)[0][0]
# Sum over the MLTs
norm_mlt = np.sum(norm.loc[idx:, :], axis=0)
scaling_factors = np.max(norm_mlt)/norm_mlt
binned_curtains, _ = np.histogram(cat.lon, bins=bins['lon'])

_, ax = plt.subplots(3, sharex=True, figsize=(8, 8))
ax[0].hist(cat.lon, bins=bins['lon'])
ax[0].set_title(f'Curtain distribution in longitude | {START_MLT} < MLT < 24')
ax[0].set_ylabel('Unnormalized\nnumber of curtains')

ax[1].step(bins['lon'][:-1], binned_curtains*scaling_factors, where='post')
ax[1].set_ylabel('Normalized\nnumber of curtains')

ax[-1].step(bins['lon'][:-1], norm_mlt/1E5, where='post')
ax[-1].set_ylabel(r'Normalization x $10^5$')
ax[-1].set_xlabel('Longitude [degrees]')
plt.show()