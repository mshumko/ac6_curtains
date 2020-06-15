import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np

import dirs

cat_path = pathlib.Path(dirs.CATALOG_DIR, 
        'AC6_curtains_baseline_method_sorted_v0.csv')
cat = pd.read_csv(cat_path)

# Load normalization file
norm_bin_path = pathlib.Path(dirs.NORM_DIR, 
        'ac6_L_lon_bins.csv')
norm_path = pathlib.Path(dirs.NORM_DIR, 
        'ac6_L_lon_norm.csv')

# Load the L-MLT normalization files.
with open(norm_bin_path) as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = list(map(float, bins[key]))

norm = pd.read_csv(norm_path, skiprows=1, names=bins['lon'])
norm.index = bins['Lm_OPQ'][:-1]

norm = norm.loc[4:, :]

# Group by latitude in 20 degree chucks
norm2 = norm.T
norm2 = norm2.groupby(norm2.index//20).sum(axis=0)
norm2.index = np.arange(-180, 181, 20)
norm2 = norm2.T


# cat.loc[:, 'Lm_OPQ'] = cat.loc[:, 'Lm_OPQ']*np.sign(cat.loc[:, 'lat'])

L_bins = np.arange(4, 9)
# lat_bins = np.arange(-90, 91, 10)
lon_bins = np.arange(-180, 181, 20)

fig, ax = plt.subplots(2, figsize=(8, 5), sharex=True)
h = ax[0].hist2d(x=cat.loc[:, 'lon'], y=cat.loc[:, 'Lm_OPQ'], bins=[lon_bins, L_bins])
# h = ax.hist2d(x=cat.loc[:, 'lon'], y=cat.loc[:, 'lat'], bins=[lon_bins, lat_bins])

plt.colorbar(h[-1], label='Number of curtains', ax=ax[0])
ax[0].set(xlabel='lon', ylabel='Lm_OPQ', title='L-lon curtain distribution')

h2 = ax[1].pcolormesh(norm2.columns, norm2.index, norm2)
plt.colorbar(h2, label='Number of 10 Hz seconds', ax=ax[1])
ax[1].set(xlabel='lon', ylabel='Lm_OPQ')

# # Plot the norm
# lon_norm = norm.sum(axis=0)
# ax[1].plot(lon_norm.index, lon_norm)

plt.show()