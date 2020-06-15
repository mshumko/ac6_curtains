"""
This scipt plots the lat-lon distribution of curtains and normalizes 
it.
"""

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
        'ac6_lat_lon_bins.csv')
norm_path = pathlib.Path(dirs.NORM_DIR, 
        'ac6_lat_lon_norm.csv')

# Load the L-MLT normalization files.
with open(norm_bin_path) as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = list(map(float, bins[key]))

norm = pd.read_csv(norm_path, skiprows=1, names=bins['lon'])
norm.index = bins['lat'][:-1]

def rebin(df, deg, coord):
    # Group by latitude into deg degree chucks
    assert coord in ['lat', 'lon'], 'Coodinates much be lat or lon!'

    if coord == 'lon':
        df = df.T

    df = df.groupby(df.index//deg).sum(axis=0)
    
    if coord == 'lon':
        df.index = np.arange(-180, 181, deg)
        df = df.T
    else:
        df.index = np.arange(-90, 91, deg)
    return df

# norm = rebin(norm, 10, 'lon')

curtain_hist, _, _ = np.histogram2d(x=cat.loc[:, 'lon'], y=cat.loc[:, 'lat'], 
                bins=[bins['lon'], bins['lat']])
                # bins=[norm.columns, norm.index])
curtain_hist = curtain_hist.T
curtain_hist_norm = curtain_hist*(np.nanmax(norm.values)/norm.values[:, :-1])
curtain_hist_norm[np.where(np.isinf(curtain_hist_norm))] = np.nan

fig, ax = plt.subplots(3, figsize=(8, 10), 
                        sharex=True, sharey=True)
curtain_hist_plt = ax[0].pcolormesh(norm.columns[:-1], norm.index, curtain_hist)
plt.colorbar(curtain_hist_plt, label='Number of curtains', ax=ax[0])
ax[0].set(ylabel='lat', title='Lat-Lon Curtain Distribution')

curtain_hist_norm_plt = ax[1].pcolormesh(norm.columns[:-1], norm.index, curtain_hist_norm)
plt.colorbar(curtain_hist_norm_plt, label='Normalized\nNumber of curtains', ax=ax[1])
ax[1].set(ylabel='lat', title='Normalized Lat-Lon Curtain Distribution')

norm_hist = ax[-1].pcolormesh(norm.columns, norm.index, norm)
plt.colorbar(norm_hist, label='Number of 10 Hz seconds', ax=ax[-1])
ax[-1].set(xlabel='lon', ylabel='lat', title='Normalization')

# Overlay BLC boundary
mirror_point_df = pd.read_csv(pathlib.Path(dirs.BASE_DIR, 'data', 'lat_lon_mirror_alt.csv'),
                                    index_col=0, header=0)
lons = np.array(mirror_point_df.columns, dtype=float)
lats = mirror_point_df.index.values.astype(float)   

# # Overlay rad belt L contours
# L_lons = np.load('/home/mike/research/mission_tools'
#                 '/misc/irbem_l_lons.npy')
# L_lats = np.load('/home/mike/research/mission_tools'
#                 '/misc/irbem_l_lats.npy')
# L = np.load('/home/mike/research/mission_tools'
#                 '/misc/irbem_l_l.npy')
# levels = [4,8]

for a in ax:
    a.contour(lons, lats, mirror_point_df.values,
            levels=[0, 100], colors=['r', 'r'], linestyles=['dashed', 'solid'], alpha=0.4)
    # a.contour(L_lons, L_lats, L, levels=levels, colors='k', linestyles='dotted')

plt.tight_layout()
plt.show()