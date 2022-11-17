"""
This scipt plots the original and the normalized lat-lon 
curtain distributions.
"""
import pathlib
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import dirs

cmap='Blues'
projection = ccrs.PlateCarree()
L_levels = [4,10]

# Load the curtain catalog.
cat_path = pathlib.Path(dirs.CATALOG_DIR, 
        'AC6_curtains_baseline_method_sorted_v0.csv')
cat = pd.read_csv(cat_path)

# Load the lat-lon normalization files.
norm_bin_path = pathlib.Path(dirs.NORM_DIR, 
        'ac6_lat_lon_bins.csv')
norm_path = pathlib.Path(dirs.NORM_DIR, 
        'ac6_lat_lon_norm.csv')
# Load the bins
with open(norm_bin_path) as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = list(map(float, bins[key]))
# Load the normalization file.
norm = pd.read_csv(norm_path, skiprows=1, names=bins['lon'])
norm.index = bins['lat'][:-1]

# Load the L shell distribution to overlay on the map
L_lons = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_lons.npy')
L_lats = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_lats.npy')
L = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_l.npy')

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

# Calculate the curtain histogram and the normalized curtain histogram.
curtain_hist, _, _ = np.histogram2d(x=cat.loc[:, 'lon'], y=cat.loc[:, 'lat'], 
                bins=[bins['lon'], bins['lat']])
curtain_hist = curtain_hist.T
curtain_hist_norm = curtain_hist*(np.nanmax(norm.values)/norm.values[:, :-1])
curtain_hist_norm[np.where(np.isinf(curtain_hist_norm))] = 0
curtain_hist_norm[np.where(np.isnan(curtain_hist_norm))] = 0

### PLOTS ###
fig = plt.figure(figsize=(6, 8))
# fig, ax = plt.subplots(3, figsize=(5, 8))#, 
                        # sharex=True, sharey=True)
n_panels = 3
ax = [fig.add_subplot(n_panels, 1, i+1, projection=projection) 
    for i in range(n_panels)]
[a.coastlines(zorder=10) for a in ax]

curtain_hist_plt = ax[0].pcolormesh(bins['lon'], bins['lat'], 
                                    curtain_hist,
                                    cmap=cmap)
plt.colorbar(curtain_hist_plt, label='Number of curtains', ax=ax[0])
ax[0].set(ylabel='latitude', title='Latitude-Longitude Curtain Distribution')

curtain_hist_norm_plt = ax[1].pcolormesh(bins['lon'], bins['lat'], 
                                        curtain_hist_norm, 
                                        cmap=cmap)
plt.colorbar(curtain_hist_norm_plt, label='Normalized\nNumber of curtains', ax=ax[1])
ax[1].set(ylabel='latitude')

norm_hist = ax[-1].pcolormesh(bins['lon'], bins['lat'], norm,
                            cmap=cmap)
plt.colorbar(norm_hist, label='Number of 10 Hz seconds', ax=ax[-1])
ax[-1].set(xlabel='longitude', ylabel='latitude')

# # Overlay BLC boundary
# mirror_point_df = pd.read_csv(pathlib.Path(dirs.BASE_DIR, 'data', 'lat_lon_mirror_alt.csv'),
#                                     index_col=0, header=0)
# lons = np.array(mirror_point_df.columns, dtype=float)
# lats = mirror_point_df.index.values.astype(float)   

subplot_titles=['Unnormalized', 'Normalized', 'Normalization']

for i, a in enumerate(ax):
#     a.contour(lons, lats, mirror_point_df.values,
#             levels=[0, 100], colors=['r', 'r'], linestyles=['dashed', 'solid'], alpha=0.4)
    # Overlay rad belt L contours  
    a.contour(L_lons, L_lats, L, levels=L_levels, colors='k', linestyles='dotted', linewidths=2)
    # a.set_aspect('equal', 'datalim')
    a.set_xlim(-180, 180)
    a.set_ylim(-90, 90)
    a.text(0, 1, f'({string.ascii_letters[i]})', va='top', ha='left', 
            transform=a.transAxes, fontsize=15, color='purple', weight='bold')
    a.text(1, 0, f'{subplot_titles[i]}', va='bottom', ha='right', 
            transform=a.transAxes, fontsize=15, color='purple', weight='bold')
    a.set_ylabel(subplot_titles[i])

plt.tight_layout()
plt.show()