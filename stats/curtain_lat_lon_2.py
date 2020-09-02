"""
This script plots the normalized lat-lon curtain distribution,
as well as the marginalized lat-lon distributions on the side.
"""
import pathlib
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import uncertainties.unumpy as unp

import dirs

cmap='Blues'
projection = ccrs.PlateCarree()
L_levels = [4,10]

### LOAD THE CATALOG ###
# Load the curtain catalog.
cat_path = pathlib.Path(dirs.CATALOG_DIR, 
        'AC6_curtains_baseline_method_sorted_v0.csv')
cat = pd.read_csv(cat_path)

### LOAD THE THREE NORMALIZATION FILES ###
# The Lon and lat normalization files.
lon_norm_path = pathlib.Path(dirs.NORM_DIR, 'ac6_lon_norm.csv')
lat_norm_path = pathlib.Path(dirs.NORM_DIR, 'ac6_lat_norm.csv')

# The full distribution normalization files.
lat_lon_norm_bin_path = pathlib.Path(dirs.NORM_DIR, 
        'ac6_lat_lon_bins.csv')
lat_lon_norm_path = pathlib.Path(dirs.NORM_DIR, 
        'ac6_lat_lon_norm.csv')

# Load the full normalization files.
with open(lat_lon_norm_bin_path) as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = list(map(float, bins[key]))
lat_lon_norm = pd.read_csv(lat_lon_norm_path, skiprows=1, names=bins['lon'])
lat_lon_norm.index = bins['lat'][:-1]

lon_norm = pd.read_csv(lon_norm_path, index_col=0)
lat_norm = pd.read_csv(lat_norm_path, index_col=0)
np_lon_bins_missing_bin = lon_norm.index.to_numpy()
lon_bins = np.append(np_lon_bins_missing_bin, 
                    2*np_lon_bins_missing_bin[-1]-np_lon_bins_missing_bin[-2])
np_lat_bins_missing_bin = lat_norm.index.to_numpy()
lat_bins = np.append(np_lat_bins_missing_bin, 
                    2*np_lat_bins_missing_bin[-1]-np_lat_bins_missing_bin[-2])

# # Load the L shell distribution to overlay on the map
L_lons = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_lons.npy')
L_lats = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_lats.npy')
L = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_l.npy')

# # Calculate the curtain histogram and the normalized curtain histogram.
curtain_hist, _, _ = np.histogram2d(x=cat.loc[:, 'lon'], y=cat.loc[:, 'lat'], 
                bins=[bins['lon'], bins['lat']])
curtain_hist = curtain_hist.T
curtain_hist_norm = curtain_hist*(np.nanmax(lat_lon_norm.values)/lat_lon_norm.values[:, :-1])
curtain_hist_norm[np.where(np.isinf(curtain_hist_norm))] = 0
curtain_hist_norm[np.where(np.isnan(curtain_hist_norm))] = 0

### MAKE PLOTS ###
fig = plt.figure(constrained_layout=True, figsize=[8.5, 4.8])
size=4
gs = fig.add_gridspec(size,size)
map_ax = fig.add_subplot(gs[:size-2, :size-2], projection=projection)
lon_ax = fig.add_subplot(gs[-2:, :size-2], sharex=map_ax)
lat_ax = fig.add_subplot(gs[:size-2, -2:], sharey=map_ax)
norm_ax = fig.add_subplot(gs[size-2:, -2:], projection=projection)
map_ax.xaxis.set_visible(False)
lat_ax.yaxis.set_visible(False)


### MAKE THE MAP PLOT ###
map_ax.coastlines(zorder=10)
curtain_hist_plt = map_ax.pcolormesh(bins['lon'], bins['lat'], 
                                    curtain_hist_norm,
                                    cmap=cmap)
map_ax.contour(L_lons, L_lats, L, levels=L_levels, colors='k', 
                linestyles='dotted', linewidths=2)
# plt.colorbar(curtain_hist_plt, label='Number of curtains', ax=ax[0])
# map_ax.set(ylabel='latitude') #title='Latitude-Longitude Curtain Distribution')

### MAKE THE LON DISTRIBUTION PLOT ###
lon_hist, _ = np.histogram(cat['lon'].to_numpy(), bins=lon_bins)
lon_hist = unp.uarray(lon_hist, np.sqrt(lon_hist))
normalized_lon_hist = lon_hist*(lon_norm['Seconds'].max()/lon_norm['Seconds'])
lon_ax.step(lon_bins, 
            np.append(unp.nominal_values(normalized_lon_hist), np.nan), where='post', c='k')
lon_ax.errorbar(lon_bins[:-1]+(lon_bins[1]-lon_bins[0])/2,
                unp.nominal_values(normalized_lon_hist),
                yerr=2*unp.std_devs(normalized_lon_hist), ls='None', c='k')
# lon_ax.set_title('Longitude marginal distribution')
lon_ax.set_ylim(0, None)

### MAKE THE LAT DISTRIBUTION PLOT ###
lat_hist, _ = np.histogram(cat['lat'].to_numpy(), bins=lat_bins)
lat_hist = unp.uarray(lat_hist, np.sqrt(lat_hist))
normalized_lat_hist = lat_hist*(lat_norm['Seconds'].max()/lat_norm['Seconds'])
lat_ax.step(unp.nominal_values(normalized_lat_hist), lat_bins[:-1], where='post', c='k')
lat_ax.errorbar(unp.nominal_values(normalized_lat_hist), 
                lat_bins[:-1]-(lat_bins[1]-lat_bins[0])/2,
                xerr=2*unp.std_devs(normalized_lat_hist), ls='None', c='k')
# lat_ax.set_title('Latitude marginal distribution')

### Make the full lat-lon normalization plot.
norm_ax.coastlines(zorder=10)
norm_ax.pcolormesh(bins['lon'], bins['lat'], lat_lon_norm,
                                cmap=cmap)

map_ax.text(0.9, 0, f'Normalized', va='bottom', ha='right', 
            transform=map_ax.transAxes, fontsize=15, color='purple', weight='bold')
norm_ax.text(0.9, 0, f'Normalization', va='bottom', ha='right', 
            transform=norm_ax.transAxes, fontsize=15, color='purple', weight='bold')
plt.show()