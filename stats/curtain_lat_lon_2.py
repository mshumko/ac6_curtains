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

cmap='Greens'
projection = ccrs.PlateCarree()
L_levels = [4,10]

### LOAD THE CATALOG ###
# Load the curtain catalog.
cat_path = pathlib.Path(dirs.CATALOG_DIR, 
        'AC6_curtains_baseline_method_sorted_v0.csv')
cat = pd.read_csv(cat_path)

# Filter by longitudes outside of the BLC and the SAA
cat_no_blc = cat[(cat.lon > 30) | (cat.lon < -60)]

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
fig = plt.figure(figsize=[9, 5.1])
gs = fig.add_gridspec(2, 2, width_ratios=(4, 2), height_ratios=(4, 2))
map_ax = fig.add_subplot(gs[0, 0], projection=projection)
lon_ax = fig.add_subplot(gs[1, 0])
lat_ax = fig.add_subplot(gs[0, 1])
norm_ax = fig.add_subplot(gs[1,1], projection=projection)
map_ax.xaxis.set_visible(False)
lat_ax.yaxis.set_label_position("right")
lat_ax.yaxis.tick_right()
# lat_ax.set_xlabel('# of curtains')

plt.suptitle('AC6 curtain latitude-longitude distribution', fontsize=15)


### MAKE THE MAP PLOT ###
map_ax.coastlines(zorder=10)
# map_ax.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
curtain_hist_plt = map_ax.pcolormesh(bins['lon'], bins['lat'], 
                                    curtain_hist_norm,
                                    cmap=cmap)
map_ax.contour(L_lons, L_lats, L, levels=L_levels, colors='k', 
                linestyles='dotted', linewidths=2)

### MAKE THE LON DISTRIBUTION PLOT ###
lon_hist, _ = np.histogram(cat['lon'].to_numpy(), bins=lon_bins)
lon_hist = unp.uarray(lon_hist, np.sqrt(lon_hist))
normalized_lon_hist = lon_hist*(lon_norm['Seconds'].max()/lon_norm['Seconds'])
lon_ax.step(lon_bins, 
            np.append(unp.nominal_values(normalized_lon_hist), np.nan), where='post', c='k')
lon_ax.errorbar(lon_bins[:-1]+(lon_bins[1]-lon_bins[0])/2,
                unp.nominal_values(normalized_lon_hist),
                yerr=unp.std_devs(normalized_lon_hist), ls='None', c='k')
lon_ax.fill_between(lon_bins, 0, np.append(unp.nominal_values(normalized_lon_hist), 0), 
                color='green', step='post', alpha=0.7)
# lon_ax.set_title('Longitude marginal distribution')
lon_ax.set_ylim(0, None)
lon_ax.set_xlim(-180, 180)
lon_ax.set_xlabel('Longitude')
# lon_ax.set_ylabel('# of curtains')

### MAKE THE LAT DISTRIBUTION PLOT ###
lat_hist, _ = np.histogram(cat['lat'].to_numpy(), bins=lat_bins)
lat_hist = unp.uarray(lat_hist, np.sqrt(lat_hist))
normalized_lat_hist = lat_hist*(lat_norm['Seconds'].max()/lat_norm['Seconds'])

lat_hist_no_blc, _ = np.histogram(cat_no_blc['lat'].to_numpy(), bins=lat_bins)
lat_hist_no_blc = unp.uarray(lat_hist_no_blc, np.sqrt(lat_hist_no_blc))
normalized_lat_hist_no_blc = lat_hist_no_blc*(lat_norm['Seconds'].max()/lat_norm['Seconds'])

# Plot all curtains in the marginalized lat plot
lat_ax.step(np.append(unp.nominal_values(normalized_lat_hist), 0), lat_bins, where='pre',
             c='k', label='All')
lat_ax.errorbar(unp.nominal_values(normalized_lat_hist), 
                lat_bins[:-1]+(lat_bins[1]-lat_bins[0])/2,
                xerr=unp.std_devs(normalized_lat_hist), 
                ls='None', c='k')
lat_ax.fill_betweenx(lat_bins, 0, np.append(unp.nominal_values(normalized_lat_hist), 0), 
                color='green', step='post', alpha=0.7)
lat_ax.set_ylim(-90, 90)
lat_ax.set_xlim(0, None)
lat_ax.set_ylabel('Latitude')

### Make the full lat-lon normalization plot.
norm_ax.coastlines(zorder=5)
norm_ax.pcolormesh(bins['lon'], bins['lat'], lat_lon_norm,
                                cmap=cmap)

fontsize=13
# Add panel titles
map_ax.text(1, 0, f'Normalized', va='bottom', ha='right', 
            transform=map_ax.transAxes, fontsize=fontsize, color='red', weight='bold')
norm_ax.text(0, 0.5, f'Normalization', va='bottom', ha='left', 
            transform=norm_ax.transAxes, fontsize=fontsize, color='red', weight='bold', zorder=10)
lon_ax.text(0, 0, f'Number of curtains\nlatitude marginalized out', va='bottom', ha='left', 
            transform=lon_ax.transAxes, fontsize=fontsize, color='red', weight='bold')
lat_ax.text(0, 0.5, f'Number of curtains\nlongitude marginalized\nout', va='center', ha='left', 
            transform=lat_ax.transAxes, fontsize=fontsize, color='red', weight='bold')

# Add panel labels
map_ax.text(0, 1, '(a)', va='top', ha='left', 
            fontsize=fontsize, color='red', weight='bold', transform=map_ax.transAxes)
lat_ax.text(0, 1, '(b)', va='top', ha='left', 
            fontsize=fontsize, color='red', weight='bold', transform=lat_ax.transAxes)
lon_ax.text(0, 1, '(c)', va='top', ha='left', 
            fontsize=fontsize, color='red', weight='bold', transform=lon_ax.transAxes)
norm_ax.text(0, 1, '(d)', va='top', ha='left', 
            fontsize=fontsize, color='red', weight='bold', transform=norm_ax.transAxes)
gs.tight_layout(fig, rect=(0, 0, 1, 0.95))
plt.show()