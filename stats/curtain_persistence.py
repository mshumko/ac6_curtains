# This script plots how many curtains were observed as a function of 
# AC6 in-track lag.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import uncertainties.unumpy as unp

import dirs

# Load the curtain catalog
cat_name = 'AC6_curtains_baseline_method_sorted_v0.txt'
cat_path = pathlib.Path(dirs.CATALOG_DIR, cat_name)
cat = pd.read_csv(cat_path)

# Load the 10 Hz normalization
hist = pd.read_csv(pathlib.Path(dirs.NORM_DIR, 'ac6_lag_dist_10hz.csv'), 
                    index_col=0)
hist['hours'] = hist / 3600 / 24 # Normalize to hours

# Sum over n bins (n kilometers unless lag_bin_s has 
# a non-default step)
bin_width_km = 5
old_hist_index = hist.index.copy()
hist = hist.groupby(hist.index//bin_width_km).sum()
hist.index = old_hist_index[::bin_width_km]

# Bin the curtain detections
curtain_H, _ = np.histogram(cat.Lag_In_Track, bins=hist.index)
curtain_H = unp.uarray(curtain_H, np.sqrt(curtain_H))  


# Normalize the curtain distribution 
curtain_H_norm = curtain_H*(max(hist.hours)/hist.hours[:-1])

# Plot the stuff
fig, ax = plt.subplots(3, sharex=True, figsize=(6, 6), constrained_layout=True)
# ax[0].hist(cat.Lag_In_Track, histtype='step', color='k', lw=2)
ax[0].step(hist.index[:-1], unp.nominal_values(curtain_H), where='post', c='k')
ax[0].errorbar(hist.index[:-1]+bin_width_km/2, unp.nominal_values(curtain_H), 
            yerr=unp.std_devs(curtain_H), ls='', c='k')
# ax[1].step(hist.index[:-1], curtain_H_norm, where='post')

ax[1].step(hist.index[:-1], unp.nominal_values(curtain_H_norm), where='post', c='k')
ax[1].errorbar(hist.index[:-1]+bin_width_km/2, unp.nominal_values(curtain_H_norm), 
            yerr=unp.std_devs(curtain_H_norm), ls='', c='k')

ax[-1].step(hist.index, hist['hours'], where='post', c='k')
ax[0].set(ylabel='Number of curtains', title='AC6 Curtain Persistence')
ax[1].set(ylabel='Normalized\nnumber of curtains')
ax[-1].set(xlim=(0, 50), xlabel='In-track lag [s]', ylabel='Hours of 10 Hz data')

# plt.tight_layout()
plt.show()