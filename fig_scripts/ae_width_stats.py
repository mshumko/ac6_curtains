# This script makes plots of the AE and width 
# distributions of curtains.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import string

import dirs

# Load the curtain catalog
cat_name = 'AC6_curtains_baseline_method_sorted_v0.csv'
cat_path = os.path.join(dirs.CATALOG_DIR, cat_name)
cat = pd.read_csv(cat_path)
cat['dateTime'] = pd.to_datetime(cat['dateTime'])

# Load the AE index for all of the years that curtains 
# were observed
ae_dir = os.path.join(dirs.BASE_DIR, 'data', 'ae')
years = sorted(set(cat['dateTime'].dt.year))
ae = pd.DataFrame(data=np.zeros((0, 1)), columns=['AE'])

for year in years:
    ae_path = os.path.join(ae_dir, f'{year}_ae.txt')
    year_ae = pd.read_csv(ae_path, delim_whitespace=True, 
                    usecols=[0, 1, 3], skiprows=14, 
                    parse_dates=[['DATE', 'TIME']])
    year_ae.index=year_ae.DATE_TIME
    del year_ae['DATE_TIME']
    ae = ae.append(year_ae)

ae_bin_width = 100
ae_bins = np.arange(0, 1200, ae_bin_width)
H_AE, _ = np.histogram(ae['AE'], density=True, bins=ae_bins)
H_c, _ = np.histogram(cat['AE'], density=True, bins=ae_bins)

# Scale the curtain distribution by the total AE time. In other
# words this normalized distribution is the distribution of
# curtains assuming any AE index is equally likeliy to occur.
H_scaled = H_c*(np.max(H_AE)/H_AE)
H_scaled = H_scaled/(np.sum(H_scaled)*ae_bin_width)

# Curtain width histogram
size_bin_width_km = 2
size_bins = np.arange(0, 35, size_bin_width_km)
H_size, _ = np.histogram(7.5*cat['width_B'], density=True, bins=size_bins)


fig, ax = plt.subplots(1, 2, figsize=(10,4))
# ax[0].hist(7.5*cat['width_B'], bins=size_bins)
ax[0].step(size_bins[:-1], H_size, where='post', c='k')
ax[1].step(ae_bins[:-1], H_AE, where='post', label='2014-2017 AE', c='k')
ax[1].step(ae_bins[:-1], H_c, where='post', label='Curtain AE', c='r', linewidth=3)

ax[0].set(title='Distribution of curtain widths',
            ylabel='Probability density', xlim=(0, size_bins[-2]), 
            xlabel='In-track width [km]')
ax[1].set(title='Distribution of AE vs. curtain AE',
          xlabel='AE [nT]', xlim=(0, ae_bins[-2]))
ax[1].legend()

for i, a in enumerate(ax):
    a.text(-0.1, 1, f'({string.ascii_letters[i]})', va='bottom',
            transform=a.transAxes, fontsize=20)

plt.tight_layout()
plt.show()