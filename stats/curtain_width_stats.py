import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

import dirs

"""
Script to calculate the curtain width statistics.
"""
similar_thresh = 0.25

curtain_catalog_name='AC6_curtains_baseline_method_sorted_v0.txt'
curtain_catalog_path = os.path.join(dirs.CATALOG_DIR, curtain_catalog_name)
curtain_cat = pd.read_csv(curtain_catalog_path)

microburst_name ='AC6_coincident_microbursts_v8.txt'
coincident_microburst_name='AC6_coincident_microbursts_sorted_v6.txt'
microburst_dir = ('/home/mike/research/ac6_microburst_scale_sizes/'
                  'data/coincident_microbursts_catalogues')

microburst_catalog_path = os.path.join(microburst_dir, microburst_name)
microburst_cat = pd.read_csv(microburst_catalog_path)

coincident_microburst_catalog_path = os.path.join(microburst_dir, coincident_microburst_name)
coincident_microburst_cat = pd.read_csv(coincident_microburst_catalog_path)

# Calculate the curtain probability density
size_bin_width_km = 5
max_bin = 56
size_bins = np.arange(0, max_bin, size_bin_width_km)
# Filter similar curtain widths
idx = (
      (curtain_cat['width_B']/curtain_cat['width_A'] < 1+similar_thresh) &
      (curtain_cat['width_A']/curtain_cat['width_B'] < 1+similar_thresh)
      )
curtain_cat = curtain_cat[idx]
# Histogram this
H_size, _ = np.histogram(7.5*curtain_cat['width_B'], 
                        density=True, bins=size_bins)

# Calculate the microburst probability density.
all_microbursts, _ = np.histogram(microburst_cat['Dist_In_Track'], 
                              density=True, bins=size_bins)
coincident_microbursts, _ = np.histogram(coincident_microburst_cat['Dist_In_Track'], 
                              density=True, bins=size_bins)
microburst_fraction = coincident_microbursts/all_microbursts
microburst_fraction /= size_bin_width_km*np.sum(microburst_fraction)

max_size=21
print(f'{100*(sum(curtain_cat["width_A"] < max_size/7.5))/len(curtain_cat["width_A"])}% '
      f'of curtains are less than {max_size} km wide.')

fig, ax = plt.subplots()
ax.step(size_bins, np.append(H_size, np.nan), where='post', c='k', 
      label='curtain', lw=4)
ax.step(size_bins, np.append(microburst_fraction, np.nan), c='r', 
      where='post', label='microburst', lw=2)

ax.set(title='Distribution of > 30 keV curtain and microburst sizes',
            ylabel='Probability density', xlim=(0, size_bins[-1]), 
            xlabel='In-track size [km]')
ax.legend()

# fig, ax = plt.subplots(3, sharex=True)
# bins = np.arange(0, 5, 0.25)
# ax[0].hist(cat['width_A'], bins=bins)
# ax[1].hist(cat['width_B'], bins=bins)
# ax[2].hist(cat_similar_width['width_B'], bins=bins)

# ax[0].set(ylabel='AC6A', title='AC6 Curtain Width Distributions')
# ax[1].set(ylabel='AC6B', xlim=(0, None))
# ax[-1].set(ylabel=f'Similar widths\n(within {100*thresh} %)', xlabel='Curtain FWHM [s]')
plt.show()