import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats

import dirs

"""
Script to calculate the curtain width statistics.
"""
similar_thresh = 0.25
include_microbursts = False

curtain_catalog_name='AC6_curtains_baseline_method_sorted_v0.csv'
curtain_catalog_path = os.path.join(dirs.CATALOG_DIR, curtain_catalog_name)
curtain_cat = pd.read_csv(curtain_catalog_path)

if include_microbursts:
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
max_bin_km = 56
size_bins = np.arange(0, max_bin_km, size_bin_width_km)
# Filter similar curtain widths
idx = (
    (curtain_cat['width_B']/curtain_cat['width_A'] < 1+similar_thresh) &
    (curtain_cat['width_A']/curtain_cat['width_B'] < 1+similar_thresh)
      )
curtain_cat = curtain_cat[idx]

def mc_err(mu, n_iter=1000):
    """ 
    Calculate the Monte Carlo standard error of n, an array of 
    counts, assuming poisson statistics 
    """
    err = np.nan*np.zeros_like(mu)

    for i, mu_i in enumerate(mu):
        dist = scipy.stats.poisson(mu_i).rvs(size=n_iter)
        #q = dist.std()
        #err[i] = (q.iloc[1] - q.iloc[0])/2
        err[i] = np.std(dist)
    return err

# Histogram and estimate errors for curtains
curtain_hist, _ = np.histogram(7.5*curtain_cat['width_B'], 
                        bins=size_bins)
curtain_mc_std = mc_err(curtain_hist)
curtain_fraction_mid = curtain_hist/(curtain_cat.shape[0]*size_bin_width_km)
curtain_fraction_upper = (curtain_hist+curtain_mc_std)/\
      (size_bin_width_km*(sum(curtain_hist)+sum(curtain_mc_std)))
curtain_fraction_lower = (curtain_hist-curtain_mc_std)/\
      (size_bin_width_km*(sum(curtain_hist)-sum(curtain_mc_std)))
curtain_fraction_err = np.abs(curtain_fraction_upper-curtain_fraction_lower)

# Histogram and estimate errors for microbursts
# all_microbursts, _ = np.histogram(microburst_cat['Dist_In_Track'], 
#                               density=True, bins=size_bins)
if include_microbursts:
    microburst_hist, _ = np.histogram(coincident_microburst_cat['Dist_In_Track'], 
                                bins=size_bins)
    microburst_mc_std = mc_err(microburst_hist)
    microburst_fraction_mid = microburst_hist/(coincident_microburst_cat.shape[0]*size_bin_width_km)
    microburst_fraction_upper = (microburst_hist+microburst_mc_std)/\
        (size_bin_width_km*(sum(microburst_hist)+sum(microburst_mc_std)))
    microburst_fraction_lower = (microburst_hist-microburst_mc_std)/\
        (size_bin_width_km*(sum(microburst_hist)-sum(microburst_mc_std)))
    microburst_fraction_err = np.abs(microburst_fraction_upper-microburst_fraction_lower)

# microburst_fraction = coincident_microbursts/all_microbursts
# microburst_fraction /= size_bin_width_km*np.sum(microburst_fraction)

max_size=21
print(f'{100*(sum(curtain_cat["width_A"] < max_size/7.5))/len(curtain_cat["width_A"])}% '
      f'of curtains are less than {max_size} km wide.')

fig, ax = plt.subplots()
ax.step(size_bins, np.append(curtain_fraction_mid, np.nan), where='post', c='k', 
      label='curtain', lw=4)
ax.errorbar(size_bins+size_bin_width_km/2, np.append(curtain_fraction_mid, np.nan), 
            yerr=np.append(curtain_fraction_err, np.nan),
            ls='', color='k', capsize=3, lw=2)
if include_microbursts:
    ax.step(size_bins, np.append(microburst_fraction_mid, np.nan), c='r', 
        where='post', label='microburst', lw=2)
    ax.errorbar(size_bins+size_bin_width_km/2, np.append(microburst_fraction_mid, np.nan), 
                yerr=np.append(microburst_fraction_err, np.nan), ls='', color='r', capsize=3, lw=1)
    ax.set(title='Distribution of > 30 keV curtain and microburst sizes',
            ylabel='Probability density', xlim=(0, size_bins[-1]), 
            xlabel='In-track size [km]')
    ax.legend()
else:
    ax.set(title='Distribution of > 30 keV curtains',
            ylabel='Probability density', xlim=(0, size_bins[-1]), 
            xlabel='In-track size [km]')

# fig, ax = plt.subplots(3, sharex=True)
# bins = np.arange(0, 5, 0.25)
# ax[0].hist(cat['width_A'], bins=bins)
# ax[1].hist(cat['width_B'], bins=bins)
# ax[2].hist(cat_similar_width['width_B'], bins=bins)

# ax[0].set(ylabel='AC6A', title='AC6 Curtain Width Distributions')
# ax[1].set(ylabel='AC6B', xlim=(0, None))
# ax[-1].set(ylabel=f'Similar widths\n(within {100*thresh} %)', xlabel='Curtain FWHM [s]')
plt.show()