# This script compares the distribution of AE when
# curtains were observed to the overall AE 
# distribution.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import dirs

# Load the curtain catalog
cat_name = 'AC6_curtains_baseline_method_sorted_v0.txt'
cat_path = os.path.join(dirs.CATALOG_DIR, cat_name)
curtain_cat = pd.read_csv(cat_path)
curtain_cat['dateTime'] = pd.to_datetime(curtain_cat['dateTime'])

# Load the microburst catalog
burst_name = 'AC6_microbursts_sorted_v6.txt'
burst_path = os.path.join(dirs.CATALOG_DIR, burst_name)
burst_cat = pd.read_csv(burst_path)

# Load the AE index for all of the years that curtains 
# were observed
ae_dir = os.path.join(dirs.BASE_DIR, 'data', 'ae')
years = sorted(set(curtain_cat['dateTime'].dt.year))
ae = pd.DataFrame(data=np.zeros((0, 1)), columns=['AE'])

for year in years:
    ae_path = os.path.join(ae_dir, f'{year}_ae.txt')
    year_ae = pd.read_csv(ae_path, delim_whitespace=True, 
                    usecols=[0, 1, 3], skiprows=14, 
                    parse_dates=[['DATE', 'TIME']])
    year_ae.index=year_ae.DATE_TIME
    del year_ae['DATE_TIME']
    ae = ae.append(year_ae)

# thresh = 20
bin_width = 100
bins = np.arange(0, 1200, bin_width)

H_AE, _ = np.histogram(ae['AE'], density=False, bins=bins)
H_AE_density, _ = np.histogram(ae['AE'], density=True, bins=bins)
H_c, _  = np.histogram(curtain_cat['AE'], density=False, bins=bins)
H_m, _  = np.histogram(burst_cat['AE'], density=False, bins=bins)
H_c_density, _  = np.histogram(curtain_cat['AE'], density=True, bins=bins)
H_m_density, _  = np.histogram(burst_cat['AE'], density=True, bins=bins)

def hist_density(dist, bin_width):
    """ Normalize a histogram to a PDF """
    dist = dist / (np.max(dist)*bin_width)
    return dist

# Scale the curtain distribution by the total AE time. In other
# words this normalized distribution is the distribution of
# curtains assuming any AE index is equally likeliy to occur.
H_c_norm = H_c*(np.max(H_AE)/H_AE)
H_c_norm_density = hist_density(H_c_norm, bin_width)

H_m_norm = H_m*(np.max(H_AE)/H_AE)
H_m_norm_density = hist_density(H_m_norm, bin_width)

#H_AE_density = hist_density(H_AE, bin_width)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(9,5))
ax[0].step(bins[:-1], H_AE_density, where='post', label='Index', 
        c='k', lw=2)
ax[0].step(bins[:-1], H_c_density, where='post', label=f'Curtains', 
        c='b', lw=2, linestyle=':')
ax[0].step(bins[:-1], H_m_density, where='post', label='Microbursts', 
        c='g', lw=2, linestyle='--')

# ax[1].step(bins[:-1], H_AE_density, where='post', label='Index', 
#         c='k', lw=2)
ax[1].step(bins[:-1], H_c_norm_density, where='post', label=f'Curtains', 
        c='b', lw=2, linestyle=':')
ax[1].step(bins[:-1], H_m_norm_density, where='post', label='Microbursts', 
        c='g', lw=2, linestyle='--')

plt.suptitle('Distribution of the Auroral Electrojet index\n'
                'for curtains, microbursts, and index')
ax[0].set(xlabel='AE [nT]', ylabel='Probability density', 
        xlim=(0, 1000), ylim=(0, 1.1*np.max(H_AE_density)))
ax[0].text(0.01, 0.98, '(a) Unnormalized', ha='left', va='top', 
        transform=ax[0].transAxes, fontsize=15)
ax[1].set(xlabel='AE [nT]')
ax[1].legend(loc=4)
ax[1].text(0.01, 0.98, '(b) Normalized', ha='left', va='top', 
        transform=ax[1].transAxes, fontsize=15)

plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.show()