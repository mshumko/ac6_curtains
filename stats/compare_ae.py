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
cat = pd.read_csv(cat_path)
cat['dateTime'] = pd.to_datetime(cat['dateTime'])

# Load the microburst catalog
burst_name = 'AC6_microbursts_sorted_v6.txt'
burst_path = os.path.join(dirs.CATALOG_DIR, burst_name)
burst_cat = pd.read_csv(burst_path)

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

thresh = 20
bin_width = 100
bins = np.arange(0, 1200, bin_width)
H_AE, _ = np.histogram(ae['AE'], density=True, bins=bins)
H_c, _ = np.histogram(cat['AE'], density=True, bins=bins)
H_c_near, _ = np.histogram(cat[cat.Lag_In_Track < thresh]['AE'], density=True, bins=bins)
H_c_far, _ = np.histogram(cat[cat.Lag_In_Track > thresh]['AE'], density=True, bins=bins)
H_m, _ = np.histogram(burst_cat['AE'], density=True, bins=bins)

print("cat[cat.Lag_In_Track < thresh]['AE'] = ", cat[cat.Lag_In_Track < thresh]['AE'].shape[0])
print("cat[cat.Lag_In_Track > thresh]['AE'] = ", cat[cat.Lag_In_Track > thresh]['AE'].shape[0])

# Scale the curtain distribution by the total AE time. In other
# words this normalized distribution is the distribution of
# curtains assuming any AE index is equally likeliy to occur.
# H_scaled = H_c*(np.max(H_AE)/H_AE)
# H_scaled = H_scaled/(np.sum(H_scaled)*bin_width)

fig, ax = plt.subplots()
ax.step(bins[:-1], H_AE, where='post', label='Index', 
        c='k', lw=2)
ax.step(bins[:-1], H_c_near, where='post', label=f'Curtains | AC6 lag < {thresh} s', 
        c='b', lw=2, linestyle=':')
ax.step(bins[:-1], H_c_far, where='post', label=f'Curtains | AC6 lag > {thresh} s', 
        c='r', lw=2, linestyle=':')
ax.step(bins[:-1], H_m, where='post', label='Microbursts', 
        c='g', lw=2, linestyle='--')

ax.set(title='Distribution of the Auroral Electrojet index\n'
                'for curtains, microbursts, and index',
        xlabel='AE [nT]', ylabel='Probability density', xlim=(0, 1000))
ax.legend()

plt.tight_layout()
plt.show()