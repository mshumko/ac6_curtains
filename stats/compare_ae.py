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

bin_width = 100
bins = np.arange(0, 1200, bin_width)
H_AE, _ = np.histogram(ae['AE'], density=True, bins=bins)
H_c, _ = np.histogram(cat['AE'], density=True, bins=bins)
H_diff = H_c - H_AE

fig, ax = plt.subplots(3, sharex=True, figsize=(8, 7))
ax[0].hist(ae['AE'], density=True, bins=bins)
ax[1].hist(cat['AE'], density=True, bins=bins)
ax[2].bar(bins[:-1], H_diff, width=bin_width, align='edge')
ax[0].set(ylabel='Probability density', title='All AE between 2014 and 2017')
ax[1].set(ylabel='Probability density', title='Curtain AE')
ax[2].set(ylabel='Probability density difference\ncurtain AE - all AE', 
            xlabel='AE [nT]', title='Difference in the probability densities')

plt.tight_layout()
plt.show()