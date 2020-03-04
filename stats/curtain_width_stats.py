import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

import dirs

"""
Script to calculate the curtain width statistics.
"""

catalog_name='AC6_curtains_baseline_method_sorted_v0.txt'
catalog_path = os.path.join(dirs.CATALOG_DIR, catalog_name)

cat = pd.read_csv(catalog_path)
thresh=0.25
cat_similar_width = cat[((cat['width_A']/cat['width_B'] > 1+thresh) | 
                        (cat['width_B']/cat['width_A'] > 1+thresh))]

fig, ax = plt.subplots(3, sharex=True)
bins = np.arange(0, 5, 0.25)
ax[0].hist(cat['width_A'], bins=bins)
ax[1].hist(cat['width_B'], bins=bins)
ax[2].hist(cat_similar_width['width_B'], bins=bins)

ax[0].set(ylabel='AC6A', title='AC6 Curtain Width Distributions')
ax[1].set(ylabel='AC6B', xlim=(0, None))
ax[-1].set(ylabel=f'Similar widths\n(within {100*thresh} %)', xlabel='Curtain FWHM [s]')
plt.show()