import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import dirs

cat = pd.read_csv(os.path.join(dirs.CATALOG_DIR, 
                f'AC6_curtains_sorted_v8.txt'))

# Find what spacecraft has the correct peak width
mask_a = (cat['dateTime'] == cat['time_spatial_A'])
mask_b = ~mask_a

widths = np.nan*np.zeros_like(cat['time_spatial_A'])
widths[mask_a] = cat['peak_width_A'][mask_a]
widths[mask_b] = cat['peak_width_B'][mask_b]

plt.hist(widths)
plt.show()