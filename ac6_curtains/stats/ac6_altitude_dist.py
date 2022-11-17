# Calculate the distribution of AC6 altitudes.

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dirs

# Find the attitude files for unit B
attitude_files = pathlib.Path(dirs.AC6_DATA_PATH('b')).rglob('AC6-B*att*')


bin_width=1
bins = np.arange(500, 1000, bin_width)
H = np.zeros(len(bins)-1)

for attitude_file in attitude_files:
    att = pd.read_csv(attitude_file, usecols=[6])
    day_H, _ = np.histogram(att, bins=bins)
    
    H += day_H

H /= (sum(H)*bin_width)
plt.step(bins[1:], H, where='post')
plt.title('AC6 orbit altitude distribution')
plt.xlabel('Altitude [km]')
plt.ylabel('Probability density')
plt.show()
    
