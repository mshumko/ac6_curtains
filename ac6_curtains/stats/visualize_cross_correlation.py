import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dirs

# Visualize the ratio of AC6 A to B median count rates 
# for every pass during the AC6 mission
catalog_name = 'cross_calibrate_pass.csv'
catalog_path = os.path.join(dirs.CATALOG_DIR, catalog_name)

df = pd.read_csv(catalog_path)
print(df.head())

# Clean the bad values
df = df.dropna()

percentile = 50
max_lag_sec = 10

df = df[df.Lag_In_Track < max_lag_sec]

r = df[f'{percentile}p_A'] / df[f'{percentile}p_B']
r = r[~np.isinf(r)]
median_r = np.nanmedian(r)
print(f'Median ratio = {round(median_r, 3)}')

plt.hist(r, bins=np.linspace(0, 3, 20))
plt.axvline(median_r, c='k')
plt.title(f'Ratio of the {percentile}% of AC6A/AC6B during rad belt passes\nMax in-track lag = {max_lag_sec} seconds')
plt.xlabel('Ratio')
plt.ylabel('Number of passes')
plt.show()