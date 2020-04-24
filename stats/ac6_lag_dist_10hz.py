import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import progressbar

import dirs


# This script estimates the distribution
# of in-track lags where AC6 was taking data
# together.

ac6_merged_paths = pathlib.Path(dirs.AC6_MERGED_DATA_DIR).glob('*')
ac6_merged_paths = list(ac6_merged_paths)

lag_bins_s = np.arange(0, 100)
hist = np.zeros(len(lag_bins_s)-1)

for path in progressbar.progressbar(ac6_merged_paths):
    ac6_data = pd.read_csv(path)
    ac6_data['Lag_In_Track'] = np.abs(ac6_data['Lag_In_Track'])

    H, _ = np.histogram(ac6_data['Lag_In_Track'], bins=lag_bins_s)

    hist += H

df = pd.DataFrame(data=hist, index=lag_bins_s[:-1], columns=['samples'])
df.to_csv(pathlib.Path(dirs.NORM_DIR, 'ac6_lag_dist_10hz.csv'), index_label='Lag_In_Track')

hist_norm = hist/np.sum(hist)
plt.step(lag_bins_s[:-1], hist_norm, where='post')
plt.title('AC6 Distribution of simultaneous 10 Hz samples')
plt.xlabel('In Track Lag [s]')
plt.ylabel('Number of 10 Hz samples')
plt.show()
