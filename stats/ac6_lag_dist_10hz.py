import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

import dirs

# This script estimates the distribution
# of in-track lags where AC6 was taking data
# together.

ac6_merged_paths = pathlib.Path(dirs.AC6_MERGED_DATA_DIR).glob('*')

lag_bins_s = np.arange(0, 100)
hist = np.zeros(len(lag_bins_s)-1)

for path in ac6_merged_paths:
    ac6_data = pd.read_csv(path)
    ac6_data['In_Track_Lag'] = np.abs(ac6_data['In_Track_Lag'])

    H, _ = np.histogram(ac6_data['In_Track_Lag'], bins=lag_bins_s)

    hist += H

df = pd.DataFrame(data=hist, index=lag_bins_s[:-1], columns=['samples'])
df.to_csv(pathlib.Path(dirs.NORM_DIR), 'ac6_lag_dist_10hz.csv')

plt.step(lag_bins_s, hist)
plt.show()
