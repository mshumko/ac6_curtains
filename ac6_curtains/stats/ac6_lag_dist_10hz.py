import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import progressbar

import dirs


# This script estimates the distribution
# of in-track lags where AC6 was taking data
# together.
run_all = False
lag_bins_s = np.arange(0, 100)

def loop_data(lag_bins_s):
    ac6_merged_paths = pathlib.Path(dirs.AC6_MERGED_DATA_DIR).glob('*')
    ac6_merged_paths = list(ac6_merged_paths)
    hist = np.zeros(len(lag_bins_s)-1)

    for path in progressbar.progressbar(ac6_merged_paths):
        ac6_data = pd.read_csv(path)
        ac6_data['Lag_In_Track'] = np.abs(ac6_data['Lag_In_Track'])

        H, _ = np.histogram(ac6_data['Lag_In_Track'], bins=lag_bins_s)

        hist += H/10

    df = pd.DataFrame(data=hist, index=lag_bins_s[:-1], columns=['seconds'])
    df.to_csv(pathlib.Path(dirs.NORM_DIR, 'ac6_lag_dist_10hz.csv'), 
                index_label='Lag_In_Track')
    return df

if run_all:
    hist = loop_data(lag_bins_s)
else:
    hist = pd.read_csv(pathlib.Path(dirs.NORM_DIR, 'ac6_lag_dist_10hz.csv'), 
                        index_col=0)

hist = hist / 3600 / 24 # Normalize to hours

# Renormalize to sum over n bins (n kilometers unless lag_bin_s has 
# a non-default step)
n = 5
old_hist_index = hist.index.copy()
hist = hist.groupby(hist.index//n).sum()
hist.index = old_hist_index[::n]

# hist_norm = hist/np.sum(hist)
plt.step(hist.index, hist.seconds, where='post', c='k')
plt.xlim(hist.index[0], hist.index[-1])
plt.title('Distribution of colocated AC6 10 Hz Data')
plt.xlabel('In Track Lag [s]')
plt.ylabel('Days of 10 Hz data')
plt.tight_layout()
plt.show()
