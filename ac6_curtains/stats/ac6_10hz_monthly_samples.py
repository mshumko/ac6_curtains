import pandas as pd
import numpy as np
from datetime import datetime
import pathlib
import matplotlib.pyplot as plt
import re

import dirs

merged_files = sorted(list(
    pathlib.Path(dirs.AC6_MERGED_DATA_DIR).glob('*.csv')
    ))
file_names = [str(f.name) for f in merged_files]
merged_files_df = pd.DataFrame(data={'file_names':file_names, 'file_paths':merged_files})
# Get the dates of the files
merged_files_df['date_str'] = [re.findall('\d{8}', f)[0] 
                            for f in merged_files_df['file_names']]
merged_files_df.index = [datetime.strptime(date_str, '%Y%m%d') 
                            for date_str in merged_files_df['date_str']]
# Use groupby to find files in each month-year.
g = merged_files_df.groupby(pd.Grouper(freq="M"))

num_samples_df = pd.DataFrame(data={'n_samples':np.zeros(len(g.groups.keys()))}, 
                            index=g.groups.keys())

# Loop over each month-year group
for name, group in g:
    # print('\n\n', name, '\n', group)
    # Loop over each date in the group and aggrigate
    for file_path in group.file_paths:
        # Load the 10Hz data
        ac6_data = pd.read_csv(pathlib.Path(file_path))
        # Calculate the number of good 10 Hz data and append it to the 
        # number of samples DataFrame.
        num_daily_samples = ac6_data[(ac6_data.flag == 0) & (ac6_data.flag_B == 0)].shape[0]
        num_samples_df.loc[str(name), 'n_samples'] += num_daily_samples

# Calculate the orger of magnitude of the largest month bin and divide n_samples 
# by one less order of magnitude.
max_order_of_magnitude = len(str(int(num_samples_df['n_samples'].max())))
num_samples_df.loc[:, 'n_samples'] /= 10**(max_order_of_magnitude-1)
num_samples_df.plot(drawstyle="steps", linewidth=2, 
                    title='AC6 | number of quality 10 Hz samples | colocated | month bins')
plt.ylabel(f'Number of 10 Hz samples x 10E{max_order_of_magnitude-1}')
plt.show()