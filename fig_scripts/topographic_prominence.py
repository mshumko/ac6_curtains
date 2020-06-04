import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates
import numpy as np
import pathlib
import string
from datetime import datetime, timedelta
import scipy.signal

from mission_tools.ac6.read_ac_data import read_ac_data_wrapper
import dirs

plt.rcParams.update({'font.size': 15})

# Load the curtain catalog
cat_name = 'AC6_curtains_baseline_method_sorted_v0.txt'
cat_path = pathlib.Path(dirs.CATALOG_DIR, cat_name)
cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)

# List of times to plot
times = [
        '2015-07-25T05:43:55.300000',
        '2015-08-07T20:38:37.100000', 
        '2016-08-12T23:10:23.900000',
        # '2016-09-26T00:11:57.400000',
        # '2017-01-22T10:00:03.400000',
        # '2017-02-02T09:36:02.300000',
        '2015-08-08T02:13:09.900000'
        ]
times_obj = sorted(pd.to_datetime(times))

# Initialize figure
fig, ax = plt.subplots(1, len(times), figsize=(14, 5))
fig.suptitle('Curtain Width Estimated Using 0.5 x Topographic Prominence', y=1)
plt.subplots_adjust(left=0.08, right=0.99, hspace=0.1, wspace=0.25, bottom=0.15, top=0.95)
plot_width_s = 4
plot_width = pd.Timedelta(seconds=plot_width_s)

def find_peak_width(df, t_0, rel_height=0.5):
    """ 
    This wrapper function finds the peak width and 
    prominence for a given dataframe df and center 
    time t_0.

    rel_height is directly passed to peak_widths
    """
    peak_index = np.where(df['dateTime'] == t_0)[0][0]
    width_tuple = scipy.signal.peak_widths(df['dos1rate'], [peak_index], 
                                        rel_height=rel_height)
    width_dict = {
        'width':width_tuple[0][0]/10, 'prominence':width_tuple[1][0],
        'left_edge':df['dateTime'].iloc[0] + pd.Timedelta(seconds=width_tuple[2][0]/10), 
        'right_edge':df['dateTime'].iloc[0] + pd.Timedelta(seconds=width_tuple[3][0]/10)
        }
    return width_dict

for i, (ax_i, t_i) in enumerate(zip(ax, times_obj)):
    # Load the 10 Hz data
    burst_data = read_ac_data_wrapper('A', t_i)
    # Filter the 10 Hz data to plot width
    burst_data_flt = burst_data[
                            (burst_data['dateTime'] > t_i-plot_width) & 
                            (burst_data['dateTime'] < t_i+plot_width)
                            ]

    # Estimate peak widths
    width_dict= find_peak_width(burst_data_flt, t_i)
    peak_index = burst_data_flt.index[0] + np.where(burst_data_flt['dateTime'] == t_i)[0][0]

    # Plot the dos1rate
    ax_i.plot(burst_data_flt['dateTime'], burst_data_flt['dos1rate'], c='r', lw=1)

    # Plot the peak width
    peak_counts = burst_data_flt.loc[peak_index, 'dos1rate']
    time_range = pd.date_range(start=width_dict['left_edge'], 
                end=width_dict['right_edge'], periods=50)
    ax_i.plot(time_range, (width_dict['prominence'])*np.ones(50), c='k', lw=1)

    # Set the x-axis labels
    ax_i.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%S'))
    ax_i.xaxis.set_major_locator(matplotlib.dates.SecondLocator(interval=2))
    ax_i.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=1))
    formatted_start_time = datetime.strftime(t_i-plot_width, "%Y/%m/%d %H:%M:00")
    xlabel = f'AC6A seconds after\n{formatted_start_time}'
    ax_i.set_xlabel(xlabel)

    ax_i.text(0, 0.99, f'({string.ascii_letters[i]}) width = '
        f'{round(cat.loc[t_i, "width_A"], 1)} [s]', va='top',
        transform=ax_i.transAxes, fontsize=15)
    # ax_i.text(1, 0.99, f'width = {round(cat.loc[t_i, "width_A"], 1)} [s]', va='top', ha='right',
    #         transform=ax_i.transAxes, fontsize=20)
    ylims = ax_i.get_ylim()
    ax_i.set_ylim(None, ylims[1]*1.1)

ax[0].set_ylabel('dos1rate [counts/s]')
plt.show()
    