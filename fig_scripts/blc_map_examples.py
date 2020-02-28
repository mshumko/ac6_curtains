import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pandas as pd
import numpy as np
import dateutil.parser
from datetime import timedelta, datetime
import string
import sys

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

sys.path.insert(0, '/home/mike/research/ac6_curtains/detect')
import detect_daily
import dirs

# curtain_times = [
#                 '2015-07-23T10:29:22.400000',
#                 '2015-07-27T10:38:16.500000', # Good
#                 '2015-08-27T23:04:37.700000', # Good
#                 '2016-09-26T00:11:57.400000', # Good
#                 '2016-10-29T01:21:38.400000', # Good
#                 '2016-12-27T00:32:10.000000',
#                 '2017-01-19T10:07:40.800000',
#                 '2017-01-19T11:45:11.100000',
#                 '2017-01-22T11:24:27.700000',
#                 '2017-01-27T10:17:47.700000',
#                 '2017-02-02T09:36:02.300000',
#                 '2017-04-27T09:34:57.200000',
#                 ]
curtain_times = [
                '2016-10-29T01:21:38.400000', # Good
                '2015-07-27T10:38:16.500000', # Good
                '2016-09-26T00:11:57.400000', # Good
                '2015-08-27T23:04:37.700000', # Good
                ]
curtain_times = [dateutil.parser.parse(t) for t in curtain_times]

projection = ccrs.PlateCarree()
fig = plt.figure(figsize=(9, 7))
gs = gridspec.GridSpec(nrows=3, ncols=len(curtain_times), figure=fig)

# Cartopy is stupid. The order of how ax and bx are 
# created matters!
bx = len(curtain_times)*[None]
for i in range(len(bx)):
    bx[i] = fig.add_subplot(gs[-1, i], projection='rectilinear')
ax = fig.add_subplot(gs[:2, :], projection=projection)

# ax = plt.subplot(111, projection=projection)
ax.set_extent([-60, 30, 40, 80], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')

# gl = ax.gridlines(color='black', linestyle=':')

# gl.ylocator = mticker.FixedLocator([0, 30, 60, 90])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER

# Load and plot the mirror_point altitude data
save_name = 'lat_lon_mirror_alt.csv'
mirror_point_df = pd.read_csv(os.path.join(dirs.BASE_DIR, 'data', save_name),
                            index_col=0, header=0)
lons = np.array(mirror_point_df.columns, dtype=float)
lats = mirror_point_df.index.values.astype(float)
ax.contour(lons, lats, 
            mirror_point_df.values, transform=projection, 
            levels=[0, 100], colors=['b', 'b'], linestyles=['dashed', 'solid'])

# Overlay L shell contours
L_lons = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_lons.npy')
L_lats = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_lats.npy')
L = np.load('/home/mike/research/mission_tools'
                '/misc/irbem_l_l.npy')
levels = [4,8]
CS = ax.contour(L_lons, L_lats, L, levels=levels, colors='k', linestyles='dotted')
plt.clabel(CS, inline=1, fontsize=10, fmt='%d')

# Load the curtain catalog.
df_cat = pd.read_csv(os.path.join(dirs.CATALOG_DIR, 
                    'AC6_curtains_baseline_method_sorted_v0.txt'), index_col=0)
df_cat.index = pd.to_datetime(df_cat.index)
coords = np.nan*np.zeros((len(curtain_times), 2))

plot_width_s = 10

for i, (time, bx_i) in enumerate(zip(curtain_times, bx)):
    coords[i] = df_cat.loc[time, ['lon', 'lat']]

    data_df = detect_daily.DetectDailyCurtains(time)
    data_df.load_data(data_df.date)
    data_df.shift_time()
    data_df.align_space_time_stamps()
    # Filter the dataframes to only the plot time range.
    start_time = time - timedelta(seconds=plot_width_s/2)
    end_time = time + timedelta(seconds=plot_width_s/2)
    df_a_flt = data_df.df_a[
        (data_df.df_a['dateTime'] > start_time) &
        (data_df.df_a['dateTime'] < end_time)
        ]
    df_b_flt = data_df.df_b[
        (data_df.df_b['dateTime_shifted'] > start_time) &
        (data_df.df_b['dateTime_shifted'] < end_time)
        ]
    
    bx_i.plot(df_a_flt['dateTime'], df_a_flt['dos1rate'], 'k', ls='--', label='AC6-A')
    bx_i.plot(df_b_flt['dateTime_shifted'], df_b_flt['dos1rate'], 'k', ls='-', label='AC6-B')

    # Add a subplot label.
    bx_i.text(0, 1, f'({string.ascii_letters[i+1]})',
         ha='left', va='top', fontsize=20, color='k',
         transform=bx_i.transAxes)
    
    bx_i.text(1, 0.99, f'dt = {abs(int(round(df_b_flt.loc[df_b_flt.index[0], "Lag_In_Track"])))} s',
         ha='right', va='top', fontsize=12, color='k',
         transform=bx_i.transAxes)

    # Format time
    bx_i.xaxis.set_major_locator(mdates.SecondLocator(interval=3))
    bx_i.xaxis.set_major_formatter(mdates.DateFormatter('%S'))
    bx_i.set_xlabel(f'Seconds from\n{datetime.strftime(start_time, "%m/%d/%y %H:%M:00")}')

bx[0].set_ylabel('dos1rate [counts/s]')

ax.scatter(coords[:,0], coords[:,1], marker='*', c='r', s=150)

ax.set_title('AC6 Curtains in the Bounce Loss Cone', fontsize=25)
ax.text(0, 1, f'(a)',
         ha='left', va='top', fontsize=20, color='k',
         transform=ax.transAxes)

for i, coord in enumerate(coords):
    ax.text(coord[0], coord[1]-1, f'{string.ascii_letters[i+1]}',
         ha='center', va='top', fontsize=20, color='r',
         transform=projection)

gs.tight_layout(fig, h_pad=-0.9, w_pad=-0.5)
plt.show()