# Plot single ASI frames during curtain passes

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime, timedelta
import matplotlib.dates as mdates

from skyfield.api import EarthSatellite, Topos, load

import plot_themis_asi
from ac6_curtains import dirs

# Load the curtain catalog
catalog_name = 'AC6_curtains_themis_asi_15deg.csv'
cat_path = dirs.CATALOG_DIR / catalog_name
cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)

# Initialize the figure to plot the ASI and AC6 10 Hz 
# spatial time series.
fig = plt.figure(constrained_layout=True, figsize=(6,8))
gs = fig.add_gridspec(4, 1)
ax = fig.add_subplot(gs[:-1, 0])
bx = fig.add_subplot(gs[-1, 0])

pass_duration_min = 1

# Loop over each curtain and try to open the ASI data and 
# calibration from that date.
for t0, row in cat.iterrows():
    for site in row['nearby_stations'].split():
        try:
            l = plot_themis_asi.Load_ASI(site, t0)
            l.load_themis_cal()
        except (FileNotFoundError, ValueError) as err:
            continue
        # Plot the THEMIS ASI image and azel contours.
        l.plot_themis_asi_frame(t0.to_pydatetime(), ax=ax)
        l.plot_azel_contours(ax=ax)

        # Load and plot the AC6 10 Hz data
        ac6_data_path = pathlib.Path(dirs.AC6_MERGED_DATA_DIR, 
                            f'AC6_{t0.strftime("%Y%m%d")}_L2_10Hz_V03_merged.csv')
        ac6_data = pd.read_csv(ac6_data_path, index_col=0, parse_dates=True)
        ac6_data['dateTime_shifted_B'] = pd.to_datetime(ac6_data['dateTime_shifted_B'])
        time_range = [t0-timedelta(minutes=pass_duration_min), 
                    t0+timedelta(minutes=pass_duration_min)]
        ac6_data = ac6_data[time_range[0]:time_range[1]]

        # Plot the AC6 data in the bottom plot
        bx.plot(ac6_data.index, ac6_data.dos1rate, 'r', label='AC6A')
        bx.plot(ac6_data['dateTime_shifted_B'], ac6_data.dos1rate_B, 'b', 
                label='AC6B')
        bx.axvline(t0, c='k', ls='--')
        ac6_location_str = (f'Curtain:'
                            f'\ntime={t0.strftime("%H:%M:%S")}'
                            f'\nlat={round(row.lat, 1)}'
                            f'\nlon={round(row.lon, 1)}'
                            f'\nalt={round(row.alt, 1)} [km]')
        if row.Lag_In_Track > 0:
            bx.text(0, 1, f'AC6A ahead by {round(row.Lag_In_Track, 1)} seconds\n' + ac6_location_str, 
                    va='top', transform=bx.transAxes)
        if row.Lag_In_Track < 0:
            bx.text(0, 1, f'AC6B ahead by {abs(round(row.Lag_In_Track, 1))} seconds\n' + ac6_location_str, 
                    va='top', transform=bx.transAxes)       

        # Plot the AC6 orbit track in the ASI
        for ac6_data_time, ac6_data_row in ac6_data.iloc[::50].iterrows():
            az, el = l.get_azel_from_lla(*ac6_data_row[['lat', 'lon', 'alt']])
            idx = l.find_nearest_azel(az.degrees, el.degrees)
            ax.scatter(idx[1], idx[0], c='r', s=2)
        # Plot the first point in the AC6 orbit track
        az, el = l.get_azel_from_lla(*ac6_data.iloc[0][['lat', 'lon', 'alt']])
        idx = l.find_nearest_azel(az.degrees, el.degrees)
        ax.scatter(idx[1], idx[0], s=50, c='r', marker='x')

        # Plot a star where AC6 observed the curtain in AzEl.
        az, el = l.get_azel_from_lla(*row[['lat', 'lon', 'alt']])
        idx = l.find_nearest_azel(az.degrees, el.degrees)
        ax.scatter(idx[1], idx[0], s=50, c='r', marker='*')
        
        bx.xaxis.set_major_locator(mdates.SecondLocator(interval=30))

        plt.savefig((f'./plots/{t0.strftime("%Y%m%dT%H%M%S")}_'
                    'themis_asi_frame.png'), dpi=200)
        ax.clear()
        bx.clear()

