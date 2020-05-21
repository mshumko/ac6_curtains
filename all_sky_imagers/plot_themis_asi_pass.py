# Plot single ASI frames during curtain passes

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
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
        print(ac6_data)

        # Plot a star where AC6 observed the curtain in AzEl.
        az, el = l.get_azel_from_lla(*row[['lat', 'lon', 'alt']])
        idx = l.find_nearest_azel(az.degrees, el.degrees)
        ax.scatter(idx[1], idx[0], s=50, c='r', marker='*')

        plt.savefig((f'./plots/{t0.strftime("%Y%m%dT%H%M%S")}_'
                    'themis_asi_frame.png'), dpi=200)
        ax.clear()

