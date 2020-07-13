# Plot single ASI frames during curtain passes

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from skyfield.api import EarthSatellite, Topos, load
import pathlib

import plot_themis_asi
from ac6_curtains import dirs

# Load the curtain catalog
catalog_name = 'AC6_curtains_themis_asi_15deg.csv'
cat_path = dirs.CATALOG_DIR / catalog_name
cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)

fig, ax = plt.subplots()

for t0, row in cat.iterrows():
    for site in row['nearby_stations'].split():
        try:
            l = plot_themis_asi.Load_ASI(site, t0)
            l.load_themis_cal()
        except (FileNotFoundError, ValueError) as err:
            continue
        l.plot_themis_asi_frame(t0.to_pydatetime(), ax=ax)
        l.plot_azel_contours(ax=ax)
        az, el = l.get_azel_from_lla(*row[['lat', 'lon', 'alt']])
        idx = l.find_nearest_azel(az.degrees, el.degrees)
        plt.scatter(idx[1], idx[0], s=50, c='r', marker='*')
        plt.savefig(
            pathlib.Path(dirs.BASE_DIR, 'all_sky_imagers', 'plots', 
                        f'{t0.strftime("%Y%m%dT%H%M%S")}_themis_asi_frame.png'), 
                    dpi=200)
        ax.clear()

