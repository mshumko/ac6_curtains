# Plot single ASI frames during curtain passes

import pandas as pd
import matplotlib.pyplot as plt
import pathlib

import plot_themis_asi

from ac6_curtains import dirs

# Load the curtain catalog
catalog_name = 'AC6_curtains_themis_asi_5deg.csv'
cat_path = dirs.CATALOG_DIR / catalog_name
cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)

fig, ax = plt.subplots()

for t0, row in cat.iterrows():
    for site in row['nearby_stations'].split():
        try:
            l = plot_themis_asi.Load_ASI(site, t0)
        except (FileNotFoundError, ValueError) as err:
            continue
        l.plot_themis_asi_frame(t0.to_pydatetime())
        plt.savefig((f'{t0.strftime("%Y%m%dT%H%M%S")}_'
                    'themis_asi_frame.png'), dpi=200)
        ax.clear()

