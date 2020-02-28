import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# from datetime import datetime
import dateutil.parser

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import dirs

curtain_times = [
                '2015-07-23T10:29:22.400000',
                '2015-07-27T10:38:06.400000',
                '2016-09-26T00:11:57.400000',
                '2016-10-29T01:21:38.400000',
                '2016-12-27T00:32:10.000000',
                '2017-01-19T10:07:40.800000',
                '2017-01-19T11:45:11.100000',
                '2017-01-22T11:24:27.700000',
                '2017-01-27T10:17:47.700000',
                '2017-02-02T09:36:02.300000',
                '2017-04-27T09:34:57.200000',
                ]
curtain_times = [dateutil.parser.parse(t) for t in curtain_times]

# projection = ccrs.NearsidePerspective(
#     central_longitude=-30, 
#     central_latitude=60.0, 
#     satellite_height=3000000
#     )
projection = ccrs.PlateCarree()
ax = plt.subplot(111, projection=projection)
ax.set_extent([-100, 60, 10, 80], crs=ccrs.PlateCarree())
ax.coastlines()
# ax.add_feature(cartopy.feature.OCEAN, zorder=0)
ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')

gl = ax.gridlines(color='black', linestyle=':')

gl.ylocator = mticker.FixedLocator([0, 30, 60, 90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Load mirror_point altitude data
save_name = 'lat_lon_mirror_alt.csv'
mirror_point_df = pd.read_csv(os.path.join(dirs.BASE_DIR, 'data', save_name),
                            index_col=0, header=0)
lons = np.array(mirror_point_df.columns, dtype=float)
lats = mirror_point_df.index.values.astype(float)
ax.contour(mirror_point_df.columns, mirror_point_df.index, 
            mirror_point_df.values, transform=projection, alpha=0,
            levels=[0, 100], colors=['k', 'k'], linestyles=['dashed', 'solid'])
ax.contour(lons, lats, 
            mirror_point_df.values, transform=projection, 
            levels=[0, 100], colors=['k', 'k'], linestyles=['dashed', 'solid'])


# Load the curtain catalog.
df_cat = pd.read_csv(os.path.join(dirs.CATALOG_DIR, 
                    'AC6_curtains_sorted_vNone.txt'), index_col=0)
df_cat.index = pd.to_datetime(df_cat.index)
coords = np.nan*np.zeros((len(curtain_times), 2))
for i, time in enumerate(curtain_times):
    coords[i] = df_cat.loc[time, ['lon', 'lat']]

# ax.scatter(coords[:, 0], coords[:, 1], marker='x')

for i, coord in enumerate(coords):
    plt.text(coord[0], coord[1], i,
         horizontalalignment='right',
         transform=projection)

plt.show()