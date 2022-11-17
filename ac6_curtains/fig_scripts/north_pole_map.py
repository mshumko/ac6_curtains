import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

ax = plt.subplot(111, projection=ccrs.Orthographic(central_longitude=60.0, central_latitude=90.0))
ax.coastlines()
#ax.add_feature(cartopy.feature.OCEAN, zorder=0)
#ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')

ax.set_global()
gl = ax.gridlines(color='black', linestyle=':')

gl.ylocator = mticker.FixedLocator([0, 30, 60, 90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.show()