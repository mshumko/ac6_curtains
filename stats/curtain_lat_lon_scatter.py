# Plot the curtain detections on a map of the Earth
import matplotlib.pyplot as plt
import matplotlib.colors    
import matplotlib.dates
from datetime import datetime
import os
import pandas as pd
import numpy as np

import cartopy.crs as ccrs

import dirs


class MapPlot:
    def __init__(self,):

        self.fig = plt.figure(figsize=(12, 6))
        self.ax = plt.subplot(111, projection=ccrs.PlateCarree())
        return

    def load_catalog(self, catalog_name='AC6_curtains_sorted_v8.txt'):
        CATALOG_PATH = os.path.join(dirs.CATALOG_DIR, catalog_name)
        self.cat = pd.read_csv(CATALOG_PATH, index_col=0)
        self.cat.index = pd.to_datetime(self.cat.index)
        return

    def draw_earth(self):
        self.ax.coastlines()
        self.ax.gridlines(draw_labels=True)
        return

    def draw_curtains(self, unique_times=None):
        if (unique_times is None):
            c = ['b' for _ in range(len(self.cat['lon']))]
            s = 10*np.ones_like(self.cat['lon'])
        else:
            c, s = self._get_marker_colors_sizes(unique_times)

        sc = self.ax.scatter(self.cat['lon'], self.cat['lat'],
                    transform=ccrs.PlateCarree(), s=s, c=c)
        self.ax.set_title('AC6 Curtain observations', y=1.08)
        self.ax.text(-0.05, 0.55, 'latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=self.ax.transAxes)
        self.ax.text(0.5, -0.1, 'longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=self.ax.transAxes)
        return

    def draw_blc(self, alpha=0.5):
        """ 
        Use a pcolormesh to draw the BLC/open field line region 
        with transparancy alpha=0.5
        """
        blc = pd.read_csv(os.path.join(dirs.BASE_DIR, 'data', 'lat_lon_blc_flag.csv'), 
                        index_col=0)
        blc[blc == 0] = np.nan
        c = matplotlib.colors.ListedColormap(['g'])
        p = plt.pcolormesh(blc.index, blc.columns, blc.T, cmap=c, alpha=alpha)
        return

    def _get_marker_colors_sizes(self, c_times):
        """ 
        For a set of curtain times find the matchihing index in self.cat and 
        give it a distinct color.
        """
        t_num = matplotlib.dates.date2num(self.cat.index)
        idx_array = np.nan*np.ones_like(c_times)
        colors = ['b' for _ in self.cat.index]
        sizes = [10 for _ in self.cat.index]

        for i, t_i in enumerate(c_times):
            t_i_num = matplotlib.dates.date2num(t_i)
            idx_array[i] = np.argmin(np.abs(t_i_num-t_num))
            colors[idx_array[i]] = 'r'
            sizes[idx_array[i]] = 40
        return colors, sizes


if __name__ == '__main__':
    m = MapPlot()
    m.load_catalog()

    cat2 = m.cat[(m.cat.lat > 56) & (m.cat.lat < 76) &
                (m.cat.lon > -30) & (m.cat.lon < 10)]
    print(cat2.shape)

    unique_times = [
        datetime(2015, 7, 23, 10, 29, 26, 600000), 
        datetime(2015, 7, 27, 10, 38, 21, 199999),
        datetime(2015, 8, 27, 23, 4, 44, 500000),
        datetime(2016, 10, 29, 1, 21, 38,399999),
        datetime(2017, 2, 2, 9, 36, 10, 900000)
         ]

    m.draw_earth()
    m.draw_blc()
    m.draw_curtains(unique_times=unique_times)
    plt.show()