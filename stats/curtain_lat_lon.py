# Plot the curtain detections on a map of the Earth
import matplotlib.pyplot as plt
import os
import pandas as pd

import cartopy.crs as ccrs

import dirs


class MapPlot:
    def __init__(self,):

        self.fig = plt.figure(figsize=(12, 6))
        self.ax = plt.subplot(111, projection=ccrs.PlateCarree())
        return

    def load_catalog(self, catalog_name='AC6_curtains_sorted_v8.txt'):
        CATALOG_PATH = os.path.join(dirs.CATALOG_DIR, catalog_name)
        self.cat = pd.read_csv(CATALOG_PATH)
        return

    def draw_earth(self):
        self.ax.coastlines()
        self.ax.gridlines(draw_labels=True)
        return

    def draw_curtains(self):
        sc = self.ax.scatter(self.cat['lon'], self.cat['lat'],
                    transform=ccrs.PlateCarree(), s=10)
        self.ax.set_title('AC6 Curtain observations', y=1.08)
        self.ax.text(-0.05, 0.55, 'latitude', va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=self.ax.transAxes)
        self.ax.text(0.5, -0.1, 'longitude', va='bottom', ha='center',
            rotation='horizontal', rotation_mode='anchor',
            transform=self.ax.transAxes)
        return

if __name__ == '__main__':
    m = MapPlot()
    m.load_catalog()

    cat2 = m.cat[(m.cat.lat > 56) & (m.cat.lat < 76) &
                (m.cat.lon > -30) & (m.cat.lon < 10)]
    #cat2 = m.cat[m.cat.Loss_Cone_Type == 2]
    print(cat2.shape)

    m.draw_earth()
    m.draw_curtains()
    plt.show()