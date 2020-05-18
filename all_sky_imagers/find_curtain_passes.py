# This program loads in the AC6 curtain catalogs and look for 
# curtains observed within some lat/lon threshold near an 
# all-sky imager.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from ac6_curtains import dirs


class FindPasses:
    def __init__(self, catalog_name, deg_thresh=5, asi_array=None, station_dir=None):
        """
        This class finds times when curtains were above
        all-sky imagers (ASI) within a degree threshold
        deg_thresh. The asi_array kwarg can be used to 
        specify a specific ASI array e.g. THEMIS or CARISMA.
        """
        self.catalog_name = catalog_name
        self.deg_thresh = deg_thresh
        self.asi_array = asi_array
        self.station_dir = station_dir

        self.load_curtain_catalog()
        return

    def loop(self):
        """ 
        Loop over the curtains and look for ground stations closer
        than self.deg_thresh.
        """
        self.nearby_station_codes = []
        self.neaby_curtain_idx = []

        current_year = pd.Timestamp.min.year # Initialization value

        for t, curtain_row in self.cat.iterrows():
            # If the station data for that year has not been loaded in yet.
            if t.year != current_year:
                self.load_station_catalog(t.year)
                current_year = t.year

            # Differences in the lat and lon between the current curtain
            # and all the ground stations in self.stations
            dlat = np.abs(curtain_row.lat - self.stations.latitude)
            dlon = np.abs(curtain_row.lon - self.stations.longitude)
            # Find when dlat and dlon are less than self.deg_thresh.
            near_stations = self.stations[
                                        (dlat < self.deg_thresh) & 
                                        (dlon < self.deg_thresh)
                                        ]
            # If there was a nearby station
            if near_stations.shape[0] > 0:
                self.neaby_curtain_idx.append(t)
                self.nearby_station_codes.append(' '.join(near_stations.code.values))  

        self.filtered_cat = self.cat.loc[self.neaby_curtain_idx]
        self.filtered_cat['nearby_stations'] = self.nearby_station_codes
        return

    def load_curtain_catalog(self):
        """
        Loads the AC6 curtain list and converts the dateTime 
        column into datetime objects. 
        """    
        cat_path = dirs.CATALOG_DIR / self.catalog_name
        self.cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)
        #self.cat['dateTime'] = pd.to_datetime(self.cat['dateTime'])
        return

    def load_station_catalog(self, year):
        """
        Load Kyle Murphy's station catalogs that were originally 
        downloaded from 
        https://github.com/kylermurphy/gmag/tree/master/gmag/Stations
        """    
        if self.station_dir is None:
            station_dir = pathlib.Path(dirs.BASE_DIR, 'all_sky_imagers', 
                                        'stations')
        station_paths = list(station_dir.glob(f'{year}*.txt'))
        assert len(station_paths) == 1, (f'Zero or more than one '
                                        f'station file found.\n{station_paths}')
        self.stations = pd.read_csv(station_paths[0])

        if self.asi_array is not None:
            self.stations = self.stations[
                self.stations.array == self.asi_array.upper()
                ]

        # Convert longitudes from 0 - 360 to -180 - 180
        # lon_remapped = self.stations['longitude'][self.stations['longitude'] > 180] % 180 - 180
        self.stations.loc[self.stations['longitude'] > 180, 'longitude'] = (
            self.stations.loc[self.stations['longitude'] > 180, 'longitude'] % 180 - 180
            )
        return

    def save_curtain_catalog(self, save_name):
        """
        Saves the filetered catalog of curtains and their nearby stations
        to a csv file.
        """
        save_path = dirs.CATALOG_DIR / save_name
        self.filtered_cat.to_csv(save_path)
        return


if __name__ == '__main__':
    deg_thresh=10
    catalog_name = 'AC6_curtains_baseline_method_sorted_v0.txt'
    f = FindPasses(catalog_name, asi_array='THEMIS', deg_thresh=deg_thresh)
    f.loop()
    f.save_curtain_catalog(f'AC6_curtains_themis_asi_{int(deg_thresh)}deg.csv')
