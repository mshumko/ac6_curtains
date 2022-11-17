import pathlib
import tarfile
from datetime import timedelta
import dateutil.parser

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import matplotlib.pyplot as plt

from ac6_curtains.ac6 import load_ac6


def load(day, wwln_dir=pathlib.Path(__file__).parents[2] /'wwln', time_range=None):
    """
    Loads WWLN data and optionally filters it by time.
    """
    if isinstance(day, str):
        day = dateutil.parser.parse(day)

    matched_files = list(pathlib.Path(wwln_dir).glob(f'AfilesGZipped{day.year}.tar'))
    assert len(matched_files) == 1, (f'{len(matched_files)} WWLN files found at '
                                     f'{wwln_dir} for year {day.year}')

    with tarfile.open(matched_files[0]) as f:
        file_names = [t.name for t in f.getmembers()]
        matched_file_name = [file_name for file_name in file_names 
                        if day.strftime("%Y%m%d") in file_name][0]
        
        df = pd.read_csv(f.extractfile(matched_file_name), compression='gzip', 
                        header=None, error_bad_lines=False, 
                        names=['date', 'time', 'lat', 'lon', 'alt', '?'])
        df.index = pd.to_datetime(df['date'] + ' ' + df['time'], 
                        format='%Y/%m/%d %H:%M:%S.%f')
        df = df.drop(['date', 'time'], axis=1)

    if time_range is not None:
        df = df.sort_index()
        df = df.loc[time_range[0]:time_range[1], :]
        assert df.shape[0] > 0, 'No time filtered data found.'
    
    return df


def lightning_map():
    strike_time = dateutil.parser.parse('2015-07-27T05:59:39')
    time_window_min = 10
    time_range = [strike_time - timedelta(minutes=time_window_min/2),
                strike_time + timedelta(minutes=time_window_min/2)]

    wwln_data = load(strike_time, time_range=time_range)
    ac6a_data = load_ac6('A', strike_time, time_range=time_range)
    ac6b_data = load_ac6('B', strike_time, time_range=time_range)

    # Map along field line
    # mapped_lla = asilib.map_along_magnetic_field()

    fig = plt.figure(figsize=(10, 5))
    lat_window = 10
    lon_window = 20
    # projection=ccrs.Orthographic(central_longitude=ac6a_data.loc[strike_time, 'lon'], 
    #                             central_latitude=ac6a_data.loc[strike_time, 'lat'])
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    lon_min = ac6a_data.loc[strike_time, 'lon']-lon_window/2
    lon_max = ac6a_data.loc[strike_time, 'lon']+lon_window/2
    lat_min = ac6a_data.loc[strike_time, 'lat']-lat_window/2
    lat_max = ac6a_data.loc[strike_time, 'lat']+lat_window/2
    ax.set_extent((lon_min, lon_max, lat_min, lat_max), crs=ccrs.PlateCarree())
    # ax.coastlines()
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')

    ax.gridlines()

    wwln_data = wwln_data[(wwln_data.lon > lon_min) & (wwln_data.lon < lon_max) & 
                        (wwln_data.lat > lat_min) & (wwln_data.lat < lat_max)]

    ax.scatter(ac6a_data.loc[strike_time, 'lon'], ac6a_data.loc[strike_time, 'lat'], 
                c='r', transform=projection)
    ax.scatter(ac6b_data.loc[strike_time, 'lon'], ac6b_data.loc[strike_time, 'lat'], 
                c='b', transform=projection)
    ax.plot(ac6a_data.loc[:, 'lon'], ac6a_data.loc[:, 'lat'], 
                c='k', transform=projection)
    ax.scatter(wwln_data.loc[time_range[0]:time_range[1], 'lon'].to_numpy(), 
            wwln_data.loc[time_range[0]:time_range[1], 'lat'].to_numpy(), 
            transform=projection)
    plt.show()