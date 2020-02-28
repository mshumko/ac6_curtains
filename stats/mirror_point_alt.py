import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.colors

import dirs
import IRBEM

def calc_mirror_point_alt(alt=600, kext='OPQ77', 
                lat_bins=np.linspace(-30, 91, num=200), 
                lon_bins=np.linspace(-100, 50, num=200),
                t0=datetime(2019, 1, 1)):
    """
    For a given time, altitude, and magnetic field model this function 
    estimates the mirror point of locally-mirroring particles in the
    southern hemisphere.

    Parameters
    ----------
    alt : float, optional
        Altitude of the spacecraft.
    kext : str, optional
        The magnetic field model to use.
    lat_bins, lon_bins : ndarray, optional
        Specify the latitude and longitude bins where to evaluate the L
        shells and the bounce loss cone. The resolution is increased with
        these parameters, at the expense of longer computational time.
    t0 : datetime object or ISO formatted date string, optional
        The time to evaluate the grid. Grid is largley independent of time
        at low altitudes.

    Returns
    -------
    lat_bins, lon_bins : ndarray
        Returns the default lat and lon bins used for the grid, or the
        bins specified by the user.
     grid : 2d ndarray
        A 2D array filled with mirror point altitudes. Mirror points below
        sea level are reported as 0. The rows correspond to latitude while 
        the columns correspond to the longitude.
    """
       
    model = IRBEM.MagFields(kext=kext)

    lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins)
    grid = np.nan*np.zeros_like(lat_grid)

    for lon_i in range(grid.shape[0]):
        for lat_j in range(grid.shape[1]):
            x = {'dateTime':t0, 'x1':alt, 'x2':lat_grid[lon_i, lat_j], 
                'x3':lon_grid[lon_i, lat_j]} # alti, lati, East longi
            
            # Calculate mirror point altitude
            try:
                output_dictionary = model.mirror_point_altitude(x, None)
                grid[lon_i, lat_j] = model.mirrorAlt
            except ValueError as err:
                if str(err) == 'Mirror point below the ground!':
                    grid[lon_i, lat_j] = 0
                    continue
                elif str(err) == 'This is an open field line!':
                    continue
                else:
                    raise
    return lat_bins, lon_bins, grid.T

if __name__ == '__main__':
    #lat_bins, lon_bins, grid = calc_mirror_point_alt()

    # Save to file
    save_name = 'lat_lon_mirror_alt.csv'
    #df = pd.DataFrame(data=grid, index=lat_bins, columns=lon_bins)
    #df.to_csv(os.path.join(dirs.BASE_DIR, 'data', save_name))
    df2 = pd.read_csv(os.path.join(dirs.BASE_DIR, 'data', save_name),
                    index_col=0, header=0)
    
    #c = matplotlib.colors.ListedColormap(['w', 'g'])
    lons = df2.columns.astype(float).values
    p = plt.pcolormesh(lons, df2.index, df2.values)
    cbar = plt.colorbar(p)
    
    c = plt.contour(lons, df2.index, df2.values, levels=[0, 100], 
                    colors=['k', 'k'], linestyles=['dashed', 'solid'])
    #cbar.ax.set_yticklabels(['Trapped/DLC', 'BLC/open'])
    plt.show()