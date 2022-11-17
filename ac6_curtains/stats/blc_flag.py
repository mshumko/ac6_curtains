import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.colors

import dirs
import IRBEM

def calc_blc_flag(alt=500, kext='OPQ77', 
                lat_bins=np.arange(-90, 91, 5), 
                lon_bins=np.arange(-180, 181, 5),
                t0=datetime(2019, 1, 1)):
    """
    For a given time, altitude, and magnetic field model this function 
    estimates at what latitude and longitude bins the spacecraft is in
    the bounce loss cone (BLC) or in the open field lines.

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
        A 2D array filled with 1s in the BLC and 0s elsewhere. The rows
        correspond to latitude while the columns correspond to the 
        longitude.
    """
       
    model = IRBEM.MagFields(kext=kext)

    lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins)
    grid = np.nan*np.zeros_like(lat_grid)

    for lon_i in range(grid.shape[0]):
        for lat_j in range(grid.shape[1]):
            x = {'dateTime':t0, 'x1':alt, 'x2':lat_grid[lon_i, lat_j], 
                'x3':lon_grid[lon_i, lat_j]} # alti, lati, East longi
            
            # Calculate L values
            output_dictionary = model.make_lstar(x, None)
            # Negative L values means we are in the BLC.
            if output_dictionary['Lm'][0] < 0:
                grid[lon_i, lat_j] = 1
            else: 
                grid[lon_i, lat_j] = 0            
    return lat_bins, lon_bins, grid.T

if __name__ == '__main__':
    lat_bins, lon_bins, grid = calc_blc_flag()

    # Save to file
    df = pd.DataFrame(data=grid, index=lat_bins, columns=lon_bins)
    df.to_csv(os.path.join(dirs.BASE_DIR, 'data', 'lat_lon_blc_flag.csv'))
    df2 = pd.read_csv(os.path.join(dirs.BASE_DIR, 'data', 'lat_lon_blc_flag.csv'))
    
    c = matplotlib.colors.ListedColormap(['w', 'g'])
    p = plt.pcolormesh(lon_bins, lat_bins, grid, cmap=c)
    cbar = plt.colorbar(p, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Trapped/DLC', 'BLC/open'])
    plt.show()