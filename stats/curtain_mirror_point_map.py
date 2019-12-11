# This program plots the world-wide distribution
# of curtains superposes the mirror point in the
# opposite hemisphere assuming locally-mirroring 
# electrons.  

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.colors

import cartopy.crs as ccrs

import dirs
import IRBEM

def calc_opposite_mirror_points(alt=500, kext='OPQ77',
                                lat_bins=np.linspace(-90, 91, num=60), 
                                lon_bins=np.linspace(-180, 191, num=50),
                                t0=datetime(2019, 1, 1)):
    """
    This function calculates the mirror in the opposite hemisphere of 
    locally mirroring electrons and saves it to a 2d array. 

    returns:
        lon_bins
        lat_bins
        mirror_grid
    """
    model = IRBEM.MagFields(kext=kext)

    lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins)
    grid = np.nan*np.zeros_like(lat_grid)
    print(lat_grid.shape, lat_bins.shape, lon_bins.shape)

    for lon_i in range(grid.shape[0]):
        for lat_j in range(grid.shape[1]):
            x = {'dateTime':t0, 'x1':alt, 'x2':lat_grid[lon_i, lat_j], 
                'x3':lon_grid[lon_i, lat_j]} # alti, lati, East longi
            output_dictionary = model.make_lstar(x, None)
            # grid[lon_i, lat_j] = np.sign(output_dictionary['Lm'][0])
            if output_dictionary['Lm'][0] < 0:
                grid[lon_i, lat_j] = 1
            else: 
                grid[lon_i, lat_j] = 0
            #print(output_dictionary)
            #print(x, IRBEM.Re*(output_dictionary['POSIT'][2]-1))
            #mirror_grid[lon_i, lat_j] = IRBEM.Re*(output_dictionary['POSIT'][2]-1)
            
    return lon_bins, lat_bins, grid

if __name__ == '__main__':
    lon_bins, lat_bins, grid = calc_opposite_mirror_points()

    # Save to file
    df = pd.DataFrame(data=grid, columns=lat_bins, index=lon_bins)
    df.to_csv('lat_lon_blc_flag.csv')

    df2 = pd.read_csv('lat_lon_blc_flag.csv')
    print(df2.head())

    
    c = matplotlib.colors.ListedColormap(['w', 'g'])

    p = plt.pcolormesh(lon_bins, lat_bins, grid.T, cmap=c)
    cbar = plt.colorbar(p, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Trapped/DLC', 'BLC/open'])
    plt.show()