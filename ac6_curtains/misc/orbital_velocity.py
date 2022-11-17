# Calculate AC6's orbital velocity

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np

from ac6_curtains.detect import dirs

Re=6371 # km

def haversine(X1, X2):
    """
    Implementation of the haversine foruma to calculate total distance
    at an average altitude. X1 and X2 must be N*3 array of 
    lat, lon, alt.
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    R = (Re+(X1[:, 2]+X2[:, 2])/2)
    s = 2*np.arcsin( np.sqrt( np.sin(np.deg2rad(X1[:, 0]-X2[:, 0])/2)**2 + \
                    np.cos(np.deg2rad(X1[:, 0]))*np.cos(np.deg2rad(X2[:, 0]))*\
                    np.sin(np.deg2rad(X1[:, 1]-X2[:, 1])/2)**2 ))
    return R*s

def load_data(file_name):
    sc_id = file_name[4].lower()
    file_path = pathlib.Path(dirs.AC6_DATA_PATH(sc_id), file_name)
    data = pd.read_csv(file_path, na_values='-1e+31')
    data['dateTime'] = pd.to_datetime(
            data[['year', 'month', 'day', 'hour', 'minute', 'second']])
    return data

if __name__ == '__main__':
    file_name = 'AC6-A_20140710_L2_att_V03.csv'
    data = load_data(file_name)

    X1 = data[['lat', 'lon', 'alt']].iloc[1:]
    X2 = data[['lat', 'lon', 'alt']].iloc[:-1]
    d_km = haversine(X1, X2)

    dt_s = (data['dateTime'].iloc[1:].values - data['dateTime'].iloc[:-1].values)/1E9

    v = d_km/dt_s.astype(float)

    plt.hist(v)
    plt.show()