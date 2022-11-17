from datetime import datetime
import pathlib

import numpy as np
import pandas as pd

from asi_conjunctions import config


def load_ac6(sc_id, date, dtype='10Hz', time_range=None):
    """
    Loads a day of AC6 data.
    """
    assert dtype in ['10Hz', 'survey', 'coords', 'att'], ("Incorrect data type: "
        "must be one of these four types: '10Hz', 'survey', 'coords', 'att'.")

    # Load in the csv data and parse the date and time columns into a dateTime column
    file_path = find_ac6_path(sc_id, date, dtype)
    data = pd.read_csv(file_path, na_values='-1e+31')
    data['dateTime'] = pd.to_datetime(
        data[['year', 'month', 'day', 'hour', 'minute', 'second']])
    data.index = data['dateTime']

    assert data.shape[0] > 1, f'{file_path} is empty.'

    # Optionally filter the data by time.
    if time_range is not None:
        data = data.loc[time_range[0]:time_range[1], :]
        assert data.shape[0] > 0, f'No time stamps found in {time_range}.'
    return data

def find_ac6_path(sc_id, day, dType):
    date_str = datetime.strftime(day, "%Y%m%d")
    file_name = f'AC6-{sc_id.upper()}_{date_str}_L2_{dType}_V03.csv'
    files = list(config.AC6_DIR.rglob(file_name))
    assert len(files) == 1, (f'{len(files)} AC6 files found in '
        f'{config.AC6_DIR} with {file_name} name.')
    return files[0]