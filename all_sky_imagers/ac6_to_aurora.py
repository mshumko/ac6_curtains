"""
This script maps the AC6 data location, along Earth's magnetic field lines
(assumed IGRF + ext_model), to map_alt in km.

Parameters
__________
ext_model: string
    The external magnetic field model. The internal model is IGRF.
map_alt: float
    The AC6 mapping altitude in kilometers.
catalog_name: str
    The catalog (or data) name to load
catalog_dir: str
    The catalog directory to load the data from.
"""
import pathlib

import pandas as pd
import IRBEM

ext_model = 'OPQ77'
map_alt = 230

catalog_name = pathlib.Path('AC6_curtains_baseline_method_sorted_v0.csv')
catalog_dir = pathlib.Path('/home/mike/research/ac6_curtains/data/catalogs')
catalog_path = catalog_dir / catalog_name
catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=True)

coord_keys = ['alt', 'lat', 'lon']
mapped_keys = [f'{key_i}_{map_alt}_km' for key_i in coord_keys]
for key in mapped_keys:
    catalog[key] = pd.NA

m = IRBEM.MagFields(kext=ext_model)

# zip_vals = zip(catalog.index, catalog.alt, catalog.lat, catalog.lon)

for i, (time, row) in enumerate(catalog.iterrows()):
    X = {'datetime':time, 'x1':row.alt, 'x2':row.lat, 'x3':row.lon}
    m_output = m.find_foot_point(X, None, map_alt, 0)
    # print(m_output['XFOOT'], row[coord_keys].to_numpy())
    catalog.loc[time, mapped_keys] = m_output['XFOOT']

save_name = f'{catalog_name.stem}_{map_alt}_mapped.csv'
catalog.to_csv(catalog_dir/save_name)