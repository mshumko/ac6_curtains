# This script downloads the THEMIS asi data using the themisasi
# API.

import numpy as np
import pandas as pd
import pathlib
import urllib.request
from datetime import datetime
from bs4 import BeautifulSoup
import re

import themisasi
import themisasi.web

from ac6_curtains import dirs

def load_curtain_catalog(catalog_name):
    cat_path = dirs.CATALOG_DIR / catalog_name
    cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)
    return cat

def download_asi_calibration(station):
    base_url = 'http://themis.ssl.berkeley.edu/data/themis/thg/l2/asi/cal/'
    
    # Find all cdf files with the station name
    html = urllib.request.urlopen(base_url).read().decode('utf-8')
    # Scrape the HTML
    soup = BeautifulSoup(html, 'html.parser')
    # Extract all cdf files
    file_name_html = soup.findAll(href=re.compile("\.cdf$"))
    # Extract all hyperlinks (href) filenames with the station name.
    file_names = [file_name.get('href') for file_name in file_name_html 
                    if station.lower() in file_name.get('href')]

    for file_name in file_names:
        urllib.request.urlretrieve(base_url + file_name, 
                                dirs.ASI_DIR / file_name)
    return

def download_asi_movie(time, station):
    base_url = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/'
    station_url = (f'{station.lower()}/{time.year}/{time.strftime("%m")}/')
    file_name = f'thg_l1_asf_{station.lower()}_{time.strftime("%Y%m%d%H")}_v01.cdf'

    try:
        urllib.request.urlretrieve(base_url + station_url + file_name, 
                                dirs.ASI_DIR / file_name)
    except urllib.error.HTTPError as err:
        if '404' in str(err):
            print(base_url + station_url + file_name)
            raise
        else:
            raise
    return

if __name__ == '__main__':
    catalog_name = 'AC6_curtains_themis_asi_5deg.csv'
    cat = load_curtain_catalog(catalog_name)
    # url = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/kuuj/2017/04/thg_l1_asf_kuuj_2017041705_v01.cdf'
    # # urllib.request.urlretrieve(url, 'test.cdf')
    # # print(cat)
    # # themisasi.web.download(['2015-04-04T06'], ['SNAP'], pathlib.Path(dirs.BASE_DIR, 'data'), ['http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/'])

    # download_asi(datetime(2017, 4, 1, 4), 'kuuj')