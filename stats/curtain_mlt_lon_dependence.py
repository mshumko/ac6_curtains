"""
Look at curtains observed in the dusk (21-1) MLTs and see how the 
number of curtains changes as a function of local time, hence the
position of the SAA.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import dateutil.parser

BASE_DIR = '/home/mike/research/ac6_curtains/'
CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
CATALOG_PATH = os.path.join(BASE_DIR, 'data/catalogs', CATALOG_NAME)
cat = pd.read_csv(CATALOG_PATH)

# Filter dusk events
cat = cat[(cat.MLT_OPQ > 21) | (cat.MLT_OPQ < 1)]
print(f"Number of duskside events {cat.shape[0]}")

# Convert the time stamps to 
hours = np.array([dateutil.parser.parse(t).hour for t in cat.dateTime])

plt.hist(cat.lon)
plt.show()