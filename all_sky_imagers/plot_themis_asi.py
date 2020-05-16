import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from datetime import datetime

import cdflib

site = 'gbay'
time = datetime(2015, 7, 21, 3)
path = (f'/home/mike/research/ac6_curtains/data/asi/'
        f'thg_l1_asf_{site.lower()}_{time.strftime("%Y%m%d%H")}_v01.cdf')

data = cdflib.cdfread.CDF(path)

time = cdflib.cdfepoch().to_datetime(data[f"thg_asf_{site}_epoch"][:], to_np=True)
imgs = data[f"thg_asf_{site}"]

fig, ax = plt.subplots()

hi = ax.imshow(imgs[500], cmap="gray", origin="lower", 
                norm=matplotlib.colors.LogNorm(), interpolation="none")  # priming

plt.show()