# This script explores how different the curtain distribution
# is as a function of L-MLT (dial plot) and split up by the 
# SAA location in the afternoon and morning.

import matplotlib.pyplot as plt
import os
import numpy as np

import pandas as pd

import dirs
import dial_plot

CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
CATALOG_PATH = os.path.join(dirs.CATALOG_DIR, CATALOG_NAME)
cat = pd.read_csv(CATALOG_PATH)
cat.dateTime = pd.to_datetime(cat.dateTime)
cat['ut_hour'] = np.array([t.hour for t in cat.dateTime])

# Filter by ut hour.
afternoon_hours = (14, 21)
morning_hours = (0, 9)
cat_afternoon = cat.copy()
cat_morning = cat.copy()
cat_afternoon = cat_afternoon[(cat_afternoon.ut_hour > afternoon_hours[0]) & 
                            (cat_afternoon.ut_hour < afternoon_hours[1])]
cat_morning = cat_morning[(cat_morning.ut_hour > morning_hours[0]) & 
                        (cat_morning.ut_hour < morning_hours[1])]


mlt_bins = np.arange(0, 25)
l_bins = np.arange(2, 10)
H_afternoon, mlt_bins, lm_bins = np.histogram2d(cat_afternoon.MLT_OPQ, cat_afternoon.Lm_OPQ,
                                            bins=[mlt_bins, l_bins])
H_morning, mlt_bins, lm_bins = np.histogram2d(cat_morning.MLT_OPQ, cat_morning.Lm_OPQ,
                                            bins=[mlt_bins, l_bins])

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(121, projection='polar')
bx = plt.subplot(122, projection='polar')

d = dial_plot.Dial(ax, mlt_bins, l_bins, H_afternoon)
d.draw_dial(mesh_kwargs={'cmap':'Reds', 'vmax':np.max([H_afternoon, H_morning])}, colorbar=False)
ax.set_title(f'{afternoon_hours[0]} - {afternoon_hours[1]} UT', y=1.08)

d2 = dial_plot.Dial(bx, mlt_bins, l_bins, H_morning)
d2.draw_dial(mesh_kwargs={'cmap':'Reds', 'vmax':np.max([H_afternoon, H_morning])}, colorbar_kwargs={'label':'Number of curtains'})
bx.set_title(f'{morning_hours[0]} - {morning_hours[1]} UT', y=1.08)

plt.tight_layout(rect=(0.02, 0.05, 1, 0.9))
plt.show()