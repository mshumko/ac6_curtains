import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import csv
import os
import dateutil.parser
plt.rcParams.update({'font.size':13})

BASE_DIR = '/home/mike/research/ac6_curtains/'
CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
CATALOG_PATH = os.path.join(BASE_DIR, 'data/catalogs', CATALOG_NAME)
cat = pd.read_csv(CATALOG_PATH)
NORM_FLAG = True

# Load the L-MLT normalization files.
with open('/home/mike/research/ac6_curtains/data/norm/ac6_L_MLT_bins_same_loc.csv') as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = list(map(float, bins[key]))
with open('/home/mike/research/ac6_curtains/data/norm/ac6_L_MLT_norm_same_loc.csv') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    norm = 10*np.array(list(reader)).astype(float) # Convert to number of samples.


def draw_earth(ax, earth_resolution=50):
    """ Given a subplot object, draws the Earth with its shadow and a few L shell contours"""
    # Just x,y coords for a line (to map to polar coords)
    earth_circ = (np.linspace(0, 2*np.pi, earth_resolution), np.ones(earth_resolution)) 
    # x, y_lower, y_upper coords for Earth's shadow (also to map to polar).
    earth_shadow = (np.linspace(-np.pi/2, np.pi/2, earth_resolution), 0, np.ones(earth_resolution))
    ax.plot(*earth_circ, c='k')
    ax.fill_between(*earth_shadow, color='k')
    return

def draw_L_contours(ax, L_labels=[2, 4, 6, 8], earth_resolution=50):
    """ Plots a subset of the L shell contours. """
    # Draw azimuthal lines for a subset of L shells.
    L_labels_names = [str(i) for i in L_labels[:-1]] + [f'L = {L_labels[-1]}']
    for L in L_labels:
        ax.plot(np.linspace(0, 2*np.pi, earth_resolution), 
                    L*np.ones(earth_resolution), ls=':', c='k')
    return L_labels, L_labels_names


fig = plt.figure(figsize=(11, 5))
ax = 2*[None]
ax[0] = plt.subplot(121, projection='polar')
ax[1] = plt.subplot(122, projection='polar')

cat_dist, mlt_bins, lm_bins = np.histogram2d(cat.MLT_OPQ, cat.Lm_OPQ,
                     bins=[bins['MLT_OPQ'], bins['Lm_OPQ']])
if NORM_FLAG:
    scaling_factors = (np.max(norm)/norm).T
    # Set the sectors with no observations to NaN.
    scaling_factors[np.isinf(scaling_factors)] = np.nan
    cat_dist *= scaling_factors

mltmlt, ll = np.meshgrid(mlt_bins, lm_bins)
if NORM_FLAG:
    p1 = ax[0].pcolormesh(mltmlt*np.pi/12, ll, cat_dist.T, cmap='Reds', vmax=300)
else:
    p1 = ax[0].pcolormesh(mltmlt*np.pi/12, ll, cat_dist.T, cmap='Reds')
plt.colorbar(p1, ax=ax[0], label=r'Number of curtains', pad=0.1)

# L shell filter for the L-MLT plot
L_lower = 0
idL = np.where(np.array(bins['Lm_OPQ']) >= L_lower)[0][0]
p2 = ax[1].pcolormesh(np.array(bins['MLT_OPQ'])*np.pi/12, 
                    bins['Lm_OPQ'], norm/1E5, 
                    cmap='Reds', vmax=4)
plt.colorbar(p2, ax=ax[1], label=r'10 Hz Samples x $10^5$', pad=0.1)

# Draw Earth and shadow
draw_earth(ax[0])
draw_earth(ax[1])

# Draw L shell contours
draw_L_contours(ax[0])
L_labels, L_labels_names = draw_L_contours(ax[1])

### PLOT TWEEKS ###
ax[0].set_title(f'(a) AC6 curtain distribution\nNormalized = {NORM_FLAG}', y=1.08)
ax[1].set_title('(b) AC6 distribution of 10 Hz data\nsame location', y=1.08)
mlt_labels = (ax[1].get_xticks()*12/np.pi).astype(int)

for a in ax:
    a.set_xlabel('MLT')
    a.set_theta_zero_location("S") # Midnight at bottom
    a.set_xticklabels(mlt_labels) # Transform back from 0->2pi to 0->24.
    a.set_yticks(L_labels)
    a.set_yticklabels(L_labels_names)
    a.set_ylim(top=10)
    
plt.tight_layout(rect=(0.02, 0.05, 1, 0.9))
plt.show()
