import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import csv
import os
import dateutil.parser
plt.rcParams.update({'font.size':13})

import dirs

CATALOG_NAME = 'AC6_curtains_baseline_method_sorted_v0.csv'
CATALOG_PATH = os.path.join(dirs.CATALOG_DIR, CATALOG_NAME)
cat = pd.read_csv(CATALOG_PATH)
NORM_FLAG = False
low_exposure_thresh = 10000

COLOR_MAP = 'jet' # Try Reds, plasma, 

# Load the L-MLT normalization files.
with open(os.path.join(dirs.NORM_DIR, 'ac6_L_MLT_bins_same_loc.csv')) as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = list(map(float, bins[key]))
with open(os.path.join(dirs.NORM_DIR, 'ac6_L_MLT_norm_same_loc.csv')) as f:
    reader = csv.reader(f)
    next(reader) # skip header
    norm = 10*np.array(list(reader)).astype(float) # Convert to number of samples.


def draw_earth(ax, earth_resolution=50):
    """ 
    Given a subplot object, draws the Earth with its shadow 
    and a few L shell contours
    """
    # Just x,y coords for a line (to map to polar coords)
    earth_circ = (
        np.linspace(0, 2*np.pi, earth_resolution), 
        np.ones(earth_resolution)
        ) 
    # x, y_lower, y_upper coords for Earth's shadow (also to map to polar).
    earth_shadow = (
        np.linspace(-np.pi/2, np.pi/2, earth_resolution), 
        0, 
        np.ones(earth_resolution)
        )
    ax.plot(*earth_circ, c='k')
    ax.fill_between(*earth_shadow, color='k')
    return

def draw_L_contours(ax, color, L_labels=[2, 4, 6, 8], earth_resolution=50):
    """ Plots a subset of the L shell contours. """
    # Draw azimuthal lines for a subset of L shells.
    L_labels_names = [str(i) for i in L_labels[:-1]] + [f'L = {L_labels[-1]}']
    for L in L_labels:
        ax.plot(np.linspace(0, 2*np.pi, earth_resolution), 
                    L*np.ones(earth_resolution), ls=':', c=color)
    return L_labels, L_labels_names


fig = plt.figure(figsize=(14, 5))
ax = 3*[None]
ax[0] = plt.subplot(131, projection='polar')
ax[1] = plt.subplot(132, projection='polar')
ax[2] = plt.subplot(133, projection='polar')

cat_dist, mlt_bins, lm_bins = np.histogram2d(cat.MLT_OPQ, cat.Lm_OPQ,
                     bins=[bins['MLT_OPQ'], bins['Lm_OPQ']])

cat_dist[cat_dist == 0] = np.nan
norm[norm == 0] = np.nan

scaling_factors = (np.nanmax(norm)/norm).T
# Set the sectors with no observations or little observations to NaN.
scaling_factors[np.isinf(scaling_factors)] = np.nan
# scaling_factors[norm.T < low_exposure_thresh] = np.nan
cat_norm = cat_dist*scaling_factors


mltmlt, ll = np.meshgrid(mlt_bins, lm_bins)

p0 = ax[0].pcolormesh(mltmlt*np.pi/12, ll, cat_dist.T, 
            cmap=COLOR_MAP)
p1 = ax[1].pcolormesh(mltmlt*np.pi/12, ll, cat_norm.T/1000, 
            cmap=COLOR_MAP, vmax=3)
plt.colorbar(p0, ax=ax[0], label=r'Observed Number of curtains', 
            pad=0.11, orientation='horizontal')
plt.colorbar(p1, ax=ax[1], label=r'Normalized Number of curtains x 1000', 
            pad=0.11, orientation='horizontal')

# L shell filter for the L-MLT plot
L_lower = 0
idL = np.where(np.array(bins['Lm_OPQ']) >= L_lower)[0][0]
p2 = ax[2].pcolormesh(np.array(bins['MLT_OPQ'])*np.pi/12, 
                    bins['Lm_OPQ'], norm, 
                    cmap=COLOR_MAP, norm=colors.LogNorm()
                    )
plt.colorbar(p2, ax=ax[2], label=r'10 Hz Samples', 
            pad=0.11, orientation='horizontal')

# Draw Earth and shadow
draw_earth(ax[0])
draw_earth(ax[1])
draw_earth(ax[2])

# Draw L shell contours
L_label_colors = ['k', 'k', 'k']
L_contour_colors = L_label_colors
draw_L_contours(ax[0], L_contour_colors[0])
draw_L_contours(ax[1], L_contour_colors[1])
L_labels, L_labels_names = draw_L_contours(ax[2], L_contour_colors[2])

### PLOT TWEEKS ###
ax[0].set_title(f'(a) AC6 curtain distribution', y=1.08)
ax[1].set_title(f'(b) Normalized AC6 curtain distribution', y=1.08)
ax[2].set_title('(c) AC6 distribution of 10 Hz data', y=1.08)
mlt_labels = (ax[1].get_xticks()*12/np.pi).astype(int)


for i, a in enumerate(ax):
    a.set_xlabel('MLT', labelpad=-2)
    a.set_theta_zero_location("S") # Midnight at bottom
    a.set_xticklabels(mlt_labels) # Transform back from 0->2pi to 0->24.
    a.set_yticks(L_labels)
    a.set_rlabel_position(45)
    a.set_yticklabels(L_labels_names, color=L_label_colors[i])
    a.set_ylim(top=10)
    
#plt.tight_layout(rect=(0.02, 0.05, 1, 0.9))
plt.tight_layout()
plt.show()
