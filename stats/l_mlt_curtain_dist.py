import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import csv
import dateutil.parser
plt.rcParams.update({'font.size':13})

# Now load the L-MLT normalization files.
with open('/home/mike/research/ac6_curtains/data/norm/ac6_L_MLT_bins.csv') as f:
    keys = next(f).rstrip().split(',')
    bins = {}
    for key in keys:
        bins[key] = next(f).rstrip().split(',')
        bins[key] = list(map(float, bins[key]))
with open('/home/mike/research/ac6_curtains/data/norm/ac6_L_MLT_norm.csv') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    norm = 10*np.array(list(reader)).astype(float) # Convert to number of samples.

earth_resolution = 50
# Just x,y coords for a line (to map to polar coords)
earth_circ = (np.linspace(0, 2*np.pi, earth_resolution), np.ones(earth_resolution)) 
# x, y_lower, y_upper coords for Earth's shadow (also to map to polar).
earth_shadow = (np.linspace(-np.pi/2, np.pi/2, earth_resolution), 0, np.ones(earth_resolution))

# Plot the lifetime AC-6 separation
sep_downsample = 100
fig = plt.figure(figsize=(11, 5))
ax = 2*[None]
ax[0] = plt.subplot(121)
ax[1] = plt.subplot(111, projection='polar')

# L shell filter for the L-MLT plot
L_lower = 0
idL = np.where(np.array(bins['Lm_OPQ']) >= L_lower)[0][0]
p = ax[1].pcolormesh(np.array(bins['MLT_OPQ'])*np.pi/12, 
                    bins['Lm_OPQ'][idL:], norm[idL:, :]/1E5, 
                    cmap='Reds', vmax=4)
# Draw Earth and shadow
ax[1].plot(*earth_circ, c='k')
ax[1].fill_between(*earth_shadow, color='k')

# Draw azimuthal lines for a subset of L shells.
L_labels = [2, 4, 6, 8]
L_labels_names = [str(i) for i in L_labels[:-1]] + [f'L = {L_labels[-1]}']
for L in L_labels:
    ax[1].plot(np.linspace(0, 2*np.pi, earth_resolution), 
                L*np.ones(earth_resolution), ls=':', c='k')

plt.colorbar(p, ax=ax[1], label=r'10 Hz Samples x $10^5$')
ax[1].set_xlabel('MLT')
ax[1].set_title('AC6 simultaneous data avaliability')
#ax[1].set_ylabel('L')
ax[1].set_theta_zero_location("S") # Midnight at bottom
mlt_labels = (ax[1].get_xticks()*12/np.pi).astype(int)
ax[1].set_xticklabels(mlt_labels) # Transform back from 0->2pi to 0->24.
ax[1].set_yticks(L_labels)
ax[1].set_yticklabels(L_labels_names)

# A and B labels
ax[0].text(-0.15, 1.03, '(a)', transform=ax[0].transAxes, fontsize=20)
ax[1].text(-0.2, 1.07, '(b)', transform=ax[1].transAxes, fontsize=20)

plt.tight_layout()
plt.show()