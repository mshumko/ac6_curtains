# Code to make a dial plot with the sun facing up.

import numpy as np
import matplotlib.pyplot as plt

def dial(ax, angular_bins, radial_bins, H, colorbar=True, mesh_kwargs={}, colorbar_kwargs={}):
    """ 
    Draws a dial plot on the ax subplot object (must have
    projection='polar' kwarg). Kwargs dictionary will be fed in directly 
    to plt.pcolormesh(). 
    """
    if 'Polar' not in str(type(ax)):
        raise ValueError('Subplot is not polar. For example, create ax with ' 
                        'ax[0] = plt.subplot(121, projection="polar")')

    angular_grid, radial_grid = np.meshgrid(angular_bins, radial_bins)
    p = ax.pcolormesh(angular_grid*np.pi/12, radial_grid, H.T, **mesh_kwargs)

    if colorbar:
        plt.colorbar(p, ax=ax, **colorbar_kwargs)
    draw_earth(ax)

    # Draw L shell contours and get L and MLT labels 
    L_labels, L_labels_names = draw_L_contours(ax)
    mlt_labels = (ax.get_xticks()*12/np.pi).astype(int)
    # Sun facing up.
    ax.set_theta_zero_location("S")
    ax.set_xlabel('MLT')
    ax.set_theta_zero_location("S") # Midnight at bottom
    ax.set_xticklabels(mlt_labels) # Transform back from 0->2pi to 0->24.
    ax.set_yticks(L_labels)
    ax.set_yticklabels(L_labels_names)
    return

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

if __name__ == '__main__':
    import pandas as pd
    import dirs
    import os

    CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
    CATALOG_PATH = os.path.join(dirs.CATALOG_DIR, CATALOG_NAME)
    cat = pd.read_csv(CATALOG_PATH)

    mlt_bins = np.arange(0, 25)
    l_bins = np.arange(2, 10)
    cat_dist, mlt_bins, lm_bins = np.histogram2d(cat.MLT_OPQ, cat.Lm_OPQ,
                                                bins=[mlt_bins, l_bins])

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    dial(ax, mlt_bins, l_bins, cat_dist, mesh_kwargs={'cmap':'Reds', 'label':'Number of curtains'})
    plt.show()
