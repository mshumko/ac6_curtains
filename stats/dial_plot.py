# Code to make a dial plot with the sun facing up.

import numpy as np
import matplotlib.pyplot as plt


class Dial:
    def __init__(self, ax, angular_bins, radial_bins, H):
        """ 
        This class makes a dial (polar) plot were MLT is the azimuthal
        coordinate and L shell is the radial coordinate. 
        """
        self.ax = ax
        self.angular_bins = angular_bins
        self.radial_bins = radial_bins
        self.H = H

        if 'Polar' not in str(type(ax)):
            raise ValueError('Subplot is not polar. For example, '
                'create ax with \n ax[0] = plt.subplot(121, projection="polar")')
        return

    def draw_dial(self, colorbar=True, L_labels=[2,4,6,8], mesh_kwargs={}, colorbar_kwargs={}):
        """
        Draws a dial plot on the self.ax subplot object (must have projection='polar' kwarg). 

        colorbar=True - Plot the colorbar or not.
        L_labels=[2,4,6,8] - What L labels to plot
        mesh_kwargs={} - The dictionary of kwargs passed to plt.pcolormesh() 
        colorbar_kwargs={} - The dictionary of kwargs passed into plt.colorbar()
        """
        self.L_labels = L_labels

        angular_grid, radial_grid = np.meshgrid(self.angular_bins, self.radial_bins)

        # Try-except block deals with the dimensions of the mesh and taransposes it
        # if necessary.
        try:
            p = self.ax.pcolormesh(angular_grid*np.pi/12, radial_grid, H.T, **mesh_kwargs)
        except TypeError as err:
            if 'Dimensions of C' in str(err):
                p = self.ax.pcolormesh(angular_grid*np.pi/12, radial_grid, H, **mesh_kwargs)
            else:
                raise

        self.draw_earth()
        self._plot_params()

        if colorbar:
            plt.colorbar(p, ax=self.ax, **colorbar_kwargs)
        return

    def draw_earth(self, earth_resolution=50):
        """ Given a subplot object, draws the Earth with a shadow"""
        # Just x,y coords for a line (to map to polar coords)
        earth_circ = (np.linspace(0, 2*np.pi, earth_resolution), np.ones(earth_resolution)) 
        # x, y_lower, y_upper coords for Earth's shadow (maps to polar).
        earth_shadow = (
                        np.linspace(-np.pi/2, np.pi/2, earth_resolution), 
                        0, 
                        np.ones(earth_resolution)
                        )
        self.ax.plot(*earth_circ, c='k')
        self.ax.fill_between(*earth_shadow, color='k')
        return

    def _plot_params(self):
        # Draw L shell contours and get L and MLT labels 
        L_labels_names = self._draw_L_contours()
        mlt_labels = (self.ax.get_xticks()*12/np.pi).astype(int)
        # Sun facing up.
        self.ax.set_xlabel('MLT')
        self.ax.set_theta_zero_location("S") # Midnight at bottom
        self.ax.set_xticklabels(mlt_labels) # Transform back from 0->2pi to 0->24.
        self.ax.set_yticks(self.L_labels)
        self.ax.set_yticklabels(L_labels_names)
        return

    def _draw_L_contours(self, earth_resolution=50):
        """ Plots a subset of the L shell contours. """
        # Draw azimuthal lines for a subset of L shells.
        L_labels_names = [str(i) for i in self.L_labels[:-1]] + [f'L = {self.L_labels[-1]}']
        for L in self.L_labels:
            self.ax.plot(np.linspace(0, 2*np.pi, earth_resolution), 
                        L*np.ones(earth_resolution), ls=':', c='k')
        return L_labels_names


if __name__ == '__main__':
    import pandas as pd
    import dirs
    import os

    CATALOG_NAME = 'AC6_curtains_sorted_v8.txt'
    CATALOG_PATH = os.path.join(dirs.CATALOG_DIR, CATALOG_NAME)
    cat = pd.read_csv(CATALOG_PATH)

    mlt_bins = np.arange(0, 25)
    l_bins = np.arange(2, 10)
    H, mlt_bins, lm_bins = np.histogram2d(cat.MLT_OPQ, cat.Lm_OPQ,
                                                bins=[mlt_bins, l_bins])

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    d = Dial(ax, mlt_bins, l_bins, H)
    d.draw_dial(mesh_kwargs={'cmap':'Reds'}, colorbar_kwargs={'label':'Number of curtains'})
    plt.show()
