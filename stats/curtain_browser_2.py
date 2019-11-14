import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import pandas as pd
from datetime import date, datetime
import plot_curtains
from matplotlib.widgets import Button, TextBox

catalog_save_dir = plot_curtains.CATALOG_DIR

class Browser(plot_curtains.PlotCurtains):
    def __init__(self, catalog_version, plot_width=5, 
                catalog_save_name=None, width_tol=None, filterDict={}, 
                jump_to_latest=True):
        """
        This class plots the AC6 microbursts and allows the user to browse
        detections in the future and past with buttons. Also there is a button
        to mark the event as a microburst.
        """
        plot_curtains.PlotCurtains.__init__(self, catalog_version, 
                                plot_width=plot_width, plot_width_flag=False, 
                                make_plt_dir_flag=False)
        self.filter_catalog(filterDict=filterDict)
        # Filter out events with widths whithin a width_tol.
        if width_tol is not None:
            self.catalog = self.catalog[np.isclose(
                            self.catalog['peak_width_A'], 
                            self.catalog['peak_width_B'], rtol=width_tol)]

        if catalog_save_name is None:
            self.catalog_save_name = f'AC6_curtains_sorted_v{catalog_version}.txt'
        else:
            self.catalog_save_name = catalog_save_name
        self.catalog_save_path = os.path.join(catalog_save_dir, 
                                            self.catalog_save_name)

        # Load the filtered catalog if it already exists. This is
        # userful if you can't sort all microbursts at once!
        if os.path.exists(self.catalog_save_path):
            self.load_filtered_catalog()
        else:
            self.curtain_idx = np.array([])

        self.current_date = date.min
        self._init_plot()
        if jump_to_latest and len(self.curtain_idx):
            self.index = self.curtain_idx[-1]
        else:
            # Start at row 0 in the dataframe.
            self.index = 0 
        self.plot()
        return

    def next(self, event):
        """ Plots the next detection """
        # Just return if at the end of the dataframe.
        if self.index + 1 >= self.catalog.shape[0]:
            return
        self.index += 1
        self.plot()
        return

    def prev(self, event):
        """ Plots the previous detection """
        # Just return if at the end of the dataframe.
        if self.index == 0:
            return
        self.index -= 1
        self.plot()
        return

    def append_remove_curtain(self, event):
        """ 
        Appends or removes the current catalog row to 
        self.filtered_catalog which will then
        be saved to a file for later processing.
        """
        if self.index not in self.curtain_idx:
            self.curtain_idx = np.append(self.curtain_idx, self.index)
            self.bmicroburst.color = 'g'
            print('Curtain saved at', self.catalog.iloc[self.index].dateTime)
        else:
            self.curtain_idx = np.delete(self.curtain_idx, 
                np.where(self.curtain_idx == self.index)[0])
            self.bmicroburst.color = '0.85'
            print('Curtain removed at', self.catalog.iloc[self.index].dateTime)
        return

    def key_press(self, event):
        """
        Calls an appropriate method depending on what key was pressed.
        """
        if event.key == 'm':
            # Mark as a curtain (can't use the "c" key since it is 
            # the clear command)
            self.append_remove_curtain(event)
        elif event.key == 'a':
            # Move self.index back and replot.
            self.prev(event)
        elif event.key =='d':
            # Move the self.index forward and replot.
            self.next(event)
        return
       
    def change_index(self, index):
        self.index = int(index)
        self.plot()
        return

    def plot(self):
        """ 
        Given a self.current_row in the dataframe, make a space-time plot 
        """
        print('Index position = {}/{}'.format(
                    self.index, self.catalog.shape[0]-1))
        current_row = self.catalog.iloc[self.index]
        self.index_box.set_val(self.index)
        self._clear_ax()

        if current_row['dateTime'].date() != self.current_date:
            # Load current day AC-6 data if not loaded already
            print('Loading data from {}...'.format(current_row['dateTime'].date()), 
                    end=' ', flush=True)
            self.load_ten_hz_data(current_row.dateTime.date())
            self.current_date = current_row.dateTime.date()
            print('done.')

        # Turn microburst button green if this index has been marked as a microburst.
        if self.index in self.curtain_idx:
            self.bmicroburst.color = 'g'
        else:
            self.bmicroburst.color = '0.85'
           
        self.make_plot(current_row, savefig=False)
        self.ax[0].set_title('AC6 Curtain Browser\n {} {}'.format(
                        current_row['dateTime'].date(), 
                        current_row['dateTime'].time()))
        self.ax[0].set_ylabel('dos1rate\n[counts/s]')
        self.ax[1].set_ylabel('dos1rate\n[counts/s]')
        self.ax[1].set_xlabel('UTC')
        
        self._print_aux_info(current_row)
        plt.draw()
        return

    def _print_aux_info(self, current_row):
        """ Print separation info as well as peak width info to the canvas. """
        self.textbox.clear()
        self.textbox.axis('off')
        col1 = ('Lag_In_Track = {} s\nDist_In_Track = {} km\n'
                    'Dist_total = {} km\npeak_width_A = {} s\n'
                    'peak_width_B = {} s'.format(
                    round(current_row['Lag_In_Track'], 1), 
                    round(current_row['Dist_In_Track'], 1), 
                    round(current_row['Dist_Total'], 1), 
                    round(current_row['peak_width_A'], 2), 
                    round(current_row['peak_width_B'], 2)))
        col2 = ('time_cc = {}\nspace_cc = {}\n'.format(
                    round(current_row['time_cc'], 2), 
                    round(current_row['space_cc'], 1)
                    ))
        self.textbox.text(0, 1, col1, va='top')
        self.textbox.text(1.3, 1, col2, va='top')
        return

    def _clear_ax(self):
        [a.clear() for a in self.ax]
        return 

    def _init_plot(self):
        """
        Initialize subplot objects and text box.
        """
        fig, self.ax = plt.subplots(2, figsize=(8, 7))
        plt.subplots_adjust(bottom=0.2)

        # Define button axes.
        self.axprev = plt.axes([0.54, 0.06, 0.12, 0.075])
        self.axburst = plt.axes([0.67, 0.06, 0.13, 0.075])
        self.axnext = plt.axes([0.81, 0.06, 0.12, 0.075])

        # Define buttons and their actions.
        self.bnext = Button(self.axnext, 'Next (d)', hovercolor='g')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(self.axprev, 'Previous (a)', hovercolor='g')
        self.bprev.on_clicked(self.prev)
        self.bmicroburst = Button(self.axburst, 'Curtain (m)', hovercolor='g')
        self.bmicroburst.on_clicked(self.append_remove_curtain)

        # Define the textbox axes.
        self.textbox = plt.axes([0.1, 0.05, 0.2, 0.075])
        self.textbox.axis('off')
        # Define index box.
        self.axIdx = plt.axes([0.59, 0.01, 0.32, 0.04])
        self.index_box = TextBox(self.axIdx, 'Index')
        self.index_box.on_submit(self.change_index)

        # Initialise button press
        fig.canvas.mpl_connect('key_press_event', self.key_press)
        return

    def save_filtered_catalog(self):
        """
        For every index that a user clicked microburst on, save
        those rows from the catalog into a new catalog with the
        name of self.catalog_save_name.
        """
        # Return if there are no micriobursts to save.
        if not hasattr(self, 'curtain_idx'):
            return
        # Remove duplicates indicies
        self.curtain_idx = np.unique(self.curtain_idx)
        save_path = os.path.join(catalog_save_dir, self.catalog_save_name)
        print('Saving filtered catalog to {}'.format(save_path))
        df = self.catalog.iloc[self.curtain_idx]
        # Remove duplicate times (different than indicies since the same time
        # from the other sc may be assigned to a different index. 
        df.drop_duplicates(subset='dateTime')
        df.to_csv(save_path, index=False)
        return

    def load_filtered_catalog(self):
        """
        Load a filtered catalog and populate the self.microbirst_idx array
        with existing detections. This method exists to help the user resume
        the 
        """
        filtered_catalog = pd.read_csv(self.catalog_save_path)
        # Convert the catalog times to datetime objects
        for timeKey in ['dateTime', 'time_spatial_A', 'time_spatial_B']:
            filtered_catalog[timeKey] = pd.to_datetime(filtered_catalog[timeKey])
        # Convert times to numeric for easier comparison
        flt_times_numeric = date2num(filtered_catalog.dateTime)
        times_numeric = date2num(self.catalog.dateTime)
        self.curtain_idx = np.where(np.in1d(times_numeric, flt_times_numeric, 
                                    assume_unique=True))[0]
        return


callback = Browser(8, width_tol=None, filterDict={})
# Initialize the GUI
plt.show()
# Save the catalog.
callback.save_filtered_catalog()
