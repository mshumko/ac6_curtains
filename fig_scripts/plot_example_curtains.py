# This program plots examples of microbursts given times.

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates
import numpy as np
import string
import os
import pandas as pd

from mission_tools.ac6.read_ac_data import read_ac_data_wrapper
# from ac6_microburst_scale_sizes.validation.plot_microbursts import PlotMicrobursts

plt.rcParams.update({'font.size': 15})

# Paths that are used everywhere in the class
CATALOG_DIR = ('/home/mike/research/ac6_microburst_scale_sizes/data/'
                'coincident_microbursts_catalogues/')
AC6_DATA_PATH = lambda sc_id: ('/home/mike/research/ac6/ac6{}/'
                                'ascii/level2'.format(sc_id))
PLOT_SAVE_DIR = '/home/mike/Desktop/ac6_microburst_validation'

class PlotCurtains:
    def __init__(self, plot_width, t0_times, sc_shift):
        """
        This class is a child of the PlotMicrobursts class and makes 
        time-aligned and space-aligned plots.
        """
        self.plot_width = timedelta(seconds=plot_width/2)
        self.sc_shift = sc_shift # What spacecraft to shift
        self.t0_times = t0_times
        return

    def plot_examples(self):
        """ 
        This method plots example curtains.
        """
        # Initialize plotting environment
        self._plot_setup()

        for i, (t0, sc) in enumerate(zip(self.t0_times, self.sc_shift)): # Loop over t0_times.
            # First load AC6 10Hz data for that day
            self.load_data(t0)
            row = self.ac6a_data[self.ac6a_data['dateTime'] == t0]
            assert len(row) > 0, 'None or multiple rows found!'
            # Make plots for that day.
            row['sc'] = sc
            self.make_plot(row, savefig=False, ax=self.ax[:, i], plot_legend=False,
                            mean_subtracted=False, plot_dos2_and_dos3=False)

            # Add text to each subplot
            # Separation info
            self.ax[0, i].text(0.99, 0.99, f's = {abs(int(round(row.Dist_In_Track)))} km', 
                            transform=self.ax[0, i].transAxes, va='top', ha='right', fontsize=15)
            self.ax[1, i].text(0.99, 0.99, f'dt = {abs(int(round(row.Lag_In_Track)))} s',
                            transform=self.ax[1, i].transAxes, va='top', ha='right', fontsize=15)
            
            self.ax[1, i].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%S'))
            self.ax[1, i].xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=1))
            xlabel = f'AC6A seconds after\n{datetime.strftime(t0-self.plot_width, "%Y/%m/%d %H:%M:00")}'
            self.ax[1, i].set_xlabel(xlabel)

            print(f'AC6A is ahead by {row["Lag_In_Track"]} s')
            
        plt.show()
        return

    def make_plot(self, row, **kwargs):
        """
        This method takes in a dataframe row from the catalog and makes a 
        space/time plot.
        """
        mean_subtracted = kwargs.get('mean_subtracted', False)
        savefig = kwargs.get('savefig', True)
        log_scale = kwargs.get('log_scale', False)
        plot_dos2_and_dos3 = kwargs.get('plot_dos2_and_dos3', False)
        plot_legend = kwargs.get('plot_legend', True)
        ax = kwargs.get('ax', self.ax)
        time_guide_flag = kwargs.get('time_guide_flag', False)

        df_time_a, df_time_b, df_space_a, df_space_b = self._get_filtered_plot_data(row)
        if mean_subtracted:
            df_time_a.loc[:, 'dos1rate'] -= df_time_a.loc[:, 'dos1rate'].mean()
            df_time_b.loc[:, 'dos1rate'] -= df_time_b.loc[:, 'dos1rate'].mean()
            df_space_a.loc[:, 'dos1rate'] -= df_space_a.loc[:, 'dos1rate'].mean()
            df_space_b.loc[:, 'dos1rate'] -= df_space_b.loc[:, 'dos1rate'].mean()

            if plot_dos2_and_dos3:
                df_time_a.loc[:, 'dos2rate'] -= df_time_a.loc[:, 'dos2rate'].mean()
                df_time_b.loc[:, 'dos2rate'] -= df_time_b.loc[:, 'dos2rate'].mean()
                #df_space_a.loc[:, 'dos2rate'] -= df_space_a.loc[:, 'dos2rate'].mean()
                df_time_a.loc[:, 'dos3rate'] -= df_time_a.loc[:, 'dos3rate'].mean()
                #df_space_a.loc[:, 'dos3rate'] -= df_space_a.loc[:, 'dos3rate'].mean()
                
        ax[0].plot(df_time_a['dateTime'], df_time_a['dos1rate'], 'r', label='AC6-A dos1')
        if plot_dos2_and_dos3:
            ax[0].plot(df_time_a['dateTime'], df_time_a['dos2rate'], 'r:', label='AC6-A dos2')
            ax[0].plot(df_time_a['dateTime'], df_time_a['dos3rate'], 'r--', label='AC6-A dos3')
            
        ax[0].plot(df_time_b['dateTime'], df_time_b['dos1rate'], 'b', label='AC6-B')
        if plot_dos2_and_dos3:
            ax[0].plot(df_time_b['dateTime'], df_time_b['dos2rate'], 'b:', label='AC6-B dos2')
        if time_guide_flag:
            ax[0].axvline(row.at['dateTime'])
        if plot_legend:
            ax[0].legend(loc=1)
        ax[1].plot(df_space_a['dateTime'], df_space_a['dos1rate'], 'r', label='AC6-A')
        ax[1].plot(df_space_b['dateTime'], df_space_b['dos1rate'], 'b', label='AC6-B')

        if log_scale:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')

        # # Print peak width if it exists in the catalog.
        # if set(['peak_width_A', 'peak_width_B']).issubset(row.index) and self.plot_width_flag:
        #     s = 'peak_width_A = {} s\npeak_width_B = {} s'.format(
        #             round(row['peak_width_A'], 2), round(row['peak_width_B'], 2))
        #     ax[0].text(0, 1, s, transform=ax[0].transAxes, va='top')
        # if savefig:
        #     save_name = '{0:%Y%m%d_%H%M%S}_ac6_validation_dist_total_{1}.png'.format(
        #                 row['dateTime'], round(row['Dist_Total']))
        #     plt.savefig(os.path.join(self.plot_save_dir, save_name))
        return

    def load_data(self, day):
        """
        Load the 10 Hz AC6 data from both spacecraft from a day.
        """
        self.ac6a_data = read_ac_data_wrapper('A', day)
        self.ac6b_data = read_ac_data_wrapper('B', day)
        return

    def _plot_setup(self):
        """
        Helper method to set up the plotting environment.
        """
        self.fig, self.ax = plt.subplots(2, len(self.t0_times), figsize=(14, 5))

        for i in range(len(self.t0_times)):
            self.ax[0, i].get_xaxis().set_visible(False)

        # Set up plot labels.
        self.ax[0, 0].set_ylabel(r'$\bf{Unshifted}$' + '\ndos1 [counts/s]')
        self.ax[1, 0].set_ylabel(r'$\bf{Shifted}$' + '\ndos1 [counts/s]')
        self.fig.suptitle('Curtains Observed by AC6', y=1)

        # subplot labels
        for i in range(len(self.t0_times)):
            self.ax[0, i].text(0, 0.99, f'({string.ascii_letters[2*i]})', va='top',
                                transform=self.ax[0, i].transAxes, fontsize=20)
            self.ax[1, i].text(0, 0.99, f'({string.ascii_letters[2*i+1]})', va='top',
                                transform=self.ax[1, i].transAxes, fontsize=20)

        plt.subplots_adjust(left=0.08, right=0.99, hspace=0.1, bottom=0.15, top=0.92)
        return

    def _get_filtered_plot_data(self, row):
        """
        Get the 
        """
        df_time_a = self.ac6a_data[
                            (self.ac6a_data['dateTime'] > row['dateTime'].iat[0]-self.plot_width) & 
                            (self.ac6a_data['dateTime'] < row['dateTime'].iat[0]+self.plot_width)
                            ]
        df_time_b = self.ac6b_data[
                            (self.ac6b_data['dateTime'] > row['dateTime'].iat[0]-self.plot_width) & 
                            (self.ac6b_data['dateTime'] < row['dateTime'].iat[0]+self.plot_width)
                            ]
                            
        if row['sc'].iat[0].lower() == 'a':
            df_space_a  = self.ac6a_data
            df_space_a.loc[:, 'dateTime'] += timedelta(seconds=row['Lag_In_Track'].iat[0])
            df_space_a = self.ac6a_data[
                            (self.ac6a_data['dateTime'] > row['dateTime'].iat[0]-self.plot_width) & 
                            (self.ac6a_data['dateTime'] < row['dateTime'].iat[0]+self.plot_width)
                            ]        
            df_space_b = df_time_b
        
        elif row['sc'].iat[0].lower() == 'b':
            df_space_a = df_time_a
            df_space_b  = self.ac6b_data
            df_space_b.loc[:, 'dateTime'] -= timedelta(seconds=row['Lag_In_Track'].iat[0])

            df_space_b = self.ac6b_data[
                            (self.ac6b_data['dateTime'] > row['dateTime'].iat[0]-self.plot_width) & 
                            (self.ac6b_data['dateTime'] < row['dateTime'].iat[0]+self.plot_width)
                            ]        
        return df_time_a, df_time_b, df_space_a, df_space_b


if __name__ == '__main__':
    plot_width_s = 17
    t0_times = [
                datetime(2015, 10, 28, 3, 19, 27, 700000),
                datetime(2015, 4, 4, 15, 2, 41, 200000),
                datetime(2015, 5, 12, 21, 27, 5, 600000),
                datetime(2017, 1, 22, 10, 0, 3, 400000)
                ]
    sc_shift = ['b', 'b', 'b', 'b']
    p = PlotCurtains(plot_width_s, t0_times, sc_shift)
    p.plot_examples()
