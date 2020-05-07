# This function calculates the normalization parameters for the 
# micorburst scale size distributions.

from datetime import datetime, timedelta
from matplotlib.dates import date2num, num2date
import csv
import numpy as np
import sys
import os

import pandas as pd

# Import personal libraries
import mission_tools.ac6.read_ac_data as read_ac_data
import IRBEM

class Hist1D:
    def __init__(self, d=None, startDate=datetime(2014, 1, 1),
                 endDate=datetime.now(), filterDict={}, flag=True):
        """
        This class calculates the 1D histograms as a function of distance
        for the various filter parameters. 

        d is the distance bin edges
        """
        if d is None:
            self.d = np.arange(0, 501, 10)
        else:
            self.d = d
        self.count = np.zeros(len(self.d)-1) # Number of days at that separation.
        dDays = (endDate - startDate).days
        self.dates = [startDate + timedelta(days=i) for i in range(dDays)] 
        self.filterDict = filterDict
        self.flag = flag
        return

    def loop_data(self, simultaneous=False, verbose=False):
        """
        Loop over every day and for each day try to open the 10 Hz data
        from both units. If data exists, filter it by time, and hisogram it. 
        """
        for day in self.dates:
            #if day != datetime(2016, 10, 31): continue
            self.load_day_data(day)
            if (self.ac6dataA is None) or (self.ac6dataB is None):
                continue # If one (or both) of the data files is empty
            ind = self.filterData(simultaneous=simultaneous, verbose=verbose)
            # If using Hist2D, the Hist1D's method will be overwritten.
            self.hist_data(ind) 

        return

    def load_day_data(self, date):
        """
        This generator function will load in one day of data at a 
        time, and return control to user.
        """
        try:   
            self.ac6dataA = read_ac_data.read_ac_data_wrapper(
                'A', date, dType='10Hz')
            self.ac6dataB = read_ac_data.read_ac_data_wrapper(
                'B', date, dType='10Hz')
            print('Loaded data from {}'.format(date))
        except AssertionError as err:
            if ( ('None or > 1 AC6 files found' in str(err)) or
                ('File is empty!' in str(err)) ):
                self.ac6dataA = None
                self.ac6dataB = None
                return self.ac6dataA, self.ac6dataB
            else:
                raise

    def filterData(self, verbose=False, simultaneous=True):
        """
        This function filters the AC-6 data by common times, data flag value,
        and filterDict dictionary.

        The simultaneous boolean kwarg changes modes from identifying
        times at the same time (default; True) or at the same position 
        by using the in-track lag (False).
        """
        if verbose:
            start_time = datetime.now()
            print('Filtering data at {}'.format(datetime.now()))
        if simultaneous:
            ind = self._filter_times()
        else:
            ind = self._filter_positions()

        ### Data quality flag filter ###
        if self.flag: # First filter by common times and flag
            indf = np.where(self.ac6dataB['flag'] == 0)[0]
            ind = np.intersect1d(ind, indf) 

        ### filerDict filter ###
        for key, value in self.filterDict.items():
            if hasattr(value, '__len__'):
                idx = np.where((self.ac6dataB[key] > np.min(value)) & 
                                (self.ac6dataB[key] < np.max(value)))[0]
            else:
                idx = np.where(self.ac6dataB[key] == value)[0]
            ind = np.intersect1d(ind, idx)
        if verbose:
            print('Data filted in {} s'.format(
                (datetime.now()-start_time).total_seconds()))
        return ind

    def hist_data(self, ind):
        """
        This method will histrogram the total distance data.
        """
        H, _ = np.histogram(self.ac6dataB['Dist_Total'][ind], bins=self.d)
        self.count += H/10
        return

    def save_data(self, fPath):
        """
        This method saves the normalization data to a csv file.
        """
        with open(fPath, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Separation [km]', 'Seconds'])

            for (d, s) in zip(self.d, self.count):
                w.writerow([d, s])
        print('Saved data to {}'.format(os.path.basename(fPath)))
        return

    def _filter_times(self):
        """ Filter the day data for data taken simultaneously """
        ### Find common times of the two data sets ###
        tA = date2num(self.ac6dataA['dateTime'])
        tB = date2num(self.ac6dataB['dateTime'])
        # np.in1d returns a boolean array that correspond to indicies in tB
        # that are also in tA. np.where will convert this mask array into
        # an index array
        ind = np.where(np.in1d(tB, tA, assume_unique=True))[0]
        return ind

    def _filter_positions(self):
        """ 
        Find how many indicies of the AC6B data were taken at the same position 
        as AC6A. 
        """
        # Shift the AC6A time stamps by the in-track lag. If these shifted 
        # time stamps are found in the AC6B data, then both units were taking 
        # data over the same spatial location.
        tA_shifted = self._get_shifted_ac6a_times()
        tB = date2num(self.ac6dataB['dateTime']) 

        # np.in1d returns a boolean array that correspond to indicies in tB
        # that are also in tA. np.where will convert this mask array into
        # an index array
        indB = np.where(np.in1d(tB, tA_shifted, assume_unique=True))[0]

        # ### TEST CODE ###
        # indA = np.where(np.in1d(tA_shifted, tB, assume_unique=True))[0]

        # for iA, iB in zip(indA, indB):
        #     print(self.ac6dataA.loc[iA, 'dateTime'], 
        #           self.ac6dataB.loc[iB, 'dateTime'],
        #           self.ac6dataA.loc[iA, 'Lag_In_Track'])#,
        #           #self.ac6dataB.loc[iB, 'dateTime'] + timedelta(seconds=self.ac6dataB.loc[iB, 'Lag_In_Track']))

        return indB

    def _get_shifted_ac6a_times(self):
        """
        Adds the in-track lag to the AC6A times and round the time stamps
        to the tenths of a second. Return the shifted time stamps in the 
        numerical format.
        """
        # Convert the AC6A time stamps to numerical format
        tA = date2num(self.ac6dataA['dateTime'])
        
        # Now look for error in-track lags (-1E31 ish values) and make them 
        # realistic. Add a day so there is no way they will be matched on the
        # same day for loop iteration.
        error_ind = np.isnan(self.ac6dataA.Lag_In_Track)
        self.ac6dataA.loc[error_ind, 'Lag_In_Track'] = 86400
        
        # Add the in-track lag in decimal day format to the AC6A.
        # If the shifted time stamp is found in the AC6B data, 
        # then they taking data in the same spatial location.
        tA_shifted_numerical = tA + self.ac6dataA.Lag_In_Track/86400
        
        # Now convert the shifted numerical AC6B times to datetimes, 
        # round to tenths of a second, and convert back to numerical 
        # times and return.
        try:
            tA_shifted_rounded = self._round_time_stamps(num2date(tA_shifted_numerical))
        except ValueError as err:
            print(min(tA_shifted_numerical), max(tA_shifted_numerical),
                 any(np.isnan(tA_shifted_numerical)))
            raise
        return date2num(tA_shifted_rounded)

    def _round_time_stamps(self, time_array):
        """ Round the time stamps to the nearest tenth of a second """
        time_rounded = len(time_array)*[None]

        for i, t in enumerate(time_array):
            try:
                time_rounded[i] = t.replace(microsecond=round(t.microsecond, -5)) 
            except ValueError as err:
                if 'microsecond must be in 0..999999' in str(err):
                    time_rounded[i] = t.replace(microsecond=0) + timedelta(seconds=1)

        # time_rounded = [t.replace(microsecond=round(t.microsecond, -5)) 
        #                 for t in time_array]
        return time_rounded

class Hist2D(Hist1D):
    def __init__(self, histKeyX, histKeyY, bins=None, startDate=datetime(2014, 1, 1),
                 endDate=datetime.now(), filterDict={}, flag=True):
        """
        This class calculates the 2D histograms as a function of distance
        for the various filter parameters. 
        """
        Hist1D.__init__(self, d=None, startDate=startDate, 
                        endDate=endDate, filterDict=filterDict, flag=flag)
        self.histKeyX = histKeyX
        self.histKeyY = histKeyY

        if bins is None: # Determine what the bins should be.
            if 'lm' in histKeyX.lower(): # L-MLT distribution
                self.bins=np.array([np.arange(2, 10), np.arange(0, 24)])
            elif 'lm' in histKeyY.lower():
                self.bins=np.array([np.arange(0, 24), np.arange(2, 10)])

            elif 'lat' in histKeyX.lower(): # lat-lon distribution
                self.bins=np.array([np.arange(-90, 90, 5), np.arange(-180, 180, 5)])
            elif 'lat' in histKeyY.lower():
                self.bins=np.array([np.arange(-90, 90, 5), np.arange(-180, 180, 5)])
            else:
                raise ValueError('Could not determine how to make bins!'
                                ' Plase supply them')
        else:
            self.bins = bins
        
        self.count = np.zeros((len(self.bins[0])-1, len(self.bins[1])-1))
        return

    def hist_data(self, ind):
        """
        This histogram method overwrites the Hist1D's hist_data() method
        """
        # Exit if only one fileterd index on that day.
        if len(ind) > 1:
            H, xedges, yedges = np.histogram2d(
                                    self.ac6dataB.loc[ind, self.histKeyX],
                                    self.ac6dataB.loc[ind, self.histKeyY], 
                                    bins=self.bins
                                    )
            self.count += H/10
        return

    def save_data(self, fPathBin, fPathNorm):
        """
        This method saves the histrogram bins and normalization coefficients.
        """
        XX, YY = np.meshgrid(self.bins[0], self.bins[1])

        with open(fPathBin, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([self.histKeyX, self.histKeyY])
            w.writerows(self.bins)

        with open(fPathNorm, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([self.histKeyX, self.histKeyY])
            w.writerows(self.count)
        return

class Equatorial_Hist(Hist1D):
    def __init__(self, sep_bins, hist_key, hist_bins, bins=None, 
                startDate=datetime(2014, 1, 1),
                endDate=datetime.now(), modelKwargs={'kext':'OPQ77'},
                filterDict={}, flag=True):
        # Initialize init of the parent class.
        Hist1D.__init__(self, d=None, startDate=startDate, 
                        endDate=endDate, filterDict=filterDict, flag=flag)
        # Initialize the equatorial separation bins as well as the histrogram key and 
        # bins.
        self.sep_bins = sep_bins
        self.histKey = hist_key
        self.hist_bins = hist_bins

        self.counts = pd.DataFrame(
                        np.zeros((len(self.sep_bins)-1, len(self.hist_bins)-1)),
                        index=self.sep_bins[:-1], columns=self.hist_bins[:-1])

        # Initialize IRBEM model
        self.model = IRBEM.MagFields(**modelKwargs)
        return

    def loop_data(self):
        """
        Loops over all of the days between startDate and endDate, loads the 10 Hz
        data, filters it, maps to equator, and bins the resulting scale sizes.
        """
        for day in self.dates:
            self.load_day_data(day)
            if (self.ac6dataA is None) or (self.ac6dataB is None):
                continue # If one (or both) of the data files is empty
            # ind contains AC6-B indicies that have a correpsonding AC6-A time
            #  after all of the filtering.
            self.indB = self.filterData()
            # Map to equator
            d_equator_array = self.map2equator(self.indB)
            if not np.isnan(d_equator_array).all():
                # Bin events if at least one event is valid.
                self.hist_data(self.indB, d_equator_array) 
        return

    def map2equator(self, indB):
        """ 
        Maps points from AC6 to the magnetic equator. ind index array
        is for AC6-B. 
        """
        d_equator_array = np.nan*np.zeros_like(indB)

        # Find corresponding indicies in AC6-A.
        # Note that indB has already been filtered for data flag
        # and by other variables in filterDict.
        tA = date2num(self.ac6dataA['dateTime'])
        tB = date2num(self.ac6dataB['dateTime'][indB])
        indA = np.where(np.in1d(tA, tB, assume_unique=True))[0]
        if len(indA) != len(indB):
            raise ValueError('indA != indB')
        
        for i, (i_a, i_b) in enumerate(zip(indA, indB)):
            # Map to the equator.
            d_equator_array[i] = self._equator_mapper(i_a, i_b)
        return d_equator_array

    def hist_data(self, indB, d_equator_array):
        """ This method histrograms the histKey value"""
        H, x_bins, y_bins = np.histogram2d(
                    d_equator_array, self.ac6dataB.loc[indB, self.histKey], 
                    bins=[self.sep_bins, self.hist_bins])
        self.counts += pd.DataFrame(H, index=self.sep_bins[:-1], 
                                        columns=self.hist_bins[:-1])
        return

    def save_data(self, file_path):
        """ 
        Saves the counts array with rows and columns to a file. 
        Example code to read this data back into python:
        df = pd.read_csv(file_path, index_col=0)
        """
        self.counts.to_csv(file_path)
        return

    def _equator_mapper(self, i_a, i_b):
        """ Helper function to format data for IRBEM and map to the equator. """
        Re = 6371 # Earth radius, km
        X1 = {'x1':self.ac6dataA['alt'][i_a], 'x2':self.ac6dataA['lat'][i_a], 
              'x3':self.ac6dataA['lon'][i_a], 
              'dateTime':self.ac6dataA['dateTime'][i_a].to_pydatetime()}
        X2 = {'x1':self.ac6dataB['alt'][i_b], 'x2':self.ac6dataB['lat'][i_b], 
              'x3':self.ac6dataB['lon'][i_b], 
              'dateTime':self.ac6dataB['dateTime'][i_b].to_pydatetime()}
        # Run IRBEM
        X1_equator = self.model.find_magequator(X1, None)['XGEO']
        X2_equator = self.model.find_magequator(X2, None)['XGEO']
       
        if (-1E31 in X1_equator) or (-1E31 in X2_equator):
            return np.nan
        else:
            # Calculate separation in units of km.
            return Re*np.linalg.norm(X1_equator-X2_equator)

if __name__ == '__main__':
    SAVE_DIR = '/home/mike/research/ac6_curtains/data/norm'

    ### SCRIPT TO MAKE "Dst_Total" NORMALIZATION ###
    #import time
    #start_time = time.time()
    # s=Hist1D(d=np.arange(0, 501, 1), 
    #             filterDict={'dos1rate':[0, 1E6], 
    #                         'Lm_OPQ':[4, 8]})
    # s.loop_data()
    # s.save_data(os.path.join(SAVE_DIR, 'ac6_norm_all_1km_bins.csv'))
    
    # bin_width = 5
    # bin_offset = 0
    # L_array = [4, 5, 6, 7, 8] #[4, 8]
    # for L_lower, L_upper in zip(L_array[:-1], L_array[1:]):
    #     ss2=Hist1D(d=np.arange(bin_offset, 501, bin_width), 
    #                 filterDict={'dos1rate':[0, 1E6], 
    #                             'Lm_OPQ':[L_lower, L_upper]})
    #     ss2.loop_data()
    #     SAVE_DIR = '/home/mike/research/ac6_microburst_scale_sizes/data/norm'
    #     ss2.save_data(os.path.join(SAVE_DIR, 
    #             f'ac6_norm_{L_lower}_L_{L_upper}'
    #             f'_{bin_width}km_bins_offset.csv'))
    #     print('Run time =', time.time()-start_time, 's')
    # print('Norm.py ran in :{} s'.format((datetime.now()-st).total_seconds()))

    ### SCRIPT TO MAKE L-dependent "Dst_Total" NORMALIZATION ###
    # st = datetime.now()
    # L = [3, 4, 5, 6, 7]
    # for (lL, uL) in zip(L[:-1], L[1:]):
    #     ss=Hist1D(filterDict={'Lm_OPQ':[lL, uL]})
    #     ss.loop_data()
    #     SAVE_DIR = '/home/mike/research/ac6-microburst-scale-sizes/data/norm/'
    #     ss.save_data(os.path.join(SAVE_DIR, 'ac6_norm_{}_L_{}.csv'.format(lL, uL)))
    # print('Norm.py ran in :{} s'.format((datetime.now()-st).total_seconds()))

    ### SCRIPT TO MAKE L-MLT NORMALIATION ###
    # ss2 = Hist2D('Lm_OPQ', 'MLT_OPQ', 
    #                 bins=[np.arange(2, 15), np.arange(0, 25)],
    #                 filterDict={'dos1rate':[0, 1E6]})
    # ss2.loop_data(simultaneous=False)
    # ss2.save_data(os.path.join(SAVE_DIR, 'ac6_L_MLT_bins_same_loc.csv'), 
    #               os.path.join(SAVE_DIR, 'ac6_L_MLT_norm_same_loc.csv'))

    ### SCRIPT TO MAKE MLT-UT NORMALIATION ###
    # ss2 = Hist2D('Lm_OPQ', 'MLT_OPQ', 
    #                 bins=[np.arange(2, 15), np.arange(0, 25)],
    #                 filterDict={'dos1rate':[0, 1E6]})
    # ss2.loop_data(simultaneous=False)
    # ss2.save_data(os.path.join(SAVE_DIR, 'ac6_L_MLT_bins_same_loc.csv'), 
    #               os.path.join(SAVE_DIR, 'ac6_L_MLT_norm_same_loc.csv'))

    # ### SCRIPT TO MAKE MLT-LON NORMALIZATION ####
    # ss = Hist2D('MLT_OPQ', 'lon', 
    #             bins=[np.arange(0, 25, 1), np.arange(-180, 181, 10)],
    #             filterDict={'dos1rate':[0, 1E6]})
    # ss.loop_data(simultaneous=False)
    # ss.save_data(os.path.join(SAVE_DIR, 'ac6_MLT_lon_bins_same_loc.csv'), 
    #             os.path.join(SAVE_DIR, 'ac6_MLT_lon_norm_same_loc.csv'))

    ### SCRIPT TO MAKE LAT-LON NORMALIZATION ####
    ss = Hist2D('lat', 'lon', 
                bins=[np.arange(-90, 91, 10), np.arange(-180, 181, 10)],
                filterDict={'dos1rate':[0, 1E6], 'flag':0})
    ss.loop_data(simultaneous=True)
    ss.save_data(os.path.join(SAVE_DIR, 'ac6_lat_lon_bins.csv'), 
                os.path.join(SAVE_DIR, 'ac6_lat_lon_norm.csv'))

    ### SCRIPT TO FIND THE EQUATORIAL NORMALIZATION ###
#    eq = Equatorial_Hist(np.arange(0, 2000, 25), 'Lm_OPQ', np.arange(4, 8.1),
#                        filterDict={'dos1rate':[0, 1E6]})
#                        # startDate=datetime(2015, 5, 26)
#    eq.loop_data()
#    eq.save_data('equatorial_test_norm.csv')

#    #eq.loop_data()
#    print(f'Run time = {time.time()-start_time} s')
