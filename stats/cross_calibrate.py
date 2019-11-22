import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from datetime import datetime, timedelta
import dateutil.parser
import csv
import itertools
import scipy.signal
#import sys
import os 
import functools

#sys.path.insert(0, '/home/mike/research/mission_tools/ac6')
import mission_tools.ac6.read_ac_data as read_ac_data

class OccuranceRate:
    def __init__(self, sc_id, date, catV, catPath=None):
        """
        NAME: OccuranceRate
        USE:  Calculates the occurance rate of microbursts
              from a single spacecraft as a function of
              radiation belt pass. Radiation belt pass is
              defined as 4 < L < 8 (kwargs for the 
              radBeltIntervals method.) Radiation belt 
              interval indicies are stored in 
              self.intervals array(nPasses x 2). Occurance 
              rate for each pass is saved in self.rates
              array. Occurance rate is calculated by adding 
              up all of the microbursts widths for each pass
              (assuming a width function for which two are
              defined and are chosen by the mode kwarg of 
              the occurance_rate method) and then divided 
              by the pass time.
        INPUT: sc_id: spacecraft id
               date:  data to load data from
               catV:  microburst catalog version
               catPath: microburst catalog path (if not
                        specied, the __init__ method 
                        will use a default path)
        AUTHOR: Mykhaylo Shumko
        RETURNS: self.intervals: array(nPasses x 2) of 
                    radiation belt pass start/end indicies
                 self.rates: array(nPasses) of the 
                    microburst duty cycle between 0 and 1 
                    for each pass.
        MOD:     2018-10-23
        """
        self.sc_id = sc_id
        self.date = date
        self._load_sc_data()

        if catPath is None:
            catPath = ('/home/mike/research/ac6_microburst_scale_sizes'
                      '/data/microburst_catalogues')
        self._load_catalog(catPath, catV)
        return

    def radBeltIntervals(self, lbound=3, ubound=10, dawnFilter=False):
        """
        This method finds all of the radiation belt passes in a day 
        and separates it into start/end indicies.
        """
        # Find the start/end indicies of each rad belt pass.
        L = np.abs(self.data['Lm_OPQ'])

        # Filter by L and MLT if dawnFilter is True.
        if dawnFilter:
            MLT = self.data['MLT_OPQ']
            idL = np.where((L > lbound) & (L < ubound) & 
                            (MLT > 0) & (MLT < 12))[0]
        else:
            idL = np.where((L > lbound) & (L < ubound))[0]
            
        if len(idL) == 0: # If it is empty.
            raise ValueError('No valid radiation belt passes found')
        # Find breaks in the rad belt indicies to separate 
        # the passes.
        conv = np.convolve([1, -1], idL, mode = 'valid') - 1
        consecutiveFlag = np.where(conv != 0)[0] + 1
        startInd = np.insert(consecutiveFlag, 0, 0)
        endInd = np.insert(consecutiveFlag, 
                len(consecutiveFlag), len(idL)-1)
        # Save pass indicies to self.intervals.
        self.intervals = np.zeros((len(startInd), 2), dtype=int)
        for i, (i_s, i_e) in enumerate(zip(startInd, endInd)):
            self.intervals[i, :] = [idL[i_s], idL[i_e-1]]
        return 

    def occurance_rate(self, mode='static', **kwargs):
        """ 
        This method calculates the microburst occurance rates
        for each pass assuming a microburst width assumption.
        The width is either a fixed value in seconds or 
        another class method.
        """
        verbose = kwargs.get('verbose', False)
        testPlot = kwargs.get('testPlot', False)
        rel_height = kwargs.get('rel_height', 1)
        
        # Assume a static microburst width.
        if mode == 'static':
            w = kwargs.get('static_width', 0.5)
            width = functools.partial(self.static_width, w)
        # Assume a width that is calculated by 
        # scipy.signal.peak_width() after detrending.
        elif mode == 'prominence':
            width = functools.partial(self.prominence_width, 
                                      rel_height=rel_height, 
                                      testPlot=testPlot)
            
        # Convert the catalog datetimes to numeric to enable
        # faster time comparison.
        nDetTime = date2num(self.cat['dateTime'])
        self.rates = np.zeros(self.intervals.shape[0], 
                                dtype=float)
        self.durations = np.zeros(self.intervals.shape[0], 
                                dtype=float)
        if testPlot:
            _, self.ax = plt.subplots(2, sharex=True)
        # Loop over the passes.
        for i, (i_start, i_end) in enumerate(self.intervals):                
            # Find microbursts in ith pass
            startTime = self.data['dateTime'][i_start]
            endTime = self.data['dateTime'][i_end]
            idt = np.where((nDetTime > date2num(startTime)) & 
                            (nDetTime < date2num(endTime)))[0]       
            if verbose: 
                print('Processing pass', i, 'len(idt)=', len(idt))

            # determine the duration of each microburst in the
            # current pass
            for t_i in idt:
                self.rates[i] += width(self.cat['dateTime'][t_i])
            self.durations[i] = (endTime - startTime).total_seconds()
            if verbose:
                print('rates =', self.rates[i], 
                    'duration=', self.durations[i])
            self.rates[i] /= self.durations[i]
        return
        
    def static_width(self, width, t):
        """ Returns a constant width """
        return width
        
    def prominence_width(self, t, testPlot=True, rel_height=0.5,
                         window_width=1):
        """ 
        This method implements an algorithm to calculate the microburst 
        width based on its topological prominence.

        scipy.signal.find_peaks()
        """
        # Clip the data to the microburst time (with some window 
        # around it).
        lt = t - timedelta(seconds=window_width)
        ut = t + timedelta(seconds=window_width)
        indicies = np.where((self.data['dateTime'] > lt) & 
                           (self.data['dateTime'] < ut))[0]
        
        # detrend microburst time series
        iPeak = np.where(self.data['dateTime'][indicies] == t)[0]
        if len(iPeak) != 1: raise ValueError('0 or > 1 peaks found!')
        detrended = scipy.signal.detrend(self.data['dos1rate'][indicies])

        # Try to find the width of the center peak.
        try:
            width = scipy.signal.peak_widths(detrended, iPeak, 
                                            rel_height=0.5)
        # If it fails, search for nearest peak to the center time 
        # and try again.
        except ValueError as err:
            if 'is not a valid peak' in str(err):
                peaks, _ = scipy.signal.find_peaks(detrended, 
                                            prominence=None)
                iPeak = peaks[np.argmin(np.abs(peaks - len(indicies)//2))]
                width = scipy.signal.peak_widths(detrended, [iPeak],
                                                 rel_height=rel_height)
        # Make a test plot for debugging purposes.
        if testPlot:
            saveDir = '/home/mike/temp_plots'
            if not os.path.exists(saveDir):
                print('made dir', saveDir)
                os.makedirs(saveDir)
            self.ax[0].plot(self.data['dos1rate'][indicies])
            self.ax[0].plot(self.data['dos1rate'][indicies]-detrended)
            self.ax[1].plot(detrended)
            self.ax[1].hlines(*width[1:])
            self.ax[1].set_ylim(top=1.5*detrended[iPeak])
                
            plt.savefig(os.path.join(saveDir,
                        '{}_microburst_validation.png'.format(
                        t.replace(microsecond=0).isoformat()))
                        )
            for a in self.ax:
                a.cla()
        return 0.1*width[0][0]

    def _load_sc_data(self):
        """ Loads AC-6 10 Hz data """
        print('Loading AC6-{} data from {}'.format(
            self.sc_id.upper(), self.date.date()))
        self.data = read_ac_data.read_ac_data_wrapper(
                                self.sc_id, self.date)
        return

    def _load_catalog(self, catPath, v):
        """ 
        Loads the microburst catalog of version v spacecraft given 
        by self.sc_id. 
        """
        fPath = os.path.join(catPath, 
            'AC6{}_microbursts_v{}.txt'.format(self.sc_id.upper(), v))
        print('Loading catalog,', 'AC6{}_microbursts_v{}.txt'.format(
            self.sc_id.upper(), v))
        with open(fPath) as f:
            r = csv.reader(f)
            keys = next(r)
            rawData = np.array(list(r))
        self.cat = {}
        for (i, key) in enumerate(keys):
            self.cat[key] = rawData[:, i]

        # Now convert the times array(s) to datetime, 
        # and all others except 'burstType' to a float.
        timeKeys = itertools.filterfalse(
            lambda x:'dateTime' in x or 'burstType' in x, self.cat.keys())
        for key in timeKeys:
                self.cat[key] = self.cat[key].astype(float)
        for key in filter(lambda x: 'dateTime' in x, self.cat.keys()):
            self.cat[key] = np.array([dateutil.parser.parse(i) 
                for i in self.cat[key]])
        return

class CoincidenceRate:
    def __init__(self, date, catV, catPath=None):
        """
        NAME: CoincidenceRate
        USE:  Calculates the coincidence rate of microbursts
              from both spacecraft as a function of
              radiation belt pass. This classe uses the 
              methods from OccuranceRate class, so refer to
              those docs on how the occurance rate and 
              radiation belt passes are calculated.
        INPUT: date:  data to load data from
               catV:  microburst catalog version
               catPath: microburst catalog path (if not
                        specied, the __init__ method 
                        will use a default path)
        AUTHOR: Mykhaylo Shumko
        RETURNS: self.intervals: array(nPasses x X) of 
                    radiation belt pass start/end indicies
                    for both spacecraft
                 self.rates: array(nPasses) of the coincident 
                    microburst duty cycle between 0 and 1 
                    for each pass.
        MOD:     2018-10-23
        """
        # Load AC6-A data and microburst catalog
        self.occurA = OccuranceRate('A', date, catV, catPath=catPath)
        # Load AC6-B data and microburst catalog.
        self.occurB = OccuranceRate('B', date, catV, catPath=catPath)
        return
    
    def radBeltIntervals(self):
        """
        NAME:  radBeltIntervals
        USE:   This method calculates the radiation belt 
               passes from both spacecraft.
        INPUT: None
        AUTHOR: Mykhaylo Shumko
        RETURNS: self.passes - an array(nPasses x 4) where
                            the columns indicate, in order,
                            start/stop indicies for AC6A, 
                            followed by indicies for AC6B 
                            for the matching passes.
        MOD:     2018-10-25
        """
        # Calculate rad belt pass intervals from each 
        # spacecraft.
        self.occurA.radBeltIntervals()
        self.occurB.radBeltIntervals()

        self.passes = np.zeros((0, 4), dtype=object)

        tAn = date2num(self.occurA.data['dateTime'][self.occurA.intervals])
        tBn = date2num(self.occurB.data['dateTime'][self.occurB.intervals])

        # Find matching indicies by looping over start times of AC6A 
        # pass interval array
        for i, t in enumerate(tAn[:, 0]):
            # For each start time in AC6A, look for a corresponding 
            # start times in AC6B data within 5 minutes
            dt = np.abs(tBn[:, 0] - t)
            idmin = np.argmin(dt)
            if dt[idmin] < sec2day(5*60):
                newRow = np.concatenate((self.occurA.intervals[i], 
                                        self.occurB.intervals[idmin]
                                        ))
                self.passes = np.vstack((self.passes, newRow))
        return

    def sortBursts(self, ccAbsThresh=0.9, ccSpatialThresh=0.1, 
                    ccWindow=0.5, ccOverlap=2, testPlots=False, 
                    testData=False):
        """
        NAME:   sortBursts
        USE:    This method loops through every pass for which there
                is data from both spacecraft, and for each detecton
                made by either AC6A or AC6B, checks if there is a 
                correlated peak at the same time, and in the same
                position. Uses cross-correlations (CC)
        INPUT:  ccAbsThresh = 0.8 - absolute CC threshold for detection.
                ccSpatialThresh = 0.1 - relative threshold of microburst
                                 to curtain e.g. a microburst is only 
                                 identified if its CC threshold is above
                                 ccAbsThresh and is ccSpatialThresh above 
                                 the spatial CC.
                ccWindow = 2 -   CC window width.
                ccOverLap = 2  - CC overlap, to account for statistical 
                                 variation for short-duration 
                                 microbursts and small timing offsets
                                 (little to none expected).
        AUTHOR: Mykhaylo Shumko
        RETURNS: self.bursts - array(nBursts, 6) where the columns
                               are: 
                               iPass = rad. belt pass number
                               t0 =  center time for microburst 
                                      observation (time-aligned).
                               t_sA = center time for AC6A shifted
                               t_sB = center time for AC6B shifted 
                               tCC  = temporal CC 
                               sCC  = spatial CC 
        MOD:     2018-10-29
        """
        self.bursts = np.zeros((0, 6), dtype=object) 
        self.ccWindow = ccWindow 
        
        # Convert detection times to numbers for easy comparison.
        catAtimes = date2num(self.occurA.cat['dateTime'])
        catBtimes = date2num(self.occurB.cat['dateTime'])

        # Loop over passes
        for i, (passAi, passAf, passBi, passBf) in enumerate(self.passes):
            print('\nProcessing pass', i)
            # Find ime range for this pass. 
            aRange = date2num([self.occurA.data['dateTime'][passAi],
                             self.occurA.data['dateTime'][passAf]])
            bRange = date2num([self.occurB.data['dateTime'][passBi],
                             self.occurB.data['dateTime'][passBf]])
            # Find all bursts separately detected on AC6A/B for 
            # this pass
            burstsA = np.where((catAtimes > aRange[0]) & 
                            (catAtimes < aRange[1]))[0]
            burstsB = np.where((catBtimes > bRange[0]) & 
                            (catBtimes < bRange[1]))[0]
            print('Found {} AC6A bursts and {} AC6B bursts'.format(
                len(burstsA), len(burstsB)))
            
            # Loop over bursts from AC6A.              
            for bA in burstsA:
                # Calculate the shifted and unshifted time bounds.
                idtA, idtB, idtA_shifted, idtB_shifted, t0, t_sA, t_sB = \
                    self._get_index_bounds('A', bA, ccWindow, ccOverlap)
                # CC in time and space
                tCC = self.CC(idtA, idtB)
                sCC = self.CC(idtA_shifted, idtB_shifted)
                # Save data
                line = [i, t0, t_sA, t_sB, tCC, sCC]
                self.bursts = np.vstack((self.bursts, line))
                if testPlots:
                    args = (idtA, idtB, idtA_shifted, idtB_shifted, t0, t_sA, t_sB)
                    self._index_test_plot('A', args)
                if testData:
                    self.save_training_data('a', idtA, ccWindow)

            # Loop over bursts from AC6B.              
            for bB in burstsB:
                # Calculate the shifted and unshifted time bounds.
                idtA, idtB, idtA_shifted, idtB_shifted, t0, t_sA, t_sB = \
                    self._get_index_bounds('B', bB, ccWindow, ccOverlap)
                # CC in time and space
                tCC = self.CC(idtA, idtB)
                sCC = self.CC(idtA_shifted, idtB_shifted)
                # Save data
                line = [i, t0, t_sA, t_sB, tCC, sCC]
                self.bursts = np.vstack((self.bursts, line))
                if testPlots:
                    args = (idtA, idtB, idtA_shifted, idtB_shifted, t0, t_sA, t_sB)
                    self._index_test_plot('B', args)
                if testData:
                    self.save_training_data('b', idtB, ccWindow)

        self.sortArrays() # Sort detections by time
        self.findMicrobursts(ccAbsThresh, ccSpatialThresh) # Apply theshold CC test
        self.removeDuplicates() # Remove duplicate events
        return
        
    def _get_index_bounds(self, sc_id, i, ccWindow, ccOverlap):
        """ 
        This method calculates the time aligned and space-aligned
        indicies for cross-correlation. As for the time shifts,
        refer to AC6 data README which specifies that the 
        Lag_In_Track == t_A - t_B.

        OUTPUT:
            first two outputs - time-aligned indicies for 
                                AC6A and AC6B, respectfully.
            second two outputs - space-aligned indicies for 
                                AC6A and AC6B, respectfully.
            last three outputs - time-aligned time, shifted 
                                time for AC6A (same as 
                                time-aligned time if sc_id == 'A')
                                and last output is shifted time
                                for AC6B which will be the shifted 
                                center time if sc_id == A.
        """
        dt = timedelta(seconds=ccWindow/2)
        overlapW = timedelta(seconds=ccOverlap/20)
        # Find the correct center time from either AC6A or B.
        if sc_id.upper() == 'A':
            t0 = self.occurA.cat['dateTime'][i]
        else:
            t0 = self.occurB.cat['dateTime'][i]

        #### Get temporally-aligned indicies ######
        idtA = np.where(  (self.occurA.data['dateTime'] >= t0-dt) 
                        & (self.occurA.data['dateTime'] < t0+dt)
                        & (self.occurA.data['dos1rate'] != -1E31))[0]
        # The overlapW windens the index array to accomidate a CC lag.
        idtB = np.where(  (self.occurB.data['dateTime'] >= t0-dt-overlapW) 
                        & (self.occurB.data['dateTime'] < t0+dt+overlapW)
                        & (self.occurB.data['dos1rate'] != -1E31))[0]

        ####### Get spatially aligned indicies ######
        # If sc_id is 'A', then, the spatial indicies are the 
        # same for AC6A, but shifted for AC6B. Since 
        #               lag = t_A - t_B
        # this implies that t_B_shifted = t_A - lag. If sc_id 
        # is 'B', then everything is opposite, i.e. 
        # t_A_shifted = t_B + lag
        if sc_id.upper() == 'A':
            idtA_shifted = idtA 
            timeLag = timedelta(seconds=self.occurA.cat['Lag_In_Track'][i])
            idtB_shifted = np.where(
                (self.occurB.data['dateTime'] >= t0-dt+timeLag-overlapW) &  
                (self.occurB.data['dateTime'] < t0+dt+timeLag+overlapW) &
                (self.occurB.data['dos1rate'] != -1E31)
                )[0]
            t_sA = t0
            t_sB = t0 + timeLag
        else:
            idtB_shifted = idtB 
            timeLag = timedelta(seconds=self.occurB.cat['Lag_In_Track'][i])
            idtA_shifted = np.where(
                (self.occurA.data['dateTime'] >= t0-dt-timeLag-overlapW) &  
                (self.occurA.data['dateTime'] < t0+dt-timeLag+overlapW) &
                (self.occurA.data['dos1rate'] != -1E31)
                )[0]
            t_sA = t0 - timeLag
            t_sB = t0
        return idtA, idtB, idtA_shifted, idtB_shifted, t0, t_sA, t_sB
        
    def sortArrays(self):
        """ 
        This method sorts self.bursts. The sort is done by time, 
        and the key function moves around all of the columns to 
        the appropriate place.
        """
        self.bursts = np.array(sorted(self.bursts, key=lambda x:x[1]))
        return 

    def removeDuplicates(self, tThresh=0.3):
        """
        This method goes through self.bursts and removes events that were
        within a small threshold of each other.
        """
        # Calculate time differences between microbursts
        tz = zip(self.microbursts[1:, 1] - self.microbursts[:-1, 1])
        dt = np.array([t[0].total_seconds() for t in tz])
        # Identify indicies when time differces were less than tThresh
        idt = np.where(dt < tThresh)[0]
        # Remove those entries.
        self.microbursts = np.delete(self.microbursts, idt, axis=0)
        return

    def findMicrobursts(self, absCC, spatialCCthresh):
        """
        This method goes through self.bursts and removes events that were
        within a small threshold of each other.
        """
        self.microbursts = np.zeros((0, 6), dtype=object) 

        # Loop over bursts and save the events into self.microbursts
        # that satisfy the CC threshold criteria.
        for line in self.bursts:
            tCC = line[-2]
            sCC = line[-1]
            # line contains (i, t0, t_sA, t_sB, tCC, sCC)
            if (tCC >= absCC) and (tCC - sCC >= spatialCCthresh):
                self.microbursts = np.vstack((self.microbursts, line))
        return

    def calcMicroburstRate(self, widthMode='static', save_data=True, width=0.5):
        """ 
        This method calculates the microburst occurance rate 
        from the self.microbursts array.

        if save==True, then the occurance rate ratios are saved
        to a file.
        """
        self.occurA.occurance_rate(mode=widthMode)
        self.occurB.occurance_rate(mode=widthMode)

        # Calculate the chance occurance rate.
        # Note that self.occurA/ objects calculate
        # occurance rates for all passes on that day.
        # Here we nee to match up the individual 
        # occurance rates to self.passes (so we can
        # caluclate chance rates for passes that 
        # both sc took data for).
        self.chanceRates = np.zeros(self.passes.shape[0])
        self.microburstRate = np.zeros(self.passes.shape[0])
        self.durations = np.zeros_like(self.chanceRates)
        self.microburstOccurance = np.zeros_like(self.chanceRates)
        self.rateRatio = np.zeros_like(self.chanceRates)

        for i, (passAi, _, passBi, _) in enumerate(self.passes):
            # Assosiate individual passes from occurA/B.intervals to
            # the shared passes saved in self.passes
            idpA = np.where(self.occurA.intervals[:,0] == passAi)[0]
            idpB = np.where(self.occurB.intervals[:,0] == passBi)[0]
            # If one of the occurance rates is 0, avoid infintiies by 
            # squaring the non-zero value. Be definition, one of these values
            # will be non-zero.
            if self.occurA.rates[idpA]*self.occurB.rates[idpB] != 0:
                self.chanceRates[i] = (self.occurA.rates[idpA]*
                                       self.occurB.rates[idpB])
            else:
                self.chanceRates[i] = np.max([self.occurA.rates[idpA],
                                              self.occurB.rates[idpB]])**2
            
            self.durations[i] = np.mean([self.occurA.durations[idpA], 
                                        self.occurB.durations[idpB]])

        # Calculate the microburst occurance rates.
        for i, (cR, dt) in enumerate(zip(self.chanceRates, self.durations)):
            # Find all of the microbursts in ith pass.
            idx = np.where(self.microbursts[:, 0] == i)[0]
            # Calculate microburst occurance rate.
            self.microburstRate[i] = len(idx)*width/dt
            self.rateRatio[i] = self.microburstRate[i]/cR

        if save_data:
            self.save_occurance_data()
        return

    def save_occurance_data(self, path=None):
        """ 
        This method saves the occurance data to a csv file.
        The data that is saved are: pass start time (from AC6A), total 
        separation, microburst occurance rate, chance 
        occurance rate, occurance rate ratio
        """
        if not path:
            path = os.path.join('./../data', 
                                'coincident_microbursts_catalogues',
                                'microburst_catalog.csv' )
        if not os.path.exists(path):
            make_header = True
        else:
            make_header = False
        #     raise OSError('Microburst catalog file already exists.')
        # Make the save data array
        #save_data = np.zeros((0, 5), dtype=object)
        #np.hstack((self.microbursts, line))
        pass_start_times = [t.isoformat() for t in 
                    self.occurA.data['dateTime'][self.passes[:, 0].astype(int)]]
        self.data = np.vstack(
                        (pass_start_times,
                         self.occurA.data['Dist_Total'][self.passes[:, 0].astype(int)],
                         self.microburstRate,
                         self.chanceRates,
                         self.rateRatio
                        ))

        # Save the data
        with open(path, 'a') as f:
            w = csv.writer(f)
            if make_header: # Make the header if file is opened for first time.
                w.writerow(['pass_start', 'total_sep', 'uburst_rate',
                          'chance_rate', 'rate_ratio'])
            w.writerows(self.data.T)
        return
        

    def CC(self, iA, iB):
        """ 
        This method calculates the normalized cross-correlation 
        between two AC6 time series indexed by iA and iB.
        """
        norm = np.sqrt(len(self.occurA.data['dos1rate'][iA])*\
                       len(self.occurB.data['dos1rate'][iB])*\
                       np.var(self.occurA.data['dos1rate'][iA])*\
                       np.var(self.occurB.data['dos1rate'][iB])) 
        # Mean subtraction.
        x = (self.occurA.data['dos1rate'][iA] - 
            self.occurA.data['dos1rate'][iA].mean() )
        y = (self.occurB.data['dos1rate'][iB] - 
            self.occurB.data['dos1rate'][iB].mean() )
        # Cross-correlate
        ccArr = np.correlate(x, y, mode='valid')
        # Normalization
        ccArr /= norm
        return max(ccArr)

    def save_training_data(self, sc_id, idx, ccWindow):
        """
        This method saves the microburst detection to a csv file.
        First coolumn is the time, and the other columns are 
        dos1rate counts for values in width. 
        """
        savePath = './../data/train/ac6{}_training_data.csv'.format(
                    sc_id.lower())
        N = int(ccWindow*10-1)
        with open(savePath, 'a') as f:
            w = csv.writer(f)
            if sc_id.upper() == 'A':
                w.writerow(np.concatenate((
                            [self.occurA.data['dateTime'][idx[len(idx)//2]]], 
                            self.occurA.data['dos1rate'][idx[:N]]
                                        )))
            else:
                w.writerow(np.concatenate((
                            [self.occurB.data['dateTime'][idx[len(idx)//2]]], 
                            self.occurB.data['dos1rate'][idx[:N]]
                                        )))
        return
    #################################################################
    #################### TEST METHODS ###############################
    #################################################################
    def _index_test_plot(self, sc_id, indexArgs, saveDir='/home/mike/temp_plots'):
        """ 
        This is a test method to check if the CC-indicies are correctly 
        picked. 
        """
        idtA, idtB, idtA_shifted, idtB_shifted, t0, t_sA, t_sB = indexArgs
        print('In AC6-{} loop. Making test plot for time {}'.format(sc_id, t0))
        _, ax = plt.subplots(3, figsize=(9, 10))  

        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            print('Made plot directory:', saveDir)
        savePath = os.path.join(saveDir, '{}_AC6{}_microburst_validation.png'.format(
                    t0.replace(microsecond=0), sc_id.upper()))
        ax[0].plot(self.occurA.data['dateTime'][idtA], 
                    self.occurA.data['dos1rate'][idtA], c='r', label='AC6A')
        ax[0].plot(self.occurB.data['dateTime'][idtB], 
                    self.occurB.data['dos1rate'][idtB], c='b', label='AC6B')
        ax[0].set_title('AC6{} validation'.format(sc_id))
        ax[0].set_ylabel('Unshifted')
        ax[0].legend(loc=1)
        ax[1].plot(self.occurA.data['dateTime'][idtA_shifted], 
                    self.occurA.data['dos1rate'][idtA_shifted], c='r')
        ax[1].axvline(t_sA, c='r')
        ax[2].plot(self.occurB.data['dateTime'][idtB_shifted], 
                    self.occurB.data['dos1rate'][idtB_shifted], c='b')
        ax[2].axvline(t_sB, c='b')
        ax[1].set_ylabel('Shifted')
        ax[2].set_ylabel('Shifted')
        plt.savefig(savePath, dpi=200)
        plt.close()
        return
    
    def test_plots(self, plot_window=5, saveDir='/home/mike/temp_plots'):
        """ 
        This method plots detections and their temporal and spatial
        cross-correlations.
        """
        from matplotlib.backends.backend_pdf import PdfPages
        _, ax = plt.subplots(4, sharex=False, figsize=(8, 10))  

        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            print('Made plot directory:', saveDir)

        # Iterate over the microburst detections.
        savePath = os.path.join(saveDir, '{}_microburst_validation.pdf'.format(
                    datetime.now().date()))
        # Plot microburst detections if they have been sorted already.
        if hasattr(self, 'microbursts'):
            arr = self.microbursts
        else:
            arr = self.bursts

        with PdfPages(savePath) as pdf:
            for row in arr:
                self._make_time_plots(ax[:2], plot_window, row)
                self._make_space_plots(ax[2:], plot_window, row)
                #self._test_plots_space(ax[1], tA, tB, lag, cc, window)
                pdf.savefig()    
                for a in ax:
                    a.cla()
            plt.close()
        # # Iterate over the curtain detections.            
        # savePath = os.path.join(saveDir, '{}_curtain_validation.pdf'.format(
        #             datetime.now().date()))
        # with PdfPages(savePath) as pdf:
        #     for (tA, tB, _, cc) in self.sBurst:
        #         self._test_plots_curtains(ax, tA, tB, cc, window)
        #         pdf.savefig()    
        #         for a in ax:
        #             a.cla()
        return

    def _make_time_plots(self, ax, windowWidth, burstInfo, fill_between=True):
        """ 
        This method handles the plotting of the time-aligned timeseries.
        """
        _, t0, t_sA, t_sB, tCC, _ = burstInfo
        pltHalfWindow = timedelta(seconds=windowWidth/2)
        ccHalfWindow = timedelta(seconds=self.ccWindow/2)
        # Find indicies for time-aligned plots in ax[0]
        idtA = np.where(
            (self.occurA.data['dateTime'] > t0-pltHalfWindow) &  
            (self.occurA.data['dateTime'] < t0+pltHalfWindow) &
            (self.occurA.data['dos1rate'] != -1E31)
            )[0]     
        idtB = np.where(
            (self.occurB.data['dateTime'] > t0-pltHalfWindow) &  
            (self.occurB.data['dateTime'] < t0+pltHalfWindow) &
            (self.occurB.data['dos1rate'] != -1E31)
            )[0]
        # Find indicies from cross-correlation
        idCCA = np.where(
            (self.occurA.data['dateTime'] > t0-ccHalfWindow) &  
            (self.occurA.data['dateTime'] < t0+ccHalfWindow) &
            (self.occurA.data['dos1rate'] != -1E31)
            )[0]     
        idCCB = np.where(
            (self.occurB.data['dateTime'] > t0-ccHalfWindow) &  
            (self.occurB.data['dateTime'] < t0+ccHalfWindow) &
            (self.occurB.data['dos1rate'] != -1E31)
            )[0]

        if hasattr(ax, '__len__'):
            # Plot the mean-subtracted values
            meanA = self.occurA.data['dos1rate'][idCCA].mean()
            meanB = self.occurB.data['dos1rate'][idCCB].mean()

            ax[1].plot(self.occurA.data['dateTime'][idtA], 
                    self.occurA.data['dos1rate'][idtA]-meanA, 
                    'r', label='AC6A')
            ax[1].plot(self.occurB.data['dateTime'][idtB], 
                        self.occurB.data['dos1rate'][idtB]-meanB, 
                        'b', label='AC6B')
            # My implementation of plt.fill_between but for this plot.
            self.fill_between(self.occurA.data['dateTime'], 
                              self.occurB.data['dateTime'], 
                              idCCA, idCCB, ax[1])  
            ax[1].set_ylabel('Mean subtracted')
        else:
            ax = [ax]

        ax[0].plot(self.occurA.data['dateTime'][idtA], 
                    self.occurA.data['dos1rate'][idtA], 
                    'r', label='AC6A')
        ax[0].plot(self.occurB.data['dateTime'][idtB], 
                    self.occurB.data['dos1rate'][idtB], 
                    'b', label='AC6B')
        ax[0].text(0, 0.9, 'Time_CC={:.2f}'.format(tCC), 
                    transform=ax[0].transAxes)
        ax[0].set(title='{} | AC6 microburst validation'
                    ''.format(t0.replace(microsecond=0).isoformat()))
        ax[0].axvline(t0)
        ax[0].legend(loc=1)
        return

    def _make_space_plots(self, ax, windowWidth, burstInfo):
        """ 
        This method handles the plotting of the time and space
        aligned timeseries.
        """
        _, t0, t_sA, t_sB, _, sCC = burstInfo
        pltHalfWindow = timedelta(seconds=windowWidth/2)
        ccHalfWindow = timedelta(seconds=self.ccWindow/2)
        
        # Find indicies for time-aligned plots in ax[0]
        idtA = np.where(
            (self.occurA.data['dateTime'] > t_sA-pltHalfWindow) &  
            (self.occurA.data['dateTime'] < t_sA+pltHalfWindow) &
            (self.occurA.data['dos1rate'] != -1E31)
            )[0]     
        idtB = np.where(
            (self.occurB.data['dateTime'] > t_sB-pltHalfWindow) &  
            (self.occurB.data['dateTime'] < t_sB+pltHalfWindow) &
            (self.occurB.data['dos1rate'] != -1E31)
            )[0]
        # Find cross-correlation indicies
        idCCA = np.where(
            (self.occurA.data['dateTime'] > t_sA-ccHalfWindow) &  
            (self.occurA.data['dateTime'] < t_sA+ccHalfWindow) &
            (self.occurA.data['dos1rate'] != -1E31)
            )[0]     
        idCCB = np.where(
            (self.occurB.data['dateTime'] > t_sB-ccHalfWindow) &  
            (self.occurB.data['dateTime'] < t_sB+ccHalfWindow) &
            (self.occurB.data['dos1rate'] != -1E31)
            )[0]            

        time_lag = self.occurA.data['Lag_In_Track'][idtA[0]]
        shifted_times = np.array([t - timedelta(seconds=time_lag) 
                    for t in self.occurB.data['dateTime']])
        shifted_times_flt = [t - timedelta(seconds=time_lag) 
                    for t in self.occurB.data['dateTime'][idtB]]

        if hasattr(ax, '__len__'):
            # Plot the mean-subtracted values
            meanA = self.occurA.data['dos1rate'][idCCA].mean()
            meanB = self.occurB.data['dos1rate'][idCCB].mean()
            
            ax[1].plot(self.occurA.data['dateTime'][idtA], 
                    self.occurA.data['dos1rate'][idtA]-meanA, 
                    'r', label='AC6A')
            ax[1].plot(shifted_times_flt, 
                        self.occurB.data['dos1rate'][idtB]-meanB, 
                        'b', label='AC6B')
            # My implementation of plt.fill_between but for this plot.
            self.fill_between(self.occurA.data['dateTime'], 
                              shifted_times, 
                              idCCA, idCCB, ax[1])  
            ax[1].set_ylabel('Mean subtracted')
        else:
            ax = [ax]
                        
        ax[0].plot(self.occurA.data['dateTime'][idtA], 
                    self.occurA.data['dos1rate'][idtA], 
                    'r', label='AC6A')
        ax[0].plot(shifted_times_flt, 
                    self.occurB.data['dos1rate'][idtB], 
                    'b', label='AC6B')
        ax[0].text(0, 1, 'space_CC={:.2f}\nAC6-B shift={:.2f}'.format(sCC, time_lag), 
                    transform=ax[0].transAxes, va='top')
        #ax.axvline(t_sA, c='r')
        #ax.axvline(t_sB, c='b')
        #ax.legend(loc=1)
        return

    def fill_between(self, timeA, timeB, idCCA, idCCB, ax, alpha=0.25):
        """
        This method fills in the mean-subtracted dos1rate 
        timeseries for CC indicies (given by idCCA and idCCB).
        ax is the subplot to plot to.
        """
        meanA = self.occurA.data['dos1rate'][idCCA].mean()
        meanB = self.occurB.data['dos1rate'][idCCB].mean()
        ax.fill_between(timeA[idCCA], 0, 
                                self.occurA.data['dos1rate'][idCCA]-meanA, 
                                where=self.occurA.data['dos1rate'][idCCA]-meanA > 0, 
                                facecolor='red', interpolate=True, alpha=alpha)
        ax.fill_between(timeA[idCCA], 0, 
                            self.occurA.data['dos1rate'][idCCA]-meanA, 
                            where=self.occurA.data['dos1rate'][idCCA]-meanA < 0, 
                            facecolor='blue', interpolate=True, alpha=alpha)
        ax.fill_between(timeB[idCCB], 0, 
                                self.occurB.data['dos1rate'][idCCB]-meanB, 
                                where=self.occurB.data['dos1rate'][idCCB]-meanB > 0, 
                                facecolor='red', interpolate=True, alpha=alpha)
        ax.fill_between(timeB[idCCB], 0, 
                                self.occurB.data['dos1rate'][idCCB]-meanB, 
                                where=self.occurB.data['dos1rate'][idCCB]-meanB < 0, 
                                facecolor='blue', interpolate=True, alpha=alpha)
        return


def sec2day(s):
    """ Convert seconds to fraction of a day."""
    return s/86400

if __name__ == '__main__':
    #cr = CoincidenceRate(datetime(2016, 10, 14), 3)
    #cr = CoincidenceRate(datetime(2017, 1, 11), 3)
    cr = CoincidenceRate(datetime(2015, 5, 6), 3)
    cr.radBeltIntervals()
    cr.sortBursts(testData=False)
    #cr.calcMicroburstRate()
    cr.test_plots()

    # # This code section loops over all of the AC6 data to 
    # # generate the occurance data data.
    # dateRange = [datetime(2014, 6, 20), datetime(2017, 8, 1)]
    # dDays = (dateRange[1] - dateRange[0]).days
    # dates = [dateRange[0] + timedelta(days=i) for i in range(dDays)]
    # for date in dates:
    #     try:   
    #         print('Loading data from {}'.format(date))
    #         cr = CoincidenceRate(date, 3)
    #     except AssertionError as err:
    #         if ( ('None or > 1 AC6 files found' in str(err)) or
    #             ('File is empty!' in str(err)) ):
    #             continue
    #         else:
    #             raise
    #     try:
    #         cr.radBeltIntervals()
    #     except ValueError as err:
    #         if str(err) == 'No valid radiation belt passes found':
    #             continue
    #         else:
    #             raise
    #     cr.sortBursts(testData=False)
    #     cr.calcMicroburstRate()
