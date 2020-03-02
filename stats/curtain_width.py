import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, '/home/mike/research/ac6_curtains/detect')
import dirs
import detect_daily

catalog_name = f'AC6_curtains_baseline_method_sorted_v0.txt'

cat = pd.read_csv(os.path.join(dirs.CATALOG_DIR, catalog_name))
cat.dateTime = pd.to_datetime(cat.dateTime)

peak_width_s = 5

class CurtainWidth:
    def __init__():

        return

    def find_peaks_loop():

        return

    def find_peaks():
        
        return



### OLD ###

# def _find_peaks(self, iA, iB, sample_thresh=1, peak_kwargs={}, smooth=3):
#     """
#     This method calls scipy.signal.find_peaks to attempt to
#     find a peak within time_thresh of the center of the index
#     array, iA and iB for indices from sc A and B, respectively.
#     """
#     if smooth > 1:
#         countsA = np.convolve(np.ones(smooth)/smooth, self.tenHzA['dos1rate'][iA], 
#             mode='same')
#         countsB = np.convolve(np.ones(smooth)/smooth, self.tenHzB['dos1rate'][iB], 
#             mode='same')
#     # Find the peaks
#     peaksA, propertiesA = scipy.signal.find_peaks(countsA, 
#             **peak_kwargs, width=(None, None))
#     peaksB, propertiesB = scipy.signal.find_peaks(countsB, 
#             **peak_kwargs, width=(None, None))
#     # The template sets just contain the indicies of iA and iB that 
#     # are at the center of iA and iB +/- sample_thresh
#     peak_template_A = set(np.arange(
#                         len(iA)//2-sample_thresh, len(iA)//2+sample_thresh+1
#                         ))
#     peak_template_B = set(np.arange(
#                         len(iB)//2-sample_thresh, len(iB)//2+sample_thresh+1
#                         ))
#     # Now find if there is a peak at or near the center.
#     peak_df_A = pd.DataFrame({'ipeak':peaksA, 'width':propertiesA['widths']})
#     peak_df_B = pd.DataFrame({'ipeak':peaksB, 'width':propertiesB['widths']})
#     valid_peaks_A = peak_df_A[peak_df_A['ipeak'].isin(peak_template_A)]
#     valid_peaks_B = peak_df_B[peak_df_B['ipeak'].isin(peak_template_B)]
#     # If only one peak was found near the center, return the width_A/B. If 
#     # none or more than 1 peaks were found, then return np.nan.
#     if len(valid_peaks_A) == 1:
#         width_A = valid_peaks_A.iloc[0, 1]/10
#     else:
#         width_A = np.nan
#     if len(valid_peaks_B) == 1:
#         width_B = valid_peaks_B.iloc[0, 1]/10
#     else:
#         width_B = np.nan
#     return width_A, width_B