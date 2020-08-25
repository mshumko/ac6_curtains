# This file includes scripts calculates the normalization parameters for the 
# micorburst scale size distributions.

# from datetime import datetime, timedelta
# from matplotlib.dates import date2num, num2date
# import csv
import numpy as np
import os

# import pandas as pd

# # Import personal libraries
# import mission_tools.ac6.read_ac_data as read_ac_data
# import IRBEM
import time

import norm
import dirs

### SCRIPT TO MAKE "Dist_Total" NORMALIZATION ###
# start_time = time.time()
# s=Hist1D(d=np.arange(0, 501, 1), 
#             filterDict={'dos1rate':[0, 1E6], 
#                         'Lm_OPQ':[4, 8]})
# s.loop_data()
# s.save_data(os.path.join(dirs.NORM_DIR, 'ac6_norm_all_1km_bins.csv'))

# bin_width = 5
# bin_offset = 0
# L_array = [4, 5, 6, 7, 8] #[4, 8]
# for L_lower, L_upper in zip(L_array[:-1], L_array[1:]):
#     ss2=Hist1D(d=np.arange(bin_offset, 501, bin_width), 
#                 filterDict={'dos1rate':[0, 1E6], 
#                             'Lm_OPQ':[L_lower, L_upper]})
#     ss2.loop_data()
#     ss2.save_data(os.path.join(dirs.NORM_DIR, 
#             f'ac6_norm_{L_lower}_L_{L_upper}'
#             f'_{bin_width}km_bins_offset.csv'))
#     print('Run time =', time.time()-start_time, 's')
# print('Norm.py ran in :{} s'.format((datetime.now()-st).total_seconds()))

### SCRIPT TO MAKE L-dependent "Dst_Total" NORMALIZATION ###
# st = datetime.now()
# L = [3, 4, 5, 6, 7]
# for (lL, uL) in zip(L[:-1], L[1:]):
#     ss=hist.Hist1D(filterDict={'Lm_OPQ':[lL, uL]})
#     ss.loop_data()
#     ss.save_data(os.path.join(dirs.NORM_DIR, 'ac6_norm_{}_L_{}.csv'.format(lL, uL)))
# print('Norm.py ran in :{} s'.format((datetime.now()-st).total_seconds()))

### SCRIPT TO MAKE L-MLT NORMALIATION ###
# ss2 = norm.Hist2D('Lm_OPQ', 'MLT_OPQ', 
#                     bins=[np.arange(1, 16), np.arange(0, 25)],
#                     filterDict={'flag':0})
# ss2.loop_data(simultaneous=False)
# ss2.save_data(os.path.join(dirs.NORM_DIR, 'ac6_L_MLT_bins_same_loc.csv'), 
#                 os.path.join(dirs.NORM_DIR, 'ac6_L_MLT_norm_same_loc.csv'))

### SCRIPT TO MAKE MLT-UT NORMALIATION ###
# ss2 = norm.Hist2D('Lm_OPQ', 'MLT_OPQ', 
#                 bins=[np.arange(2, 15), np.arange(0, 25)],
#                 filterDict={'dos1rate':[0, 1E6]})
# ss2.loop_data(simultaneous=False)
# ss2.save_data(os.path.join(dirs.NORM_DIR, 'ac6_L_MLT_bins_same_loc.csv'), 
#               os.path.join(dirs.NORM_DIR, 'ac6_L_MLT_norm_same_loc.csv'))

# ### SCRIPT TO MAKE MLT-LON NORMALIZATION ####
# ss = norm.Hist2D('MLT_OPQ', 'lon', 
#             bins=[np.arange(0, 25, 1), np.arange(-180, 181, 10)],
#             filterDict={'dos1rate':[0, 1E6]})
# ss.loop_data(simultaneous=False)
# ss.save_data(os.path.join(dirs.NORM_DIR, 'ac6_MLT_lon_bins_same_loc.csv'), 
#             os.path.join(dirs.NORM_DIR, 'ac6_MLT_lon_norm_same_loc.csv'))

### SCRIPT TO MAKE LAT-LON NORMALIZATION ####
ss = norm.Hist2D('lat', 'lon', 
            bins=[np.arange(-90, 91, 10), np.arange(-180, 181, 10)],
            filterDict={'flag':0})
ss.loop_data(simultaneous=True)
ss.save_data(os.path.join(dirs.NORM_DIR, 'ac6_lat_lon_bins.csv'), 
            os.path.join(dirs.NORM_DIR, 'ac6_lat_lon_norm.csv'))

### SCRIPT TO FIND THE EQUATORIAL NORMALIZATION ###
# eq = norm.Equatorial_Hist(np.arange(0, 2000, 25), 'Lm_OPQ', np.arange(4, 8.1),
#                     filterDict={'dos1rate':[0, 1E6]})
#                     # startDate=datetime(2015, 5, 26)
# eq.loop_data()
# eq.save_data('equatorial_test_norm.csv')

#eq.loop_data()
