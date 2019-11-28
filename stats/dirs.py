import os

# Specify AC6 directories
BASE_DIR = '/home/mike/research/ac6_curtains/'
CATALOG_DIR = os.path.join(BASE_DIR, 'data/catalogs')
NORM_DIR = os.path.join(BASE_DIR, 'data/norm')
AC6_DATA_PATH = lambda sc_id: ('/home/mike/research/ac6/ac6{}/'
                                'ascii/level2'.format(sc_id))
PLOT_SAVE_DIR = os.path.join(BASE_DIR, 'plots', 'curtain_validation')