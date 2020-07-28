import pathlib

# Specify AC6 directories
BASE_DIR = pathlib.Path('/home/mike/research/ac6_curtains/')
CATALOG_DIR = pathlib.Path(BASE_DIR, 'data/catalogs')
NORM_DIR = pathlib.Path(BASE_DIR, 'data/norm')
AC6_DATA_PATH = lambda sc_id: (f'/home/mike/research/ac6/ac6{sc_id}/'
                                'ascii/level2')
AC6_MERGED_DATA_DIR = '/home/mike/research/ac6/merged/'
PLOT_SAVE_DIR = pathlib.Path(BASE_DIR, 'plots', 'curtain_validation')
ASI_DIR = BASE_DIR / 'data' / 'asi'
RBSP_DIR = lambda inst, sc_id: (f'/home/mike/research/rbsp/data/'
                                f'{inst.lower()}/rbsp{sc_id.lower()}')