import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
from datetime import datetime
import dateutil.parser
import pathlib

import cdflib # https://github.com/MAVENSDC/cdflib

from ac6_curtains import dirs

class Load_ASI:
    def __init__(self, site, time):
        """
        Loads the All-Sky Imager images and calibration.
        """
        self.asi_dir = dirs.ASI_DIR
        self.site = site.lower()
        self.time = time

        if isinstance(self.time, str):
            # Convert to datetime object if passes a tring time
            self.time = dateutil.parser.parse(self.time)
        self.load_themis_asf()
        return

    def load_themis_asf(self):
        """
        Load the THEMIS ASF data and convert the time keys to datetime objects
        """
        file_name = f'thg_l1_asf_{self.site}_{self.time.strftime("%Y%m%d%H")}_v01.cdf'
        cdf_path = pathlib.Path(self.asi_dir, file_name)

        self.asi = cdflib.cdfread.CDF(cdf_path)
        # Convert time
        self.time = cdflib.cdfepoch().to_datetime(self.asi[f"thg_asf_{self.site}_epoch"][:], 
                                            to_np=True)
        # Copy images into another variable
        self.imgs = self.asi[f"thg_asf_{self.site}"]
        return self.asi, self.time, self.imgs

    def load_themis_cal(self):
        """
        Loads the THEMIS calibration file.
        """
        raise NotImplementedError

        return

    def plot_themis_asi_frame(self, t0, ax=None):
        """
        Plot a ASI frame with a time nearest to t0.
        """
        if ax is None:
            _, ax = plt.subplots()

        if isinstance(t0, str):
            # Convert to datetime object if passes a tring time
            t0 = dateutil.parser.parse(t0) 
        dt = self.time-t0
        dt_sec = np.abs([dt_i.total_seconds() for dt_i in dt])
        t0_nearest = self.time[np.argmin(dt_sec)]

        title_text = f'{self.site.upper()}\n{t0_nearest}'
        self.hi = ax.imshow(self.imgs[0], cmap="gray", origin="lower", 
                norm=matplotlib.colors.LogNorm(), interpolation="none")
        self.ht = ax.set_title(title_text, color="k")
        return t0_nearest

    def keys(self):
        """
        Gets the variable names from the ASI data.
        """
        if hasattr(self, self.asi):
            return self.asi.cdf_info()['zVariables']
        else:
            raise AttributeError('The ASI CDF file is probably not loaded.')
        return


site = 'GBAY'
time = datetime(2015, 7, 21, 3)

l = Load_ASI(site, time)

l.plot_themis_asi_frame(datetime(2015, 7, 21, 3, 0, 5, 65000))

plt.show()
# path = (f'/home/mike/research/ac6_curtains/data/asi/'
#         f'thg_l1_asf_{site.lower()}_{time.strftime("%Y%m%d%H")}_v01.cdf')

# data = cdflib.cdfread.CDF(path)
# # Get variables via data.cdf_info()['zVariables']
# time = cdflib.cdfepoch().to_datetime(data[f"thg_asf_{site}_epoch"][:], to_np=True)
# imgs = data[f"thg_asf_{site}"]

# fig, ax = plt.subplots()

# # Initialize the first frame.
# hi = ax.imshow(imgs[0], cmap="gray", origin="lower", 
#                 norm=matplotlib.colors.LogNorm(), interpolation="none")
# ht = ax.set_title('time', color="g")
