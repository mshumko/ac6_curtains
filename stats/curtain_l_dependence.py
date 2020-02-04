import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import dirs

df = pd.read_csv(os.path.join(dirs.CATALOG_DIR, 
'AC6_curtains_sorted_v8.txt'))

df_morning = df[(df.MLT_OPQ > 7) & (df.MLT_OPQ < 13)]
df_night = df[(df.MLT_OPQ > 18) & (df.MLT_OPQ < 24)]

bins = np.arange(2, 14)
plt.hist(df.Lm_OPQ, bins=bins, color='g', alpha=0.4, label='all')
plt.hist(df_morning.Lm_OPQ, bins=bins, color='r', alpha=0.4, label='7-13 MLT')
plt.hist(df_night.Lm_OPQ, bins=bins, color='b', alpha=0.4, label='18-24 MLT')
plt.legend()

plt.xlabel('Lm_OPQ')
plt.ylabel('Number of curtains')
plt.title('L distribution of curtains')
plt.show()