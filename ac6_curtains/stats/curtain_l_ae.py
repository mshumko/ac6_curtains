import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

import dirs

df = pd.read_csv(os.path.join(dirs.CATALOG_DIR, 
'AC6_curtains_baseline_method_sorted_v0.csv'))

df_morning = df[(df.MLT_OPQ > 7) & (df.MLT_OPQ < 13)]
df_night = df[(df.MLT_OPQ > 18) & (df.MLT_OPQ < 24)]
bins = np.arange(2, 14)

# plt.figure()
# plt.hist(df.Lm_OPQ, bins=bins, color='g', alpha=0.4, label='all')
# plt.hist(df_morning.Lm_OPQ, bins=bins, color='r', alpha=0.4, label='7-13 MLT')
# plt.hist(df_night.Lm_OPQ, bins=bins, color='b', alpha=0.4, label='18-24 MLT')
# plt.legend()

# plt.xlabel('Lm_OPQ')
# plt.ylabel('Number of curtains')
# plt.title('L distribution of curtains')

plt.figure()
boxplot_bins = np.arange(0, 1001, 100)
boxplot_points = [df.loc[(df.AE > low_AE) & (df.AE < high_AE), 'Lm_OPQ'].dropna().to_numpy()
                for (low_AE, high_AE) in zip(boxplot_bins[:-1], boxplot_bins[1:])]
plt.boxplot(boxplot_points, labels=boxplot_bins[:-1])
plt.xlabel('AE [nT]')
plt.ylabel('Lm_OPQ')
plt.title('AC6 curtain L vs AE distributions')

### Now look at the dependence as a function of AE ###
calm_ae_thresh = 200
disturbed_ae_thresh = 500
density=False

plt.figure()
df_calm = df[df['AE'] < calm_ae_thresh]
df_disturbed = df[df['AE'] > disturbed_ae_thresh]

plt.hist(df_calm['Lm_OPQ'], bins=bins, color='r', lw=3, 
        label=f'AE < {calm_ae_thresh}', histtype='step', density=density)
plt.hist(df_disturbed['Lm_OPQ'], bins=bins, color='k', lw=3, 
        label=f'AE > {disturbed_ae_thresh}', histtype='step', density=density)
plt.legend()
plt.title('Curtain L distribution for calm and disturbed times')
plt.xlabel('Lm_OPQ')
if density:
    plt.ylabel('Probability density')
else:
    plt.ylabel('Number of curtains')
plt.show()