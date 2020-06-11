# This script compares the distribution of AE when
# curtains were observed to the overall AE 
# distribution.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import os
import pathlib

import uncertainties as unc  
import uncertainties.unumpy as unp

import dirs

include_microbursts = False
load_AE_norm = True

bin_width = 100
bins = np.arange(0, 1600, bin_width)

# Load the curtain catalog
cat_name = 'AC6_curtains_baseline_method_sorted_v0.txt'
cat_path = pathlib.Path(dirs.CATALOG_DIR, cat_name)
curtain_cat = pd.read_csv(cat_path)
curtain_cat['dateTime'] = pd.to_datetime(curtain_cat['dateTime'])

# Load the microburst catalog
burst_name = 'AC6_microbursts_sorted_v6.txt'
burst_path = pathlib.Path(dirs.CATALOG_DIR, burst_name)
burst_cat = pd.read_csv(burst_path)


if load_AE_norm:
    # Load the AE normalization
    ae = pd.read_csv(pathlib.Path(dirs.NORM_DIR, 'AE_10Hz_AE.csv'), index_col=0)
    # ae = ae.loc[bins[0]:bins[:-1]]
    H_AE = ae.groupby(ae.index//bin_width).sum()
    H_AE.index = H_AE.index*bin_width
    # Filter to the bin extent
    H_AE = H_AE.loc[bins[0]:bins[-2]]
    H_AE = H_AE['10Hz_samples'].values
    H_AE_density = H_AE/(sum(H_AE)*bin_width)

else:
    # Load the AE index for all of the years that curtains 
    # were observed
    ae_dir = pathlib.Path(dirs.BASE_DIR, 'data', 'ae')
    years = sorted(set(curtain_cat['dateTime'].dt.year))
    ae = pd.DataFrame(data=np.zeros((0, 1)), columns=['AE'])

    for year in years:
        ae_path = pathlib.Path(ae_dir, f'{year}_ae.txt')
        year_ae = pd.read_csv(ae_path, delim_whitespace=True, 
                        usecols=[0, 1, 3], skiprows=14, 
                        parse_dates=[['DATE', 'TIME']])
        year_ae.index=year_ae.DATE_TIME
        del year_ae['DATE_TIME']
        ae = ae.append(year_ae)

    H_AE, _ = np.histogram(ae['AE'], density=False, bins=bins)
    H_AE_density, _ = np.histogram(ae['AE'], density=True, bins=bins)

def hist_density(dist, bin_width):
    """ Normalize a histogram to a PDF """
    dist = dist / (np.sum(dist)*bin_width)
    return dist

H_c, _  = np.histogram(curtain_cat['AE'], density=False, bins=bins)
H_m, _  = np.histogram(burst_cat['AE'], density=False, bins=bins)

H_c = unp.uarray(H_c, np.sqrt(H_c))  
H_m = unp.uarray(H_m, np.sqrt(H_m)) 

# The microburst and curtain probability density and error
H_c_density = hist_density(H_c, bin_width)
H_m_density = hist_density(H_m, bin_width)

# Scale the observed number of curtain and microburst distributions
# by the AE index. This can be thought of the number of curtains
# and microburst observed as a function of AE, assuming all AE values
# occur equally frequently. 
H_c_norm = H_c * (np.max(H_AE)/H_AE)
H_m_norm = H_m * (np.max(H_AE)/H_AE)
H_c_norm_density = hist_density(H_c_norm, bin_width)
H_m_norm_density = hist_density(H_m_norm, bin_width)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(9,4))

### Plot the raw histograms
ae_plot = ax[0].step(bins[:-1], H_AE_density, where='post', label='Index', 
        c='k', lw=2)
curtain_plot = ax[0].step(bins[:-1], unp.nominal_values(H_c_density), 
        where='post', label=f'Curtains', c='b', lw=2, linestyle='-')
ax[0].errorbar(bins[:-1]+bin_width/2, unp.nominal_values(H_c_density), 
        yerr=unp.std_devs(H_c_density), ls='', c='b', lw=1, capsize=3)

if include_microbursts:
    microburst_plot = ax[0].step(bins[:-1], unp.nominal_values(H_m_density), 
            where='post', label='Microbursts', c='g', lw=2, linestyle='--')
    ax[0].errorbar(bins[:-1]+bin_width/2, unp.nominal_values(H_m_density), 
            yerr=unp.std_devs(H_m_density), ls='', c='g', lw=1, capsize=3)


### Plot the normalized histograms
ax[1].step(bins[:-1], unp.nominal_values(H_c_norm_density), where='post', label=f'Curtains', 
        c='b', lw=2, linestyle='-')
ax[1].errorbar(bins[:-1]+bin_width/2, unp.nominal_values(H_c_norm_density), 
        yerr=unp.std_devs(H_c_norm_density), ls='', c='b', lw=1, capsize=3)

if include_microbursts:
    ax[1].step(bins[:-1], unp.nominal_values(H_m_norm_density), where='post', label=f'Microbursts', 
            c='g', lw=2, linestyle='--')
    ax[1].errorbar(bins[:-1]+bin_width/2, unp.nominal_values(H_m_norm_density), 
            yerr=unp.std_devs(H_m_norm_density), ls='', c='g', lw=1, capsize=3)

ax[0].set(xlabel='AE [nT]', ylabel='Probability density', 
        xlim=(0, 1000), ylim=(0, 1.1*np.max(H_AE_density)))
ax[0].text(0.01, 0.98, '(a) Unnormalized', ha='left', va='top', 
        transform=ax[0].transAxes, fontsize=15)
ax[1].set(xlabel='AE [nT]', ylim=(0, 1.1*np.max(unp.nominal_values(H_m_norm_density))))
if include_microbursts:
    plt.suptitle('The distributions of the Auroral Electrojet index for curtains and microbursts', fontsize=15)
    ax[1].legend(handles=[ae_plot[0], curtain_plot[0], microburst_plot[0]], loc=1)
else:
    plt.suptitle('The distribution of the Auroral Electrojet index for curtains', fontsize=15)
    ax[1].legend(handles=[ae_plot[0], curtain_plot[0]], loc=1)
ax[1].text(0.01, 0.98, '(b) Normalized', ha='left', va='top', 
        transform=ax[1].transAxes, fontsize=15)

plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.show()