import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaus(x, p):
    A, x0, sigma = p
    return A*np.exp(-0.5*((x-x0)/sigma)**2)

def baseline(y, width=100):
    # Convert from seconds to data points assuming 10 Hz sampling rate
    width *= 10 
    baseline = pd.DataFrame(y).rolling(width, center=True).mean()
    baseline = baseline.values.reshape(baseline.shape[0])
    return baseline

baseline_width_s = 10

if False:
    curtain_p = [100, 0, 3] 
    x = np.linspace(-15, 15, 300)
    c = gaus(x, curtain_p)
    bl = baseline(c, width=baseline_width_s)
    std = (c-bl)/np.sqrt(bl)

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_title(f'Gaus std={curtain_p[-1]} s | baseline width={baseline_width_s} s')
    ax[0].plot(x, c, 'r')
    ax[0].plot(x, bl, 'k')

    ax[1].plot(x, std, 'k')
    ax[1].axhline(2, c='r')

if True:
    curtain_amplitude = 300
    curtain_widths = np.arange(1, 10)
    max_std = np.nan*np.ones_like(curtain_widths)

    for i, width in enumerate(curtain_widths):
        p = [curtain_amplitude, 0, width] 
        x = np.linspace(-15, 15, 300)
        c = gaus(x, p)
        bl = baseline(c, width=baseline_width_s)
        std = (c-bl)/np.sqrt(bl)
        max_std[i] = np.nanmax(std)

    plt.plot(curtain_widths, max_std)
    plt.axhline(2)
    plt.title(f'Number of standard deviations vs. curtain width\n'
            f'baseline_width={baseline_width_s} s | curtain_amplitude={curtain_amplitude}')
    plt.xlabel('Curtain width [s]')
    plt.ylabel('standard deviation above baseline')
    plt.show()