import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaus(x, p):
    A, x0, sigma = p
    return A*np.exp(-0.5*((x-x0)/sigma)**2)

def baseline(y, width=100):
    baseline = pd.DataFrame(y).rolling(width, center=True).mean()
    baseline = baseline.values.reshape(baseline.shape[0])
    return baseline

# baseline_width = 100
# p = [100, 0, 3] 
# x = np.linspace(-15, 15, 300)
# c = gaus(x, p)
# bl = baseline(c, width=baseline_width)
# std = (c-bl)/np.sqrt(bl)

# fig, ax = plt.subplots(2, sharex=True)
# ax[0].set_title(f'Gaus std={p[-1]} s | baseline width={baseline_width/10} s')
# ax[0].plot(x, c, 'r')
# ax[0].plot(x, bl, 'k')

# ax[1].plot(x, std, 'k')
# ax[1].axhline(2, c='r')

widths = np.arange(1, 10)
std_max = np.nan*np.ones(len(widths))

for i, width in enumerate(widths):
    baseline_width = 100
    p = [50, 0, width] 
    x = np.linspace(-15, 15, 300)
    c = gaus(x, p)
    bl = baseline(c, width=baseline_width)
    std = (c-bl)/np.sqrt(bl)
    std_max[i] = np.nanmax(std)

plt.plot(widths, std_max)
plt.axhline(2)
plt.show()