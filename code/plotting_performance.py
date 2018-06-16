import matplotlib
matplotlib.use('Agg');

from matplotlib import rc
rc('font',**{'family':'serif'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np

DPI = 100

def plot_performance(x, y, 
                     title=None, 
                     xlabel=None, 
                     ylabel=None,
                     figfile=None):
    print('plot_performance', flush=True);
    fig, ax = plt.subplots()

    ax.plot(x, y);

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if figfile is None:
        plt.show()
    else:
        fig.savefig(figfile + '.pdf', dpi=DPI, transparent=True)
        plt.close(fig)
    print('plot_performance end', flush=True);