"""A collection of functions for visualizing data.

Author: Tuomas Puolivali (tuomas.puolivali@helsinki.fi)

Last modified 14th September 2017.
"""
import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt

from scipy.stats import f_oneway

import seaborn as sb

def scatter_violinplot(y, oneway=True, violin_widths=0.5, x_jitter=0.15):
    """
    Input arguments:
    y              - An observation matrix where each column corresponds to 
                     a category or condition and each row to an observation.
    oneway         - Whether to test for difference in means
    violin_widths  - The width of all violins.
    x_jitter       - The amount of horizontal jitter added to the individual 
                     data points.
    """
    n_rows, n_cols = np.shape(y)    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    """Draw the violins and boxplots."""
    for i in range(0, n_cols):
        """Draw the individual data points."""
        x = np.asarray([i] * n_rows, dtype='float')
        if x_jitter > 0:
            """The function 'rand' gives random number in [0, 1). Shift the 
            distribution to [-1/2, 1/2) and multiply with 'x_jitter' to 
            produce an effect that is aligned with the x-tick centered 
            boxplots."""
            x += x_jitter*(rand(n_rows)-0.5)
        ax.plot(x, y[:, i], '.')
        """Draw the violin plots."""
        ax.violinplot(y[:, i], positions=[i], widths=violin_widths, 
                      showmedians=False, showextrema=False)
        ax.boxplot(y[:, i], positions=[i], widths=0.4*violin_widths,
                   medianprops={'linewidth': 1.5, 'color': 'orange'})

    ax.set_xlim([-1, n_cols])
    """The x-axis ticks & labels have to be rewritten. They are here 
    set by default to the column indexes."""
    ax.set_xticks(range(0, n_cols))
    ax.set_xticklabels(range(0, n_cols))
    ax.set_xlabel('Column')

    """Test for difference in means using 1-way analysis of variance."""
    if oneway:
        stat, pval = f_oneway(*y.T)
        ax.annotate('1-way: F = %.3f, p = %1.4f' % (stat, pval), xy=(0,0), 
                    xytext=(0,0), textcoords='figure fraction', 
                    horizontalalignment='left', verticalalignment='bottom')
    return fig

from sklearn import datasets

X = datasets.load_iris().data
#X = np.random.normal(loc=0, scale=1, size=(50, 5))
fig = scatter_violinplot(X)
# manipulate the 'fig' object here to set proper axes etc.
plt.show()
