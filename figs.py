"""A collection of functions for visualizing data.

Author: Tuomas Puolivali (tuomas.puolivali@helsinki.fi)

Last modified 12th September 2017.
"""
import numpy as np
from numpy.random import rand

import matplotlib.pyplot as plt

def scatterviolinplot(y, violin_widths=0.5, x_jitter=0.25):
    """
    Input arguments:
    y              - An observation matrix where each column corresponds to 
                     a category or condition and each row to an observation.
    violin_widths  - The width of all violins.
    x_jitter       - The amount of horizontal jitter added to the individual 
                     data points.
    """
    n_rows, n_cols = np.shape(y)    
    fig = plt.figure()
    ax = fig.add_subplot(111)

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
        ax.violinplot(y[:, i], positions=[i], widths=violin_widths)

    ax.set_xlim([-1, 5])
    """The x-axis ticks & labels have to be rewritten. They are here 
    set by default to the column indexes."""
    ax.set_xticks(range(0, n_cols))
    ax.set_xticklabels(range(0, n_cols))
    ax.set_xlabel('Column')
    return fig

fig = scatterviolinplot(np.random.normal(loc=0, scale=1, size=(20, 5)))
# manipulate the 'fig' object here to set proper axes etc.
plt.show()
