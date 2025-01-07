from itertools import cycle
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot


def scatter_plot(
    transformed_data: pd.DataFrame,
    color: str = "y",
    size: float = 2.0,
    splot: pyplot.subplot = None,
    label: Optional[List[str]] = None,
):
    """Write a function to generate a 2D scatter plot."""
    if splot is None:
        pyplot.figure(figsize=(12, 6))
        splot = pyplot.subplot()
    columns = transformed_data.columns
    splot.scatter(
        transformed_data.loc[:, columns[0]],
        transformed_data.loc[:, columns[1]],
        size,
        c=color,
        label=label,
    )
    splot.set_aspect("auto")
    splot.set_xlabel("1st Component")
    splot.set_ylabel("2nd Component")
    splot.legend()


def plot_density_estimation_results(
    X: pd.DataFrame,
    Y_: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    title: str,
):
    """Use this function to plot the estimated distribution"""
    color_iter = cycle(["navy", "c", "cornflowerblue", "gold", "darkorange", "g"])
    pyplot.figure()
    splot = pyplot.subplot()
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        scatter_plot(X.loc[Y_ == i], color=color, splot=splot)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    pyplot.title(title)


def plot_finnish_parties(transformed_data: pd.DataFrame, splot: pyplot.subplot = None):
    """Write a function to plot the following finnish parties on a 2D scatter plot"""
    finnish_parties = [
        {"parties": ["SDP", "VAS", "VIHR"], "country": "fin", "color": "r"},
        {"parties": ["KESK", "KD"], "country": "fin", "color": "g"},
        {"parties": ["KOK", "SFP"], "country": "fin", "color": "b"},
        {"parties": ["PS"], "country": "fin", "color": "k"},
    ]
    ##### YOUR CODE GOES HERE #####
    pass

'''
i have a data file like this. on this i am applying a dimensionality reduction technique whose output is 2d with the indices and the target variable as ['lrgen']. how do i make a plot like this:
def plot_finnish_parties(transformed_data: pd.DataFrame, splot: pyplot.subplot = None):
    """Write a function to plot the following finnish parties on a 2D scatter plot"""
    finnish_parties = [
        {"parties": ["SDP", "VAS", "VIHR"], "country": "fin", "color": "r"},
        {"parties": ["KESK", "KD"], "country": "fin", "color": "g"},
        {"parties": ["KOK", "SFP"], "country": "fin", "color": "b"},
        {"parties": ["PS"], "country": "fin", "color": "k"},
    ]
    ##### YOUR CODE GOES HERE #####
    pass
'''

synth_array= np.random.randn(100, 2)

def trans_df(X: np.array) -> pd.DataFrame:
    return pd.DataFrame(data= X)

if __name__ == '__main__':
    scatter_plot(
    transformed_data= trans_df(synth_array),    
    color= 'y',
    size= 2.0,
    splot= None,
    label= None)
    pyplot.show()


