"""Plotting helper for line segments coloured by a scalar quantity."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.collections import LineCollection


def plotcolored(ax, x, y, c, *, lw=None, cmap=None, norm=None):
    """Add a colour-mapped line to a Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that receives the line collection.
    x, y : array_like
        Coordinates of the line vertices.
    c : array_like
        Scalar values used to colour the line segments.
    lw : float, optional
        Line width. The Matplotlib default is used when omitted.
    cmap, norm : optional
        Matplotlib colormap and normalization. By default, the active colormap
        and a normalization over ``[0, 1]`` are used.

    Returns
    -------
    matplotlib.collections.LineCollection
        The collection added to ``ax``.
    """

    if lw is None:
        lw = mpl.rcParams["lines.linewidth"]
    if cmap is None:
        cmap = mpl.rcParams["image.cmap"]
    if norm is None:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap=cmap, norm=norm)
    collection.set_array(c)
    collection.set_linewidth(lw)
    ax.add_collection(collection)
    return collection


__all__ = ["plotcolored"]
