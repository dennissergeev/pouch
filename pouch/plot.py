# -*- coding: utf-8 -*-
"""Temporary storage for plotting functions."""
from pathlib import Path

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

import numpy as np

from scipy import interpolate

from aeolus.plot import (
    GeoAxesGrid,
    label_global_map_gridlines,
    subplot_label_generator,
)
from aeolus.model import um

from . import RUNTIME


KW_CART = dict(transform=ccrs.PlateCarree())
KW_SBPLT_LABEL = dict(fontsize="x-large", fontweight="bold", pad=5, loc="left")
KW_MAIN_TTL = dict(fontsize="large", pad=5, loc="center")
KW_AUX_TTL = dict(fontsize="small", pad=5, loc="right")
KW_SYM0 = dict(norm=mcol.CenteredNorm(0.0), cmap="coolwarm")
# Axes grid specs
KW_AXGR = dict(
    axes_pad=(0.7, 0.4),
    cbar_location="right",
    cbar_mode="single",
    cbar_pad=0.1,
    cbar_size="3%",
    label_mode="",
)
KW_ZERO_LINE = dict(color="tab:grey", alpha=0.5, linestyle=":", dash_capstyle="round")

# Locations of grid lines on maps
XLOCS = np.arange(-180, 181, 90)
YLOCS = np.arange(-90, 91, 30)


def pprint_dict(d):
    """Print each dictionary key-value pair on a new line."""
    return "\n".join(f"{k} = {v}" for k, v in d.items())


def add_aux_yticks(
    ax, src_points, src_values, target_points, twin_ax_ylim=None, twin_ax_inv=False
):
    """
    Add Y-axis ticks at desired locations.
    Examples
    --------
    >>> ax = plt.axes()
    >>> ax.plot(temperature, pressure)
    >>> add_aux_yticks(ax, heights, pressure, [0, 10, 40], twin_ax_ylim=[0, 40], twin_ax_inv=True)
    """
    int_func = interpolate.interp1d(src_points, src_values, fill_value="extrapolate")
    new_points = int_func(target_points)
    _ax = ax.twinx()
    _ax.set_ylim(twin_ax_ylim)
    _ax.set_yticks(new_points)
    _ax.set_yticklabels(target_points)
    if twin_ax_inv:
        _ax.invert_yaxis()
    return _ax


def use_style():
    """Load custom matplotlib style sheet."""
    plt.style.use(Path(__file__).parent / "simple.mplstyle")


def make_map_figure(ncols, nrows, rect=111, **axgr_kw):
    """
    Make a figure with a grid of cartopy axes with the Robinson projection.

    Parameters
    ----------
    ncols: int
        Number of columns
    nrows: int
        Number of rows
    axgr_kw: dict, optional
        Parameters passed to `aeolus.plot.cart.GeoAxesGrid`.

    Returns
    -------
    matplotlib.figure.Figure, aeolus.plot.cart.GeoAxesGrid
        The figure and axes grid.
    """

    iletters = subplot_label_generator()

    fig = plt.figure(figsize=(8 * ncols, 4 * nrows))

    axgr = GeoAxesGrid(
        fig, rect, projection=ccrs.Robinson(), nrows_ncols=(nrows, ncols), **axgr_kw
    )
    for ax in axgr.axes_all:
        label_global_map_gridlines(
            fig, ax, XLOCS[1:-1], YLOCS[1:-1], degree=True, size="x-small", xoff=-15
        )
        ax.gridlines(xlocs=XLOCS, ylocs=YLOCS, crs=ccrs.PlateCarree())
        ax.set_title(f"({next(iletters)})", fontsize="small", pad=5, loc="left")

    return fig, axgr


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    add_cbar=True,
    cbar_kw={},
    cbarlabel="",
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        add_cbar   : Add colorbar
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if add_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=["black", "white"],
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.
    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.
    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[(im.norm(data[i, j]) > threshold).astype(int)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def draw_scalar_cube(
    cube,
    ax,
    method="contourf",
    cax=None,
    tex_units=None,
    cbar_ticks=None,
    use_cyclic=True,
    model=um,
    **plt_kw,
):
    """
    Plot a cube on a map.

    Parameters
    ----------
    cube: iris.cube.Cube
        Cube to plot.
    ax: matplotlib.axes._subplots.AxesSubplot
        Cartopy axes.
    method: str, optional
        Method of plotting, e.g. "contour", "pcolormesh", etc.
    cax: matplotlib.axes._subplots.AxesSubplot or similar
        Axes for the colorbar.
    tex_units: str, optional
        TeX string of cube units to be attached to the colorbar.
    cbar_ticks: sequence, optional
        Colorbar ticks.
    use_cyclic: bool, optional
        Use `cartopy.utils.add_cyclic_point` for the data.
    model: aeolus.model.Model, optional
        Model-specific names and coordinates.
    plt_kw: dict, optional
        Keywords for the plotting method.

    Returns
    -------
    Result of the plotting method.
    """
    lons = cube.coord(model.x).points
    lats = cube.coord(model.y).points
    cb_ttl_kw = dict(fontsize="x-small", pad=5)

    if use_cyclic:
        _data, _lons = add_cyclic_point(cube.data, coord=lons)
    else:
        _data, _lons = cube.data, lons
    h = getattr(ax, method)(_lons, lats, _data, **plt_kw, **KW_CART)
    if cax is not None:
        cb = ax.figure.colorbar(h, cax=cax)
        cb_ttl_kw = dict(fontsize="x-small", pad=5)
        if tex_units is not None:
            cb.ax.set_title(f"[{tex_units}]", **cb_ttl_kw)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
    return h


def draw_vector_cubes(
    u,
    v,
    ax,
    cax=None,
    tex_units=None,
    cbar_ticks=None,
    mag=(),
    xstride=1,
    ystride=1,
    add_wind_contours=False,
    model=um,
    qk_ref_wspd=None,
    kw_quiver={},
    kw_quiverkey={},
    quiverkey_xy=(0.17, 0.87),
):
    """
    Plot vectors of two cubes on a map.

    Parameters
    ----------
    u: iris.cube.Cube
        X-component of the vector.
    v: iris.cube.Cube
        Y-component of the vector.
    ax: matplotlib.axes._subplots.AxesSubplot
        Cartopy axes.
    cax: matplotlib.axes._subplots.AxesSubplot or similar
        Axes for the colorbar.
    tex_units: str, optional
        TeX string of cube units to be attached to the colorbar.
    cbar_ticks: sequence, optional
        Colorbar ticks.
    mag: tuple, optional
        Tuple of numpy arrays to be used for colour-coding the vectors.
    xstride: int, optional
        Stride x-component data.
    ystride: int, optional
        Stride y-component data.
    add_wind_contours: bool, optional
        Add contours of the vector magnitude (wind speed).
    model: aeolus.model.Model, optional
        Model-specific names and coordinates.
    qk_ref_wspd: float, optional
        Reference vector magnitude (wind speed).
        If given, a reference arrow (quiver key) is added to the figure.
    kw_quiver: dict, optional
        Keywords passed to quiver().
    kw_quiverkey: dict, optional
        Keywords passed to quiverkey().
    quiverkey_xy: tuple, optional
        Quiver key position.
    """
    cb_ttl_kw = dict(fontsize="x-small", pad=5)
    xsl = slice(xstride, -xstride, xstride)
    ysl = slice(ystride, -ystride, ystride)

    lons = u.coord(model.x).points
    lats = u.coord(model.y).points

    h = ax.quiver(
        lons[xsl], lats[ysl], u.data[ysl, xsl], v.data[ysl, xsl], *mag, **kw_quiver
    )
    if cax is not None and mag:
        cb = ax.figure.colorbar(h, cax=cax)
        if tex_units is not None:
            cb.ax.set_title(f"[{tex_units}]", **cb_ttl_kw)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)

    if qk_ref_wspd is not None:
        ax.quiverkey(
            h,
            *quiverkey_xy,
            qk_ref_wspd,
            fr"${qk_ref_wspd}$" + r" $m$ $s^{-1}$",
            **kw_quiverkey,
        )

    if add_wind_contours:
        wspd = (u ** 2 + v ** 2) ** 0.5
        ax.contour(
            lons,
            lats,
            wspd.data,
            transform=kw_quiver["transform"],
            levels=np.arange(30, 105, 5),
            cmap="Greens",
        )


def figsave(fig, imgname, **kw_savefig):
    """Save figure and print relative path to it."""
    if RUNTIME.figsave_stamp:
        fig.suptitle(
            imgname.name,
            x=0.5,
            y=0.05,
            ha="center",
            fontsize="xx-small",
            color="tab:grey",
            alpha=0.5,
        )
    save_dir = imgname.absolute().parent
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(imgname, **kw_savefig)
    pth = Path.cwd()
    rel_path = None
    pref = ""
    for par in pth.parents:
        pref += ".." + pth.anchor
        try:
            rel_path = f"{pref}{imgname.relative_to(par)}"
            break
        except ValueError:
            pass
    if rel_path is not None:
        print(f"Saved to {rel_path}.{plt.rcParams['savefig.format']}")


def linspace_pm1(n):
    """Return 2n evenly spaced numbers from -1 to 1, always skipping 0."""
    seq = np.linspace(0, 1, n + 1)
    return np.concatenate([-seq[1:][::-1], seq[1:]])
