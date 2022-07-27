#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Triage a standard UM run."""
# Standard library tools
import argparse
from pathlib import Path
from time import time
import warnings

# Sci
import iris.analysis
import iris.cube
import iris.exceptions
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np

# My packages and local scripts
from aeolus.calc import (
    horiz_wind_cmpnts,
    spatial,
    toa_net_energy,
    zonal_mean,
    time_mean,
)
from aeolus.coord import get_cube_rel_days, isel, roll_cube_pm180
from aeolus.io import load_data
from aeolus.model import um
from aeolus.plot import tex2cf_units

from pouch.proc_um_output import get_filename_list
from pouch.log import create_logger

# Global definitions and styles
warnings.filterwarnings("ignore")
SCRIPT = Path(__file__).name
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["savefig.bbox"] = "tight"
GLM_FILE_REGEX_OPTS = [
    r"umglaa.p[a]{1}(?P<timestamp>[0-9]{6})",
    r"umglaa.p[b,c,d]{1}[0]{6}(?P<timestamp>[0-9]{2,6})_00",
]


def _get_uv_and_scalar(cubelist, scalar_name, z_idx=20, model=um):
    """
    Extract u and v winds and a scalar field and average them in time.

    Parameters
    ----------
    cubelist: iris.cube.CubeList
        List of cubes with horizontal wind components
    scalar_name: str
        Name of the scalar variable to extract.
    z_idx: int, optional
        Index in the vertical dimension.
    model: aeolus.model.Model, optional
        Model class with relevant variable names.

    Returns
    -------
    u, v, scalar: iris.cube.Cube
        Cubes of wind components and a scalar variable.
    """
    cl = isel(cubelist, model.z, z_idx)
    u, v = horiz_wind_cmpnts(cl, model=model)
    s = cl.extract_cube(scalar_name)
    cubes = []
    for cube in [u, v, s]:
        cube = time_mean(cube)
        cube.coord(model.x).bounds = None
        cube = roll_cube_pm180(cube, model=model)
        cubes.append(cube)
    return cubes


def parse_args(args=None):
    """Argument parser."""
    ap = argparse.ArgumentParser(
        SCRIPT,
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f"""Usage:
./{SCRIPT} ~/path/to/inp/dir/ -p trap1e -s 1000 -l test_label -o ~/plots/triage
""",
    )

    ap.add_argument("inpdir", type=str, help="Input directory")
    ap.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=str,
        help="Directory to save plots",
    )
    ap.add_argument(
        "--filemask",
        type=int,
        required=False,
        default=0,
        help=f"File mask option {[*enumerate(GLM_FILE_REGEX_OPTS)]}",
    )
    ap.add_argument(
        "-c",
        "--constdir",
        required=True,
        type=str,
        help="Directory with constants for aeolus",
    )
    ap.add_argument(
        "-p",
        "--planet",
        type=str,
        required=False,
        default="earth",
        help="Planet configuration",
    )
    ap.add_argument(
        "-s",
        "--startday",
        type=int,
        default=0,
        help="Load files with timestamp >= this",
    )
    ap.add_argument(
        "-e",
        "--endday",
        type=int,
        default=-1,
        help="Load files with timestamp <= this",
    )
    ap.add_argument(
        "-l",
        "--label",
        type=str,
        help="Custom run label",
    )
    ap.add_argument(
        "--noshow",
        action="store_true",
        default=False,
        help="Do not show the image and only save it to file",
    )
    return ap.parse_args(args)


def plotter(cubelist, label, outdir, show=True):
    """Create a plot from the given cubelist."""
    vrbl2plot = {
        "toa_net": {
            "func": lambda cl: spatial(toa_net_energy(cl), "mean"),
            "tex_units": "$W$ $m^{-2}$",
            "title": "TOA radiation balance",
        },
        "p_sfc": {
            "func": lambda cl: spatial(cl.extract_cube(um.p_sfc), "mean"),
            "tex_units": "$hPa$",
            "title": "Surface pressure",
        },
        "t_sfc": {
            "func": lambda cl: [
                spatial(cl.extract_cube(um.t_sfc), i) for i in ["min", "mean", "max"]
            ],
            "tex_units": ["$K$"] * 3,
            "title": "Surface temperature",
        },
        "u": {
            "func": lambda cl: [
                spatial(cl.extract_cube(um.u), i).collapsed(
                    um.z, getattr(iris.analysis, i.upper())
                )
                for i in ["min", "mean", "max"]
            ],
            "tex_units": ["$m$ $s^{-1}$"] * 3,
            "title": "Zonal wind",
        },
        "temp": {
            "func": lambda cl: [
                spatial(isel(cl.extract_cube(um.temp), um.z, idx), "mean")
                for idx in [1, 5, 10, 15, 20, 25, 30]
            ],
            "tex_units": ["$K$"] * 7,
            "title": "Potential temperature",
        },
        "sh": {
            "func": lambda cl: [
                spatial(isel(cl.extract_cube(um.sh), um.z, idx), "mean")
                for idx in [1, 5, 10, 15, 20, 25, 30]
            ],
            "tex_units": ["$kg$ $kg^{-1}$"] * 7,
            "title": "Specific humidity",
        },
        "map_low_atm": {
            "func": lambda cl: _get_uv_and_scalar(cl, um.t_sfc, z_idx=0),
            "tex_units": ["$m$ $s^{-1}$", "$m$ $s^{-1}$", "$K$"],
            "title": "Surface temperature and wind vectors",
        },
        "map_up_atm": {
            "func": lambda cl: _get_uv_and_scalar(cl, um.toa_olr, z_idx=20),
            "tex_units": ["$m$ $s^{-1}$", "$m$ $s^{-1}$", "$W$ $m^{-2}$"],
            "title": "TOA OLR and wind vectors",
        },
        "vcross": {
            "func": lambda cl: [
                time_mean(zonal_mean(cl.extract_cube(i))) for i in [um.u, um.temp]
            ],
            "tex_units": ["$m$ $s^{-1}$", "$K$"],
            "title": "Zonal mean zonal wind and potential temperature",
        },
    }
    # Equalise time coordinate
    # cubelist = iris.cube.CubeList(
    #     [
    #         interp_to_cube_time(cube, cubelist.extract_cube(um.t_sfc))
    #         for cube in cubelist
    #     ]
    # )

    # Compute the specified diagnostics
    for vrbl_key, vrbl_dict in vrbl2plot.items():
        try:
            vrbl_dict["cubes"] = vrbl_dict["func"](cubelist)
        except iris.exceptions.ConstraintMismatchError:
            L.info(f"{vrbl_key=} - missing.")
            continue
        if isinstance(vrbl_dict["cubes"], iris.cube.Cube):
            vrbl_dict["cubes"].convert_units(tex2cf_units(vrbl_dict["tex_units"]))
        else:
            for cube, t_u in zip(vrbl_dict["cubes"], vrbl_dict["tex_units"]):
                cube.convert_units(tex2cf_units(t_u))
        L.info(f"{vrbl_key=} - done.")

    # Make the plot
    nrows = 3
    ncols = 3
    yskip = 5
    xskip = 4
    kw_grid = dict(color="#AAAAAA", linestyle="-", linewidth=0.5)

    fig = plt.figure(figsize=(nrows * 6, ncols * 4))

    for plt_lab in vrbl2plot.keys():
        if not (cubes := vrbl2plot[plt_lab].get("cubes")):
            continue
        match plt_lab:
            case "toa_net":
                ax = fig.add_subplot(nrows, ncols, 1, label=plt_lab)
                cube = vrbl2plot[plt_lab]["cubes"]
                ax.plot(get_cube_rel_days(cube), cube.data, marker=".")
                ax.set_xlabel("Time [day]")
                ax.set_ylabel(vrbl2plot[plt_lab]["tex_units"])
                ax.grid(**kw_grid)

            case "p_sfc":
                ax = fig.add_subplot(nrows, ncols, 4, label=plt_lab)
                cube = vrbl2plot[plt_lab]["cubes"]
                ax.plot(get_cube_rel_days(cube), cube.data, marker=".")
                ax.set_xlabel("Time [day]")
                ax.set_ylabel(vrbl2plot[plt_lab]["tex_units"])
                ax.grid(**kw_grid)

            case "t_sfc":
                ax = fig.add_subplot(nrows, ncols, 2, label=plt_lab)
                ax.fill_between(
                    get_cube_rel_days(cubes[0]), cubes[0].data, cubes[2].data, alpha=0.5
                )
                ax.plot(get_cube_rel_days(cubes[1]), cubes[1].data, marker=".")
                ax.set_xlabel("Time [day]")
                ax.set_ylabel(vrbl2plot[plt_lab]["tex_units"][0])
                ax.grid(**kw_grid)

            case "u":
                ax = fig.add_subplot(nrows, ncols, 5, label=plt_lab)
                ax.fill_between(
                    get_cube_rel_days(cubes[0]), cubes[0].data, cubes[2].data, alpha=0.5
                )
                ax.plot(get_cube_rel_days(cubes[1]), cubes[1].data, marker=".")
                ax.set_xlabel("Time [day]")
                ax.set_ylabel(vrbl2plot[plt_lab]["tex_units"][0])
                ax.grid(**kw_grid)

            case "temp":
                ax = fig.add_subplot(nrows, ncols, 3, label=plt_lab)
                for cube in cubes:
                    leg_label = f"{cube.coord(um.z).points[0]:.0f} m"
                    ax.plot(
                        get_cube_rel_days(cube),
                        cube.data,
                        label=leg_label,
                    )
                ax.legend(title="Levels")
                ax.set_xlabel("Time [day]")
                ax.set_ylabel(vrbl2plot[plt_lab]["tex_units"][0])
                ax.grid(**kw_grid)

            case "sh":
                ax = fig.add_subplot(nrows, ncols, 6, label=plt_lab)
                for cube in cubes:
                    leg_label = f"{cube.coord(um.z).points[0]:.0f} m"
                    ax.plot(
                        get_cube_rel_days(cube),
                        cube.data,
                        label=leg_label,
                    )
                ax.legend(title="Levels")
                ax.set_xlabel("Time [day]")
                ax.set_ylabel(vrbl2plot[plt_lab]["tex_units"][0])
                ax.set_yscale("log")
                ax.grid(**kw_grid)

            case "map_low_atm":
                ax = fig.add_subplot(nrows, ncols, 7, label=plt_lab)
                cntrf = ax.contourf(
                    cubes[-1].coord(um.x).points,
                    cubes[-1].coord(um.y).points,
                    cubes[-1].data,
                    cmap="inferno",
                )
                cb = fig.colorbar(cntrf, ax=ax, orientation="horizontal", pad=0.2)
                ax.quiver(
                    cubes[0].coord(um.x).points[::xskip],
                    cubes[0].coord(um.y).points[::yskip],
                    cubes[0].data[::yskip, ::xskip],
                    cubes[1].data[::yskip, ::xskip],
                    color="k",
                )
                ax.set_xlim(cubes[0].coord(um.x).points[0], cubes[0].coord(um.x).points[-1])
                ax.set_ylim(cubes[0].coord(um.y).points[0], cubes[0].coord(um.y).points[-1])
                cb.ax.set_xlabel(vrbl2plot[plt_lab]["tex_units"][-1])
                ax.set_xlabel(r"Longitude [$^\circ$]")
                ax.set_ylabel(r"Latitude [$^\circ$]")
                ax.add_artist(
                    AnchoredText(f"Wind at {cubes[0].coord(um.z).points[0]:.0f} m", loc=1)
                )
                ax.grid(**kw_grid)

            case "map_up_atm":
                ax = fig.add_subplot(nrows, ncols, 8, label=plt_lab)
                cntrf = ax.contourf(
                    cubes[-1].coord(um.x).points,
                    cubes[-1].coord(um.y).points,
                    cubes[-1].data,
                    cmap="cubehelix",
                )
                cb = fig.colorbar(cntrf, ax=ax, orientation="horizontal", pad=0.2)
                ax.quiver(
                    cubes[0].coord(um.x).points[::xskip],
                    cubes[0].coord(um.y).points[::yskip],
                    cubes[0].data[::yskip, ::xskip],
                    cubes[1].data[::yskip, ::xskip],
                    color="k",
                )
                ax.set_xlim(cubes[0].coord(um.x).points[0], cubes[0].coord(um.x).points[-1])
                ax.set_ylim(cubes[0].coord(um.y).points[0], cubes[0].coord(um.y).points[-1])
                cb.ax.set_xlabel(vrbl2plot[plt_lab]["tex_units"][-1])
                ax.set_xlabel(r"Longitude [$^\circ$]")
                ax.set_ylabel(r"Latitude [$^\circ$]")
                ax.add_artist(
                    AnchoredText(f"Wind at {cubes[0].coord(um.z).points[0]:.0f} m", loc=1)
                )
                ax.grid(**kw_grid)

            case "vcross":
                ax = fig.add_subplot(nrows, ncols, 9, label=plt_lab)
                cntrf = ax.contourf(
                    cubes[0].coord(um.y).points,
                    cubes[0].coord(um.z).points,
                    cubes[0].data,
                    cmap="viridis",
                )
                cntr = ax.contour(
                    cubes[1].coord(um.y).points,
                    cubes[1].coord(um.z).points,
                    cubes[1].data,
                    cmap="inferno",
                    levels=np.concatenate([np.arange(0, 400, 20), np.arange(400, 2000, 200)]),
                )
                ax.clabel(cntr, fmt="%.0f")
                cb = fig.colorbar(cntrf, ax=ax, orientation="horizontal", pad=0.2)
                cb.ax.set_xlabel(vrbl2plot[plt_lab]["tex_units"][0])
                ax.set_xlabel(r"Latitude [$^\circ$]")
                ax.set_ylabel("Height [$m$]")
                ax.set_ylim(cubes[0].coord(um.z).points[0], cubes[0].coord(um.z).points[-1])

    axdict = {ax.get_label(): ax for ax in fig.axes if ax.get_label() != "<colorbar>"}

    for plt_lab, ax in axdict.items():
        ax.set_title(vrbl2plot[plt_lab]["title"], loc="left", fontsize=8)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle(f"Experiment: {label}", y=0.925)

    fname_out = outdir / f"triage_{label}.png"
    fig.savefig(fname_out, dpi=200)
    if show:
        plt.show()
    return fname_out


def main(args=None):
    """Main entry point of the script."""
    t0 = time()
    global L
    L = create_logger(Path(__file__))
    # Parse command-line arguments
    args = parse_args(args)
    planet = args.planet
    L.info(f"{planet=}")

    # Input directory
    inpdir = Path(args.inpdir)
    L.info(f"{inpdir=}")

    # GLM regex option
    glm_regex = GLM_FILE_REGEX_OPTS[args.filemask]

    # Make a list of files matching the file mask and the start day threshold
    fnames = get_filename_list(
        inpdir, ts_start=args.startday, ts_end=args.endday, every=1, regex=glm_regex
    )
    if len(fnames) == 0:
        L.critical("No files found!")
        return
    L.info(f"fnames({len(fnames)}) = {fnames[0]} ... {fnames[-1]}")

    # Output image file name
    if args.label:
        run_label = args.label
    else:
        # Determine run label automatically
        pth = Path(fnames[0])
        try:
            run_label = pth.parts[pth.parts.index("cylc-run") + 1]
        except ValueError:
            run_label = "unknown"
            L.info(f"'cylc-run' is not in the {pth}, run_label is set to {run_label}")

    # Load data from selected files
    cl = load_data(fnames)  # , name=run_label, planet=planet, const_dir=args.constdir)
    if len(cl) == 0:
        L.critical("Files are empty!")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Process data and make a plot
    fname_out = plotter(cl, run_label, outdir=outdir, show=(not args.noshow))
    L.success(f"Image saved to {fname_out}")
    L.info(f"Execution time: {time() - t0:.1f}s")


if __name__ == "__main__":
    main()
