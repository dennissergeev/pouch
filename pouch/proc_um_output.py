#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for post-processing UM output."""

# Commonly used standard library tools
import re
import warnings

# Scientific stack
import iris
import iris.cube

# My packages and local scripts
from aeolus.coord import (
    add_planet_calendar,
    regrid_3d,
    replace_z_coord,
    roll_cube_pm180,
    ensure_bounds,
    interp_to_cube_time,
)
from aeolus.model import um
from aeolus.subset import CM_INST_CONSTR, CM_MEAN_CONSTR, DimConstr, unique_cubes

from .path import full_path_glob


GLM_MODEL_TIMESTEP = 86400 / 72
INCR_CONSTR = iris.AttributeConstraint(STASH=lambda x: x.item in [181, 182, 233])
GLM_RUNID = r"umglaa"  # file prefix
GLM_FILE_REGEX = GLM_RUNID + r".p[b,c,d,e]{1}[0]{6}(?P<timestamp>[0-9]{2,6})_00"


def process_cubes(
    cubelist,
    timestep=GLM_MODEL_TIMESTEP,
    ref_cube_constr=um.sh,
    extract_incr=True,
    extract_mean=True,
    regrid_multi_lev=True,
    roll_pm180=True,
    add_calendar=False,
    calendar=None,
    planet="earth",
    remove_duplicates=True,
    use_varpack=False,
    varpack=None,
    interp_time=True,
    model=um,
):
    """Post-process data for easier analysis."""
    DC = DimConstr(model=model)

    if remove_duplicates:
        cubelist = unique_cubes(cubelist)

    cm_constr = iris.Constraint()
    if extract_mean:
        cm_constr &= CM_MEAN_CONSTR
    else:
        cm_constr &= CM_INST_CONSTR
    cubelist = cubelist.extract(cm_constr)

    cubes = iris.cube.CubeList()

    # First, extract all multi-level fields
    if use_varpack:
        cubes += cubelist.extract(varpack["multi_level"])

        # Increments
        if extract_incr:
            cubes += cubelist.extract(INCR_CONSTR)
    else:
        cubes = cubelist.extract(DC.strict.tmyx)

    if regrid_multi_lev:
        # Interpolation & regridding to common grid
        ref_cube = cubes.extract_cube(ref_cube_constr)
        ref_cube = replace_z_coord(ref_cube, model=model)

        # Interpolate to common levels
        cubes = iris.cube.CubeList(
            [
                regrid_3d(replace_z_coord(cube, model=model), ref_cube, model=model)
                for cube in cubes
            ]
        )
    # Fix units of increments
    for cube in cubes:
        if cube.attributes["STASH"].item in [181, 233]:
            incr_unit = "K"
        elif cube.attributes["STASH"].item in [182]:
            incr_unit = "kg kg^-1"
        if cube.attributes["STASH"].item == 233:
            cube.units = f"{incr_unit} s^-1"
        elif cube.attributes["STASH"].item in [181, 182]:
            cube.units = f"{1/timestep} {incr_unit} s^-1"
            cube.convert_units(f"{incr_unit} s^-1")

    # Add all single-level cubes
    if use_varpack:
        cubes += cubelist.extract(varpack["single_level"])
    else:
        cubes += cubelist.extract(DC.strict.tyx)

    # Roll cubes to +/- 180 degrees in longitude for easier analysis
    if roll_pm180:
        rolled_cubes = iris.cube.CubeList()
        for cube in cubes:
            r_c = roll_cube_pm180(cube)
            ensure_bounds(r_c)
            rolled_cubes.append(r_c)
    else:
        rolled_cubes = cubes

    if interp_time:
        ref_cube = rolled_cubes.extract_cube(model.t_sfc)  # TODO: make an arg
        final = iris.cube.CubeList()
        for cube in rolled_cubes:
            final.append(interp_to_cube_time(cube, ref_cube, model=model))
    else:
        final = rolled_cubes

    if add_calendar:
        try:
            cal = calendar[planet]
            for cube in final:
                add_planet_calendar(
                    cube,
                    model.t,
                    days_in_year=cal["year"],
                    days_in_month=cal["month"],
                    days_in_day=cal["day"],
                    planet=planet,
                )
        except KeyError:
            warnings.warn(f"Calendar for {planet =} not found.")
    return final


def get_filename_list(
    path_to_dir,
    glob_pattern=f"{GLM_RUNID}*",
    ts_start=0,
    ts_end=-1,
    every=1,
    regex=GLM_FILE_REGEX,
    regex_key="timestamp",
    sort=True,
):
    """Get a list of files with timestamps greater or equal than start in a directory."""
    glob_gen = full_path_glob(path_to_dir / glob_pattern)
    fnames = []
    tstamps = {}
    for fpath in glob_gen:
        match = re.match(regex, fpath.name)
        if match:
            ts_num = int(match[regex_key])
            if (ts_num >= ts_start) and (ts_num % every == 0):
                if (ts_end == -1) or (ts_num <= ts_end):
                    fnames.append(fpath)
                    tstamps[fpath] = ts_num
    if sort:
        fnames = sorted(fnames, key=lambda x: tstamps[x])
    return fnames
