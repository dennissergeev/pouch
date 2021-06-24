# coding: utf-8
"""Common objects for mean climate diagnostics of global UM runs."""
# Scientific stack
import iris
from iris.exceptions import ConstraintMismatchError as ConMisErr

import scipy.stats.mstats

import numpy as np

# My packages and local scripts
from aeolus.calc import (
    air_density,
    air_potential_temperature,
    air_temperature,
    # cumsum,
    deriv,
    geopotential_height,
    integrate,
    dry_lapse_rate,
    div_h,
    ghe_norm,
    heat_redist_eff,
    bond_albedo,
    spatial,
    last_n_day_mean,
    minmaxdiff,
    meridional_mean,
    region_mean_diff,
    toa_eff_temp,
    toa_cloud_radiative_effect,
    toa_net_energy,
    precip_sum,
    sfc_net_energy,
    sfc_water_balance,
    vertical_mean,
    water_path,
    # zonal_mean,
)
from aeolus.coord import ensure_bounds, interp_cube_from_height_to_pressure_levels
from aeolus.exceptions import MissingCubeError
from aeolus.meta import const_from_attrs, update_metadata
from aeolus.model import um
from aeolus.region import Region
from aeolus.subset import l_range_constr


# Tidally-locked setups
DAYSIDE = Region(-90, 90, -90, 90, "dayside")
NIGHTSIDE = Region(90, -90, -90, 90, "nightside")

KURTOSIS = iris.analysis.Aggregator("kurtosis", scipy.stats.mstats.kurtosis)

hgt_cnstr_above_5km = l_range_constr(5, 1e99)
hgt_cnstr_0_5km = l_range_constr(0, 5)
hgt_cnstr_0_15km = l_range_constr(0, 15)
hgt_cnstr_5_20km = l_range_constr(5, 20)
hgt_cnstr_0_1km = l_range_constr(0, 1)

ONLY_GLOBAL = ["eta", "b_alb", "t_sfc_diff_dn", "nondim_rossby", "nondim_rhines"]
ONLY_LAM = ["hflux_q", "hflux_t"]

DIAGS = {
    "mse_hdiv": lambda cl: mse_hdiv_time_spatial_mean(cl)["mse"],
    "dse_hdiv": lambda cl: mse_hdiv_time_spatial_mean(cl)["dse"],
    "lse_hdiv": lambda cl: mse_hdiv_time_spatial_mean(cl)["lse"],
    "nondim_rossby": lambda cl: nondim_rossby_deformation_radius(cl),
    "nondim_rhines": lambda cl: nondim_rhines_number(cl.extract(hgt_cnstr_0_1km)),
    "e_net_toa": lambda cl: spatial(toa_net_energy(cl), "mean"),
    "eta": lambda cl: heat_redist_eff(cl, NIGHTSIDE, DAYSIDE),
    "b_alb": lambda cl: spatial(bond_albedo(cl), "mean"),
    "toa_olr": lambda cl: spatial(cl.extract_cube(um.toa_olr), "mean"),
    "toa_osr": lambda cl: spatial(cl.extract_cube(um.toa_osr), "mean"),
    "cre_sw": lambda cl: spatial(toa_cloud_radiative_effect(cl, kind="sw"), "mean"),
    "cre_lw": lambda cl: spatial(toa_cloud_radiative_effect(cl, kind="lw"), "mean"),
    "cre_tot": lambda cl: spatial(toa_cloud_radiative_effect(cl, kind="total"), "mean"),
    "gh_norm": lambda cl: spatial(ghe_norm(cl), "mean"),
    "e_net_sfc": lambda cl: spatial(sfc_net_energy(cl), "mean"),
    "t_sfc_diff_dn": lambda cl: region_mean_diff(cl, um.t_sfc, DAYSIDE, NIGHTSIDE),
    "ice_free_frac": lambda cl: spatial(sfc_ice_free_frac(cl), "mean"),
    "t_sfc": lambda cl: spatial(cl.extract_cube(um.t_sfc), "mean"),
    "t_sfc_min": lambda cl: spatial(cl.extract_cube(um.t_sfc), "min"),
    "t_sfc_max": lambda cl: spatial(cl.extract_cube(um.t_sfc), "max"),
    "t_sfc_extremes": lambda cl: minmaxdiff(cl, name=um.t_sfc),
    "t_eff": lambda cl: spatial(toa_eff_temp(cl), "mean"),
    "wspd_rms": lambda cl: wspd_typical(cl, "rms"),
    "wspd_rms_0_15km": lambda cl: wspd_typical(cl.extract(hgt_cnstr_0_15km), "rms"),
    "wspd_mean": lambda cl: wspd_typical(cl, "mean"),
    "wspd_mean_0_15km": lambda cl: wspd_typical(cl.extract(hgt_cnstr_0_15km), "mean"),
    "wspd_max": lambda cl: wspd_typical(cl, "max"),
    "wspd_max_0_15km": lambda cl: wspd_typical(cl.extract(hgt_cnstr_0_15km), "max"),
    "up_atm_wv": lambda cl: upper_atm_vap_mean(cl),
    "wvp": lambda cl: spatial(water_path(cl, "water_vapour"), "mean"),
    "lwp": lambda cl: spatial(water_path(cl, "liquid_water"), "mean"),
    "iwp": lambda cl: spatial(water_path(cl, "ice_water"), "mean"),
    "water_balance": lambda cl: spatial(sfc_water_balance(cl), "mean"),
    "precip_total": lambda cl: spatial(precip_sum(cl), "mean"),
    "precip_stra": lambda cl: spatial(precip_sum(cl, ptype="stra"), "mean"),
    "precip_conv": lambda cl: spatial(precip_sum(cl, ptype="conv"), "mean"),
    "precip_rain": lambda cl: spatial(precip_sum(cl, ptype="rain"), "mean"),
    "precip_snow": lambda cl: spatial(precip_sum(cl, ptype="snow"), "mean"),
    "caf_h": lambda cl: spatial(cl.extract_cube(um.caf_h), "mean"),
    "caf_m": lambda cl: spatial(cl.extract_cube(um.caf_m), "mean"),
    "caf_l": lambda cl: spatial(cl.extract_cube(um.caf_l), "mean"),
    "caf": lambda cl: spatial(cl.extract_cube(um.caf), "mean"),
    "hdiv_5_20km": lambda cl, model=um: spatial(
        vertical_mean(
            div_h(
                *cl.extract([model.u, model.v]).extract(hgt_cnstr_5_20km), model=model
            ),
            weight_by=cl.extract_cube(model.dens).extract(hgt_cnstr_5_20km),
        ),
        "mean",
    ),
    "hdiv_0_5km": lambda cl, model=um: spatial(
        vertical_mean(
            div_h(
                *cl.extract([model.u, model.v]).extract(hgt_cnstr_0_5km), model=model
            ),
            weight_by=cl.extract_cube(model.dens).extract(hgt_cnstr_0_5km),
        ),
        "mean",
    ),
    "dtdz_0_1km": lambda cl, model=um: spatial(
        mean_dry_lapse_rate(cl.extract(hgt_cnstr_0_1km), model=model), "mean"
    ),
}


@const_from_attrs()
def d_dphi(cube, const=None, model=um):
    r"""
    Calculate a derivative w.r.t. latitude.

    .. math::
        \frac{1}{r}\frac{\partal A}{\partial \phi}
    """
    ycoord = cube.coord(model.y).copy()
    cube.coord(model.y).convert_units("radians")
    out = deriv(cube, model.y)
    out /= const.radius
    cube.coord(model.y).convert_units("degrees")
    cube.replace_coord(ycoord)
    out.coord(model.y).convert_units("degrees")
    out.replace_coord(ycoord)
    return out


def last_500d(cube, model=um):
    """Return the time average of the last 500 days of a cube."""
    return last_n_day_mean(cube, days=500)


@const_from_attrs()
def calc_derived_cubes(cubelist, const=None, model=um):
    """Calculate additional variables."""
    try:
        cubelist.extract_cube(model.temp)
    except ConMisErr:
        cubelist.append(air_temperature(cubelist, const=const, model=model))
    try:
        cubelist.extract_cube(model.thta)
    except ConMisErr:
        cubelist.append(air_potential_temperature(cubelist, const=const, model=model))
    try:
        cubelist.extract_cube(model.dens)
    except ConMisErr:
        cubelist.append(air_density(cubelist, const=const, model=model))
    try:
        cubelist.extract_cube(model.ghgt)
    except ConMisErr:
        cubelist.append(geopotential_height(cubelist, const=const, model=model))


@update_metadata(name="wind_speed", units="m s^-1")
def wspd_typical(cubelist, aggr="median", model=um):
    """
    Get an estimate of the typical wind speed.

    Calculate the horizontal wind speed from u and v components
    and collapse the array along spatial and time dimensions.
    """
    u = cubelist.extract_cube(model.u)
    v = cubelist.extract_cube(model.v)
    return spatial((u ** 2 + v ** 2) ** 0.5, aggr).collapsed(
        [model.t, model.z], getattr(iris.analysis, aggr.upper())
    )


@const_from_attrs()
@update_metadata(name="nondimensional_rossby_deformation_radius", units="1")
def nondim_rossby_deformation_radius(
    cubelist, const=None, method="isothermal", model=um
):
    r"""
    Estimate the non-dimensional Rossby radius of deformation.

    .. math::
        \Lambda_{Rossby} = \sqrt{\frac{\sqrt{gH}}{2\beta}} / R_p (isothermal)
        \Lambda_{Rossby} = \sqrt{\frac{NH}{2\beta}} / R_p (stratified)

    Parameters
    ----------
    cubelist: iris.cube.CubeList
        Input cubelist.
    const: aeolus.const.const.ConstContainer, optional
        If not given, constants are attempted to be retrieved from
        attributes of a cube in the cube list.
    method: str, optional
        Method of calculation.
        "isothermal" (default): use isothermal approximation and mean surface temperature as proxy.
        "stratified": estimate scale height and BV frequency from air temperature.

    model: aeolus.model.Model, optional
        Model class with relevant variable names.

    Returns
    -------
    iris.cube.Cube
        Cube with collapsed spatial dimensions.

    References
    ----------
    * Haqq-Misra et al. (2020), https://iopscience.iop.org/article/10.3847/1538-4357/ab9a4b
      - "isothermal": eq. (1)
      - "stratified": eq. (2)
    """

    if method == "stratified":
        double_omega_radius = 2 * const.planet_rotation_rate * const.radius
        rho = cubelist.extract_cube(model.dens)
        bv_freq_proxy = (
            spatial(
                vertical_mean(
                    bv_freq_sq(cubelist, model=model), weight_by=rho, model=model
                ),
                "mean",
            )
            ** 0.5
        )
        temp_proxy = spatial(
            vertical_mean(
                cubelist.extract_cube(model.temp), weight_by=rho, model=model
            ),
            "mean",
        )
        scale_height = const.dry_air_gas_constant * temp_proxy / const.gravity
        nondim_rossby = (
            bv_freq_proxy * scale_height / (2 * double_omega_radius)
        ) ** 0.5

    elif method == "isothermal":
        temp_proxy = spatial(cubelist.extract_cube(model.t_sfc), "mean")
        beta = 2 * const.planet_rotation_rate / const.radius
        rossby_sq = (const.dry_air_gas_constant * temp_proxy) ** 0.5 / (2 * beta)
        nondim_rossby = rossby_sq ** 0.5 / const.radius
    return nondim_rossby


@const_from_attrs()
@update_metadata(name="nondimensional_rhines_scale", units="1")
def nondim_rhines_number(cubelist, const=None, wspd_aggr="mean", model=um):
    r"""
    Estimate the non-dimensional Rhines number.

    Parameters
    ----------
    cubelist: iris.cube.CubeList
        Cubelist containing u- and v-components of wind vector.
    const: aeolus.const.const.ConstContainer, optional
        If not given, constants are attempted to be retrieved from
        attributes of a cube in the cube list.
    wspd_aggr: str, optional
        Aggregation method (one of the available in `iris.analysis`).
    model: aeolus.model.Model, optional
        Model class with relevant variable names.

    Returns
    -------
    nondim_rhines: iris.cube.CubeList
        Cube with collapsed spatial dimensions.

    References
    ----------
    * Haqq-Misra et al. (2017), https://doi.org/10.3847/1538-4357/aa9f1f
    .. math::
        \lambda_{Rhines} = \pi \sqrt{U/\beta} / R_p
    """
    # wspd_aggr = "rms"

    def _wspd_typical(cubelist, aggr=wspd_aggr):
        u = cubelist.extract_cube(model.u)
        v = cubelist.extract_cube(model.v)
        return spatial((u ** 2 + v ** 2) ** 0.5, aggr)

    double_omega_radius = 2 * const.planet_rotation_rate * const.radius
    u_proxy = _wspd_typical(cubelist, wspd_aggr) / 2
    u_proxy = u_proxy.collapsed(model.z, getattr(iris.analysis, wspd_aggr.upper()))
    nondim_rhines = np.pi * (u_proxy / double_omega_radius) ** 0.5
    return nondim_rhines


def mean_dry_lapse_rate(cubelist, model=um):
    rho = air_density(cubelist, model=model)
    dtdz = dry_lapse_rate(cubelist, model=model)
    res = vertical_mean(dtdz, weight_by=rho, model=model)
    return res


@const_from_attrs()
def moist_static_energy(cubelist, const=None, model=um):
    """
    Calculate full, dry and latent components of the moist static energy.

    .. math::
        h = c_p T + g z + L q
    """
    import metpy.calc as metcalc  # noqa
    from aeolus.calc.metpy import preprocess_iris  # noqa

    # Get all the necessary cubes
    try:
        temp = cubelist.extract_cube(model.temp)
        ghgt = cubelist.extract_cube(model.ghgt)
        q = cubelist.extract_cube(model.sh)
        q = preprocess_iris(metcalc.mixing_ratio_from_specific_humidity)(q)
    except ConMisErr:
        varnames = [model.temp, model.ghgt, model.sh]
        raise MissingCubeError(
            f"{varnames} required to calculate mixing ratio are missing from cubelist:\n{cubelist}"
        )
    # Dry component: c_p T + g z
    dry = temp * const.dry_air_spec_heat_press + ghgt
    dry.rename("dry_static_energy")

    # Latent component: L q
    latent = q * const.condensible_heat_vaporization
    latent.rename("latent_static_energy")

    # Total
    mse = dry + latent
    mse.rename("moist_static_energy")
    return dict(mse=mse, dse=dry, lse=latent)


def mse_hdiv_time_merid_mean(cubelist, const=None, model=um):
    """
    Calculate the time and meridional mean of the vertical integral of the horizontal divergence
    of fluxes of moist static energy components.
    """
    mse_cmpnts = moist_static_energy(cubelist, const=const, model=model)

    rho = cubelist.extract_cube(model.dens)
    ensure_bounds(rho, coords="z", model=model)
    u = cubelist.extract_cube(model.u)
    ensure_bounds(u, coords="z", model=model)
    v = cubelist.extract_cube(model.v)
    ensure_bounds(v, coords="z", model=model)

    mse_hdiv_cmpnts = {}
    for key, cmpnt in mse_cmpnts.items():
        flux_x = (u * rho * cmpnt).collapsed(model.t, iris.analysis.MEAN)
        flux_x = integrate(flux_x, model.z)
        flux_x.rename(f"eastward_{cmpnt.name()}_flux")
        flux_y = (v * rho * cmpnt).collapsed(model.t, iris.analysis.MEAN)
        flux_y = integrate(flux_y, model.z)
        flux_y.rename(f"northward_{cmpnt.name()}_flux")

        flux_div = div_h(flux_x, flux_y, model=model)

        flux_div_mean = meridional_mean(flux_div, model=model)
        flux_div_mean.rename(
            f"integrated_meridional_mean_flux_divergence_of_{cmpnt.name()}"
        )
        flux_div_mean.convert_units("W m^-2")
        mse_hdiv_cmpnts[key] = flux_div_mean
    return mse_hdiv_cmpnts


def mse_hdiv_time_spatial_mean(cubelist, model=um):
    """
    Calculate the spatial and temporal mean of the vertical integral of the horizontal divergence
    of fluxes of moist static energy components.
    """
    rho = cubelist.extract_cube(model.dens)
    ensure_bounds(rho, "z", model=model)
    u = cubelist.extract_cube(model.u)
    ensure_bounds(u, "z", model=model)
    v = cubelist.extract_cube(model.v)
    ensure_bounds(v, "z", model=model)

    mse_cmpnts = moist_static_energy(cubelist, model=model)

    mse_hdiv_cmpnts = {}
    for key, cmpnt in mse_cmpnts.items():
        ensure_bounds(cmpnt, "z", model=model)
        flux_x = u * cmpnt
        flux_x.rename(f"eastward_{cmpnt.name()}_flux")
        flux_y = v * cmpnt
        flux_y.rename(f"northward_{cmpnt.name()}_flux")

        flux_div = div_h(flux_x, flux_y, model=model)

        flux_div_vmean = integrate(rho * flux_div, model.z)
        flux_div_mean = spatial(flux_div_vmean, "mean")
        try:
            flux_div_mean = flux_div_mean.collapsed(model.t, iris.analysis.MEAN)
        except iris.exceptions.CoordinateCollapseError:
            pass
        flux_div_mean.rename(f"integrated_horizontal_divergence_of_{cmpnt.name()}")
        flux_div_mean.convert_units("W m^-2")
        mse_hdiv_cmpnts[key] = flux_div_mean
    return mse_hdiv_cmpnts


@update_metadata(name="change_over_time_in_air_temperature_due_to_latent_heat_release")
def latent_heating_rate(cubelist):
    """Temperature increments due to LS precip and (if available) convection."""
    cubes = cubelist.extract(
        [
            "change_over_time_in_air_temperature_due_to_convection",
            "change_over_time_in_air_temperature_due_to_stratiform_precipitation",
            # "change_over_time_in_air_temperature_due_to_boundary_layer_mixing",
        ]
    )
    lh = sum(cubes)
    return lh


@const_from_attrs()
@update_metadata(name="brunt_vaisala_frequency_squared", units="s-2")
def bv_freq_sq(cubelist, const=None, model=um):
    """
    Brunt-Vaisala frequency squared (depends on MetPy).

    Parameters
    ----------
    cubelist: iris.cube.CubeList
        Cubelist containing potential temperature with a height coordinate.
    const: aeolus.const.const.ConstContainer, optional
        If not given, constants are attempted to be retrieved from
        attributes of a cube in the cube list.
    model: aeolus.model.Model, optional
        Model class with relevant variable names.

    Returns
    -------
    iris.cube.Cube
        Cube of N^2
    """
    import metpy.calc as metcalc  # noqa
    import metpy.constants as metconst  # noqa
    import metpy.units as metunits  # noqa

    from aeolus.calc.metpy import preprocess_iris  # noqa

    theta = cubelist.extract_cube(model.thta)
    heights = theta.coord(model.z).points * metunits.units.metre
    metconst.g = const.gravity.data * metunits.units(const.gravity.units.format(1))
    res = preprocess_iris(metcalc.brunt_vaisala_frequency_squared)(
        heights, theta, vertical_dim=theta.coord_dims(theta.coord(model.z))[0]
    )
    return res


@update_metadata(name="ice_free_area_fraction", units="1")
def sfc_ice_free_frac(cubelist, t_freeze=271.35, model=um):
    """
    Calculate the fraction of planet surface not covered by ice, using surface temperature
    above a threshold.

    Parameters
    ----------
    cubelist: iris.cube.CubeList
        Cubelist containing potential temperature with a height coordinate.
    t_freeze: float, optional
        Temperature threshold to determine if the surface is covered by ice [K].
    model: aeolus.model.Model, optional
        Model class with relevant variable names.

    Returns
    -------
    iris.cube.Cube
        Cube of area fraction [0-1].
    """
    t_sfc = cubelist.extract_cube(model.t_sfc)
    t_sfc.convert_units("K")

    frac = np.exp(-(t_freeze - t_sfc.core_data()) / 2)
    frac = np.where(frac < 1, frac, 1)
    frac = t_sfc.copy(data=frac)
    return frac


def upper_atm_vap_mean(cubelist, levels=0.1, const=None, model=um):
    """
    Estimate the mean vapour content in the upper atmosphere.

    Parameters
    ----------
    cubelist: iris.cube.CubeList
        Input cubelist.
    levels: float, optional
        Fraction of reference surface pressure at which the estimate is made.
        The default value is 0.1, which for an Earth-like atmosphere means 100 hPa.
    const: aeolus.const.const.ConstContainer, optional
        If not given, constants are attempted to be retrieved from
        attributes of a cube in the cube list.
    model: aeolus.model.Model, optional
        Model class with relevant variable names.

    Returns
    -------
    iris.cube.Cube
        Cube with collapsed spatial dimensions.
    """
    sh = cubelist.extract_cube(model.sh)
    pres = cubelist.extract_cube(model.pres)
    sh_plev = interp_cube_from_height_to_pressure_levels(
        sh, pres, levels=levels, p_ref_frac=True, const=const, model=model
    )
    sh_plev = spatial(sh_plev, "mean", model=model)
    return sh_plev


def vertical_eddy_flux(cubelist, scalar, model=um):
    """Vertical eddy flux."""
    w_cube = cubelist.extract_cube(um.w)
    s_cube = cubelist.extract_cube(scalar)

    w_mean = w_cube.copy()
    s_mean = s_cube.copy()
    try:
        w_mean = spatial(w_mean, "mean", model=model)
        s_mean = spatial(s_mean, "mean", model=model)
    except iris.exceptions.CoordinateCollapseError:
        pass
    try:
        w_mean = w_mean.collapsed(model.t, iris.analysis.MEAN)
        s_mean = s_mean.collapsed(model.t, iris.analysis.MEAN)
    except iris.exceptions.CoordinateCollapseError:
        pass
    w_dev = w_cube - w_mean
    s_dev = s_cube - s_mean

    flux = w_dev * s_dev
    try:
        flux = flux.collapsed(model.t, iris.analysis.MEAN)
    except iris.exceptions.CoordinateCollapseError:
        pass
    return flux


def low_pass_weights(window, cutoff):
    """
    Calculate weights for a low pass Lanczos filter.

    Taken from SciTools example gallery.

    Parameters
    ----------
    window: int
        The length of the filter window.
    cutoff: float
        The cutoff frequency in inverse time steps.

    Returns
    -------
    w: numpy.ndarray
        Array of weights.
    """
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1.0, n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2.0 * np.pi * cutoff * k) / (np.pi * k)
    w[n - 1 : 0 : -1] = firstfactor * sigma  # noqa
    w[n + 1 : -1] = firstfactor * sigma  # noqa
    return w[1:-1]


def rolling_mean(cube, coord, window=20, cutoff=1 / 5):
    wgts = low_pass_weights(window, cutoff)
    return cube.rolling_window(coord, iris.analysis.SUM, len(wgts), weights=wgts)


def rescale_day_length(cube, day_length, model=um):
    """
    Rescale time coordinates using a new day length.

    The input cube has to contain `model.fcst_ref`, `model.fcst_prd` and `model.t` coordinates
    and be ordered by forecast reference time (`model.fcst_ref`).

    Parameters
    ----------
    cube: iris.cube.Cube
        Input cube containing time dimensions.
    day_length: int or float
        Number of hours per day.
    model: aeolus.model.Model, optional
        Model class with relevant variable names.

    Returns
    -------
    iris.cube.Cube
        Cube with rescaled time coordinates.
    """
    BASE_DAY_LEN = 24
    cubes = iris.cube.CubeList()
    # Iterate over forecast reference time and create a list of cubes
    for i, cube in enumerate(cube.slices_over(model.fcst_ref)):
        # Copy the forecast reference period coordinate to be used later
        orig_fcst_prd_coord = cube.coord(model.fcst_prd).copy()
        cube.remove_coord(model.fcst_prd)
        if i == 0:
            orig_fcst_ref_coord = cube.coord(model.fcst_ref).copy()
        cube.remove_coord(model.fcst_ref)
        # Promote `time` to be a dimensional coordinate instead of forecast period/reference
        iris.util.promote_aux_coord_to_dim_coord(cube, model.t)
        cubes.append(cube)
    # Concatenate cubes w/o the forecast-related coordinates, effectively
    # flattenning the input cube along the time axis.
    cube = cubes.concatenate_cube()
    orig_time_coord = cube.coord(model.t)
    time_len = orig_time_coord.shape[0]
    # Rescale the main time coordinate by starting from the same start date, but
    # having a different step for each hour
    scaled_time_coord = orig_time_coord.copy(
        points=np.arange(time_len) * (BASE_DAY_LEN / day_length) + orig_time_coord.points[0]
    )
    cube.replace_coord(scaled_time_coord)
    # Create a sawtooth-like array of hours, which are reset to 0 every forecast day
    hour_of_day = np.tile(
        np.arange(1, day_length + 1), np.ceil(time_len / day_length).astype(int)
    )[:time_len]
    # Add the sawtooth array as an auxiliary coordinate using the original `fcst_prd` coordinate
    cube.add_aux_coord(
        iris.coords.AuxCoord.from_coord(orig_fcst_prd_coord).copy(
            points=hour_of_day * BASE_DAY_LEN / day_length
        ),
        data_dims=cube.coord_dims(scaled_time_coord),
    )
    cube.add_aux_coord(orig_fcst_ref_coord)
    return cube
