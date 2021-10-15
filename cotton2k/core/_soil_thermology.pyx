# cython: language_level=3
from libc.math cimport exp
from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange


cdef double form(double c0, double d0, double g0) nogil:
    """Computes the aggregation factor for 2 mixed soil materials.

    Arguments
    ---------
    c0
        heat conductivity of first material
    d0
        heat conductivity of second material
    g0
        shape factor for these materials
    """
    return (2 / (1 + (c0 / d0 - 1) * g0) + 1 / (1 + (c0 / d0 - 1) * (1 - 2 * g0))) / 3


cdef double soil_thermal_conductivity(
    double cka,
    double ckw,
    double dsand,
    double bsand,
    double dclay,
    double bclay,
    double soil_sand_volume_fraction,
    double soil_clay_volume_fraction,
    double pore_space,
    double field_capacity,
    double marginal_water_content,
    double heat_conductivity_dry_soil,
    double q0,
    double t0,
) nogil:
    """Computes and returns the thermal conductivity of the soil
    (cal cm-1 s-1 oC-1). It is based on the work of De Vries(1963).

    Arguments
    ---------
    q0
        volumetric soil moisture content.
    t0
        soil temperature (K).
    """
    # Convert soil temperature to degrees C.
    cdef double tcel = t0 - 273.161  # soil temperature, in C.
    # Compute cpn, the apparent heat conductivity of air in soil pore spaces, when
    # saturated with water vapor, using a function of soil temperature, which
    # changes linearly between 36 and 40 C.
    # effect of temperature on heat conductivity of air saturated with water vapor.
    cdef double bb
    if tcel <= 36:
        bb = 0.06188
    elif 36 < tcel <= 40:
        bb = 0.0977 - 0.000995 * tcel
    else:
        bb = 0.05790
    # apparent heat conductivity of air in soil pore spaces, when it is saturated
    # with water vapor.
    cdef double cpn = cka + 0.05 * exp(bb * tcel)
    # Compute xair, air content of soil per volume, from soil porosity and moisture
    # content.
    # Compute thermal conductivity
    # (a) for wet soil (soil moisture higher than field capacity),
    # (b) for less wet soil.
    # In each case compute first ga, and then dair.
    # air content of soil, per volume.
    cdef double xair = max(pore_space - q0, 0)
    cdef double dair  # aggregation factor for air in soil pore spaces.
    cdef double ga  # shape factor for air in pore spaces.
    cdef double hcond  # computed heat conductivity of soil, mcal cm-1 s-1 oc-1.
    cdef double qq  # soil water content for computing ckn and ga.
    cdef double ckn  # heat conductivity of air in pores in soil.
    if q0 >= field_capacity:
        # (a) Heat conductivity of soil wetter than field capacity.
        ga = 0.333 - 0.061 * xair / pore_space
        dair = form(cpn, ckw, ga)
        hcond = (
            q0 * ckw
            + dsand * bsand * soil_sand_volume_fraction
            + dclay * bclay * soil_clay_volume_fraction
            + dair * cpn * xair
        ) / (
            q0
            + dsand * soil_sand_volume_fraction
            + dclay * soil_clay_volume_fraction
            + dair * xair
        )
    else:
        # (b) For soil less wet than field capacity, compute also ckn (heat
        # conductivity of air in the soil pores).
        qq = max(q0, marginal_water_content)
        ckn = cka + (cpn - cka) * qq / field_capacity
        ga = 0.041 + 0.244 * (qq - marginal_water_content) / (
            field_capacity - marginal_water_content
        )
        dair = form(ckn, ckw, ga)
        hcond = (
            qq * ckw
            + dsand * bsand * soil_sand_volume_fraction
            + dclay * bclay * soil_clay_volume_fraction
            + dair * ckn * xair
        ) / (
            qq
            + dsand * soil_sand_volume_fraction
            + dclay * soil_clay_volume_fraction
            + dair * xair
        )
        # When soil moisture content is less than the limiting value
        # marginal_water_content, modify the value of hcond.
        if qq <= marginal_water_content:
            hcond = (
                hcond - heat_conductivity_dry_soil
            ) * q0 / marginal_water_content + heat_conductivity_dry_soil
    # The result is hcond converted from mcal to cal.
    return hcond / 1000


@cython.boundscheck(False)
@cython.wraparound(False)
def soil_thermal_conductivity_np(
    double cka,
    double ckw,
    double dsand,
    double bsand,
    double dclay,
    double bclay,
    double[:] soil_sand_volume_fraction,
    double[:] soil_clay_volume_fraction,
    double[:] pore_space,
    double[:] field_capacity,
    double[:] marginal_water_content,
    double[:] heat_conductivity_dry_soil,
    double[:] q0,
    double[:] t0,
    uint64_t[:] l0,
):
    x_max = l0.shape[0]
    result = np.zeros(x_max, dtype=np.double)
    cdef double[:] result_view = result
    cdef uint64_t l
    cdef Py_ssize_t i
    for i in prange(x_max, nogil=True):
        l = l0[i]
        result_view[i] = soil_thermal_conductivity(
            cka,
            ckw,
            dsand,
            bsand,
            dclay,
            bclay,
            soil_sand_volume_fraction[l],
            soil_clay_volume_fraction[l],
            pore_space[l],
            field_capacity[l],
            marginal_water_content[l],
            heat_conductivity_dry_soil[l],
            q0[i],
            t0[i]
        )
    return result
