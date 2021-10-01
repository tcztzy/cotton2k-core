import numpy as np


def compute_soil_surface_albedo(
    water_content: float,
    field_capacity: float,
    residual_water_content: float,
    upper_albedo: float,
    lower_albedo: float,
) -> float:
    """Computes albedo of the soil surface
    Less soil water content, higher albedo
    :param water_content:
    :type water_content: float
    :param field_capacity:
    :type field_capacity: float
    :param residual_water_content:
    :type residual_water_content: float
    :param upper_albedo:
    :type upper_albedo: float
    :param lower_albedo:
    :type lower_albedo: float
    """
    if water_content <= residual_water_content:
        soil_surface_albedo = upper_albedo
    elif water_content >= field_capacity:
        soil_surface_albedo = lower_albedo
    else:
        soil_surface_albedo = lower_albedo + (upper_albedo - lower_albedo) * (
            field_capacity - water_content
        ) / (field_capacity - residual_water_content)
    return soil_surface_albedo


def compute_incoming_short_wave_radiation(
    radiation: float, intercepted_short_wave_radiation: float, albedo: float
) -> tuple[float, float, float]:
    """SHORT WAVE RADIATION ENERGY BALANCE
    :return: short wave (global) radiation (ly / sec), global radiation absorbed
             by soil surface, global radiation reflected up to the vegetation
    :rtype: tuple[float, float, float]
    """
    # Division by 41880 (= 698 * 60) converts from Joules per sq m to
    # langley (= calories per sq cm) Or: from Watt per sq m to langley per sec.
    rzero = radiation / 41880  # short wave (global) radiation (ly / sec).
    rss0 = rzero * (
        1 - intercepted_short_wave_radiation
    )  # global radiation after passing through canopy
    return rzero, rss0 * (1 - albedo), rss0 * albedo


def root_psi(soil_psi: float) -> float:
    """This function returns the effect of soil moisture in cell_{l,k} on cotton root
    potential growth rate. It is called from PotentialRootGrowth() and uses the matric
    potential of this cell.

    It is assumed that almost no root growth occurs when the soil is dryer than -p1
    (-20 bars), and root growth rate is maximum at a matric potential of -4 bars
    (p2 - p1) or wetter.

    Effect of soil moisture on root growth (the return value).

    root_psi is computed here as an empirical third degree function, with values
    between 0.02 and 1.
    """
    return min(max(((20.0 + soil_psi) / 16.0) ** 3, 0.02), 1)


def SoilTemOnRootGrowth(t: float) -> float:
    """This function is called from PotentialRootGrowth(), TapRootGrowth() and
    LateralRootGrowth(). It computes the effects of soil temperature on the rate
    growth. It is essentially based on the usage of GOSSYM, but relative values
    are computed here. The computed value returned by this function is between 0 and 1.

    It is assumed that maximum root growth occurs at or above 30 C, and no root growth
    occurs at or below 13.5 C. A quadratic response to temperature between these limits
    is assumed.

    Parameters
    ----------
    t : float
        Soil temperature (C), daily average.

    Examples
    --------
    >>> SoilTemOnRootGrowth(30)
    1
    >>> SoilTemOnRootGrowth(20)
    0.6
    >>> SoilTemOnRootGrowth(14)
    0.0528
    >>> SoilTemOnRootGrowth(13.5)
    0
    """
    if t >= 30:
        return 1
    p = np.polynomial.Polynomial([-2.12, 0.2, -0.0032])
    return min(max(p(t), 0), 1)


def SoilAirOnRootGrowth(psislk: float, pore_space: float, vh2oclk: float) -> float:
    """Calculates the reduction of potential root growth rate in cells with low oxygen
    content (high water content).

    It has been adapted from GOSSYM, but the critical value of soil moisture potential
    for root growth reduction (i.e., water logging conditions) has been changed.

    Arguments
    ---------
    psislk
        value of SoilPsi for this cell.
    pore_space
        value of PoreSpace (v/v) for this layer.
    vh2oclk
        water content (v/v) of this cell

    Examples
    --------
    >>> SoilAirOnRootGrowth(0, 0.1, 0.05)
    1.0
    >>> SoilAirOnRootGrowth(1, 0.1, 0.1)
    0.1
    >>> SoilAirOnRootGrowth(1, 0.1, 0.05)
    1.0
    """
    # Constant parameters:
    p1 = 0
    p2 = 1.0
    p3 = 0.1
    # The following is actually disabled by the choice of the calibration parameters.
    # It may be redefined when more experimental data become available. Reduced root
    # growth when water content is at pore - space saturation (below water table).
    if vh2oclk >= pore_space:
        return p3
    # Effect of oxygen deficiency on root growth (the return value).
    if psislk > p1:
        return p2
    return 1.0


def SoilNitrateOnRootGrowth(  # pylint: disable=unused-argument
    vno3clk: float,
) -> float:
    """Calculates the reduction of potential root growth rate in cells with low nitrate
    content.

    It has been adapted from GOSSYM. It is assumed that root growth is reduced when
    nitrate N content falls below a certain level.

    NOTE: This function actually does nothing. It is disabled by the choice of the
    constant parameters. It may be redefined when more experimental data become
    available.

    Arguments
    ---------
    vno3clk
        VolNo3NContent value for this cell

    Examples
    --------
    >>> SoilNitrateOnRootGrowth(0.07)
    1.0
    """
    return 1.0


def PsiOnTranspiration(psi_average: float) -> float:
    """Computes and returns the effect of the average soil matrix water potential on
    transpiration rate.

    :math:`(\\frac{20 + \\psi}{14})^3`

    Arguments
    ---------
    psi_average
        the average soil water matrix potential, bars.

    Examples
    --------
    >>> PsiOnTranspiration(-6)
    1.0
    >>> PsiOnTranspiration(-10)
    0.36443148688046634
    >>> PsiOnTranspiration(-14.842355901903458)
    0.05
    """
    p = np.polynomial.Polynomial((20 ** 3, 3 * 20 ** 2, 3 * 20, 1)) / 14 ** 3
    return min(max(p(psi_average), 0.05), 1.0)


def SoilTemperatureEffect(tt: float) -> float:
    """Computes the effect of temperature on the rate of mineralization of organic
    mineralizable nitrogen. It is based on GODWIN and JONES (1991).

    Arguments
    ---------
    tt : float
        soil temperature (C).

    Examples
    --------
    >>> SoilTemperatureEffect(12)
    0.050530155584212866
    >>> SoilTemperatureEffect(18)
    0.11009133047465763
    >>> SoilTemperatureEffect(24)
    0.23985877157019808
    >>> SoilTemperatureEffect(30)
    0.5225863839696988
    >>> SoilTemperatureEffect(36)
    1.1385721978093266
    """
    # The following constant parameters are used:
    tfpar1 = 0.010645
    tfpar2 = 0.12979
    # The temperature function of CERES is replaced by the function suggested by
    # Vigil and Kissel (1995):
    #     tfm = 0.010645 * exp(0.12979 * tt)
    # NOTE: tfm = 0.5 for 29.66 C, tfm = 1 for 35 C, tfm = 2 for 40.34 C.
    tfm: float = tfpar1 * np.exp(tfpar2 * tt)
    return min(max(tfm, 0), 2)


def SoilWaterEffect(
    volumetric_water_content: float,
    field_capacity: float,
    volumetric_water_content_at_permanent_wilting_point: float,
    volumetric_water_content_saturated: float,
    xx: float,
) -> float:
    """Computes the effect of soil moisture on the rate of mineralization of organic
    mineralizable nitrogen, and on the rates of urea hydrolysis and nitrification.

    It is based on Godwin and Jones (1991).

    The argument xx is 0.5 when used for mineralization and urea hydrolysis, or 1.0
    when used for nitrification."""

    # the effect of soil moisture on process rate.
    if volumetric_water_content <= field_capacity:
        # Soil water content less than field capacity:
        wf = (
            volumetric_water_content
            - volumetric_water_content_at_permanent_wilting_point
        ) / (field_capacity - volumetric_water_content_at_permanent_wilting_point)
    else:
        # Soil water content more than field capacity:
        wf = 1 - xx * (volumetric_water_content - field_capacity) / (
            volumetric_water_content_saturated - field_capacity
        )

    return max(wf, 0)
