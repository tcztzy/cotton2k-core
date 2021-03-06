import datetime
from enum import Enum, auto

import numpy as np


class SoilRunoff(Enum):
    Low = auto()
    Moderate = auto()
    High = auto()


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
        value of self.pore_space (v/v) for this layer.
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


def wcond(  # pylint: disable=too-many-arguments
    q: float,
    qr: float,
    qsat: float,
    beta: float,
    saturated_hyd_cond: float,
    pore_space: float,
) -> float:
    """Computes soil water hydraulic conductivity for a given value of soil water
    content, using the Van-Genuchten equation. The units of the computed conductivity
    are the same as the given saturated conductivity.

    Arguments
    ---------
    beta
        parameter of the van-genuchten equation.
    saturated_hyd_cond
        saturated hydraulic conductivity (at qsat).
    pore_space
        pore space volume.
    q
        soil water content, cm3 cm-3.
    qr
        residual water content, cm3 cm-3.
    qsat
        saturated water content, cm3 cm-3.
    """
    # For very low values of water content (near the residual water content) wcond is 0
    if (q - qr) < 0.0001:
        return 0
    # Water content for saturated conductivity is minimum of self.pore_space and qsat.

    # For very high values of water content (exceeding the saturated water content or
    # pore space) conductivity is saturated_hydraulic_conductivity.
    xsat = min(qsat, pore_space)
    if q >= xsat:
        return saturated_hyd_cond
    # The following equation is used (in FORTRAN notation):
    #   WCOND = CONDSAT * ((Q-QR)/(XSAT-QR))**0.5
    #           * (1-(1-((Q-QR)/(XSAT-QR))**(1/GAMA))**GAMA)**2
    gama = 1 - 1 / beta
    gaminv = 1 / gama
    sweff = (q - qr) / (xsat - qr)  # intermediate variable (effective water content).
    acoeff = (1.0 - sweff ** gaminv) ** gama  # intermediate variable
    bcoeff = (1.0 - acoeff) ** 2  # intermediate variable
    return np.sqrt(sweff) * bcoeff * saturated_hyd_cond


def qpsi(psi: float, qr: float, qsat: float, alpha: float, beta: float) -> float:
    """This function computes soil water content (cm3 cm-3) for a given value of matrix
    potential, using the Van-Genuchten equation.

    Arguments
    ---------
    psi
        soil water matrix potential (bars).
    qr
        residual water content, cm3 cm-3.
    qsat
        saturated water content, cm3 cm-3.
    alpha, beta
        parameters of the van-genuchten equation.

    Returns
    -------

    """
    # For very high values of PSI, saturated water content is assumed.
    # For very low values of PSI, air-dry water content is assumed.
    if psi >= -0.00001:
        return qsat
    if psi <= -500000:
        return qr
    # The soil water matric potential is transformed from bars (psi) to cm in positive
    # value (psix).
    psix = 1000 * np.abs(psi + 0.00001)
    # The following equation is used (in FORTRAN notation):
    #   QPSI = QR + (QSAT-QR) / (1 + (ALPHA*PSIX)**BETA)**(1-1/BETA)
    gama = 1 - 1 / beta
    term = 1 + (alpha * psix) ** beta  # intermediate variable
    swfun = qr + (qsat - qr) / term ** gama  # computed water content
    return np.maximum(qr + 0.0001, swfun)


def psiq(q: float, qr: float, qsat: float, alpha: float, beta: float) -> float:
    """Computes soil water matric potential (in bars) for a given value of soil water
    content, using the Van-Genuchten equation.

    Arguments
    ---------
    q
        soil water content, cm3 cm-3.
    qr
        residual water content, cm3 cm-3.
    qsat
        saturated water content, cm3 cm-3.
    alpha, beta
        parameters of the van-genuchten equation.

    Returns
    -------
    float
    """
    # For very low values of water content (near the residual water content) psiq is
    # -500000 bars, and for saturated or higher water content psiq is -0.00001 bars.
    if (q - qr) < 0.00001:
        return -500000
    if q >= qsat:
        return -0.00001
    # The following equation is used (FORTRAN notation):
    #   PSIX = (((QSAT-QR) / (Q-QR))**(1/GAMA) - 1) **(1/BETA) / ALPHA
    gama = 1 - 1 / beta
    gaminv = 1 / gama
    term = ((qsat - qr) / (q - qr)) ** gaminv  # intermediate variable
    psix = (term - 1) ** (1 / beta) / alpha
    psix = max(psix, 0.01)
    # psix (in cm) is converted to bars (negative value).
    psix = (0.01 - psix) * 0.001
    return min(max(psix, -500000), -0.00001)


def PsiOsmotic(q: float, qsat: float, ec: float) -> float:
    """Computes soil water osmotic potential (in bars, positive value).

    Arguments
    ---------
    q
        soil water content, cm3 cm-3.
    qsat
        saturated water content, cm3 cm-3.
    ec
        electrical conductivity of saturated extract (mmho/cm)

    Returns
    -------
    float
    """
    if ec > 0:
        return min(0.36 * ec * qsat / q, 6)
    return 0


def SoilMechanicResistance(rtimpdmin: float) -> float:
    """Calculates soil mechanical resistance of cell l,k. It is computed on the basis
    of parameters read from the input and calculated in RootImpedance().

    The function has been adapted, without change, from the code of GOSSYM. Soil
    mechanical resistance is computed as an empirical function of bulk density and
    water content. It should be noted, however, that this empirical function is based
    on data for one type of soil only, and its applicability for other soil types is
    questionable. The effect of soil moisture is only indirectly reflected in this
    function. A new module (root_psi) has therefore been added in COTTON2K to simulate
    an additional direct effect of soil moisture on root growth.

    The minimum value of rtimpd of this and neighboring soil cells is used to compute
    rtpct. The code is based on a segment of RUTGRO in GOSSYM, and the values of the p1
    to p3 parameters are based on GOSSYM usage:

    Arguments
    ---------
    rtimpdmin

    Returns
    -------

    """
    p1 = 1.046
    p2 = 0.034554
    p3 = 0.5

    # effect of soil mechanical resistance on root growth (the return value).
    rtpct = p1 - p2 * rtimpdmin
    return min(max(rtpct, p3), 1)


def form(c0: float, d0: float, g0: float) -> float:
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


class SoilProcedure:  # pylint: disable=too-few-public-methods,W0201,E1101
    runoff: float  # in mm

    @property
    def effective_rain(self):
        return max(self.rain - self.runoff, 0)

    def soil_procedures(self):
        """Manages all the soil related processes, and is executed once each day."""
        # The following constant parameters are used:
        cpardrip = 0.2
        cparelse = 0.4
        DripWaterAmount = 0  # amount of water applied by drip irrigation
        # Call SimulateRunoff() only if the daily rainfall is more than 2 mm.
        # NOTE: this is modified from the original GOSSYM - RRUNOFF routine. It is
        # called here for rainfall only, but it is not activated when irrigation is
        # applied.
        rainToday = self.meteor[self.date]["rain"]  # the amount of rain today, mm
        runoffToday = 0  # amount of runoff today, mm
        if rainToday >= 2.0:
            runoffToday = self.simulate_runoff()
        self.runoff = runoffToday
        # Call function ApplyFertilizer() for nitrogen fertilizer application.
        self.apply_fertilizer(self.row_space, self.plant_population)
        # amount of water applied by non-drip irrigation or rainfall
        # Check if there is rain on this day
        WaterToApply = self.effective_rain
        # When water is added by an irrigation defined in the input: update the amount
        # of applied water.
        if self.date in self.irrigation:
            irrigation = self.irrigation[self.date]
            if irrigation.get("method", 0) == 2:
                DripWaterAmount += irrigation["amount"]
                self.drip_x = irrigation.get("drip_x", 0)
                self.drip_y = irrigation.get("drip_y", 0)
            else:
                WaterToApply += irrigation["amount"]
        # The following will be executed only after plant emergence
        if self.date >= self.emerge_date and self.emerge_switch > 0:
            # computes roots capable of uptakefor each soil cell
            self.roots_capable_of_uptake()
            self.average_soil_psi = self.average_psi(
                self.row_space
            )  # function computes the average matric soil water
            # potential in the root zone, weighted by the roots-capable-of-uptake.
            self.water_uptake(
                self.row_space, self.per_plant_area
            )  # function  computes water and nitrogen uptake by plants.
        if WaterToApply > 0:
            # For rain or surface irrigation.
            # The number of iterations is computed from the thickness of the first soil
            # layer.
            noitr = int(cparelse * WaterToApply / (self.layer_depth[0] + 2) + 1)
            # the amount of water applied, mm per iteration.
            applywat = WaterToApply / noitr
            # The following subroutines are called noitr times per day:
            # If water is applied, state.gravity_flow() is called when the method of
            # irrigation is not by drippers, followed by CapillaryFlow().
            for _ in range(noitr):
                self.gravity_flow(applywat)
                self.capillary_flow(noitr)
        if DripWaterAmount > 0:
            # For drip irrigation.
            # The number of iterations is computed from the volume of the soil cell in
            # which the water is applied.
            noitr = int(
                cpardrip * DripWaterAmount / (self.cell_area[self.drip_y, self.drip_x])
                + 1
            )
            # the amount of water applied, mm per iteration.
            applywat = DripWaterAmount / noitr
            # If water is applied, drip_flow() is called followed by CapillaryFlow().
            for _ in range(noitr):
                self.drip_flow(applywat, self.row_space)
                self.capillary_flow(noitr)
        # When no water is added, there is only one iteration in this day.
        if WaterToApply + DripWaterAmount <= 0:
            self.capillary_flow(1)

    def simulate_runoff(self):
        """Executed on each day with raifall more than 2 mm. It computes the runoff and
        the retained portion of the rainfall.

        NOTE: This function is based on the code of GOSSYM. No changes have been made
        from the original GOSSYM code (except translation to Python). It has not been
        validated by actual field measurement.

        It calculates the portion of rainfall that is lost to runoff, and reduces
        rainfall to the amount which is actually infiltrated into the soil. It uses the
        soil conservation service method of estimating runoff.

        References
        ----------
        Brady, Nyle C. 1984. The nature and properties of soils, 9th ed. Macmillan
        Publishing Co.

        Schwab, Frevert, Edminster, and Barnes. 1981. Soil and water conservation
        engineering, 3rd ed. John Wiley & Sons, Inc.

        Returns
        -------
        float
            the amount of water (mm) lost by runoff.
        """
        iGroup: SoilRunoff
        d01: float  # Adjustment of curve number for soil groups A,B,C.

        # Infiltration rate is estimated from the percent sand and percent clay in the
        # Ap layer.
        # If clay content is greater than 35%, the soil is assumed to have a higher
        # runoff potential, if clay content is less than 15% and sand is greater than
        # 70%, a lower runoff potential is assumed. Other soils (loams) assumed
        # moderate runoff potential. No 'impermeable' (group D) soils are assumed.
        # References: Schwab, Brady.

        if (
            self.soil_sand_volume_fraction[0] > 0.70
            and self.soil_clay_volume_fraction[0] < 0.15
        ):
            # Soil group A = 1, low runoff potential
            iGroup = SoilRunoff.Low
            d01 = 1.0
        elif self.soil_clay_volume_fraction[0] > 0.35:
            # Soil group C = 3, high runoff potential
            iGroup = SoilRunoff.High
            d01 = 1.14
        else:
            # Soil group B = 2, moderate runoff potential
            iGroup = SoilRunoff.Moderate
            d01 = 1.09
        # Loop to accumulate 5-day antecedent rainfall (mm) which will affect the
        # soil's ability to accept new rainfall. This also includes all irrigations.
        PreviousWetting = 0  # five day total (before this day) of rain and irrigation
        for i in range(5):
            d = self.date - datetime.timedelta(days=i)
            if d in self.irrigation:
                # mm water applied on this day by irrigation
                PreviousWetting += self.irrigation[d]["amount"]
            if d in self.meteor:
                PreviousWetting += self.meteor[d]["rain"]

        d02: float  # Adjusting curve number for antecedent rainfall conditions.
        if PreviousWetting < 3:
            # low moisture, low runoff potential.
            d02 = {
                SoilRunoff.Low: 0.71,
                SoilRunoff.Moderate: 0.78,
                SoilRunoff.High: 0.83,
            }[iGroup]
        elif PreviousWetting > 53:
            # wet conditions, high runoff potential.
            d02 = {
                SoilRunoff.Low: 1.24,
                SoilRunoff.Moderate: 1.15,
                SoilRunoff.High: 1.10,
            }[iGroup]
        else:
            # moderate conditions
            d02 = 1.00
        # Assuming straight rows, and good cropping practice:
        crvnum = 78.0  # Runoff curve number, unadjusted for moisture and soil type.
        crvnum *= d01 * d02  # adjusted curve number
        # maximum potential difference between rainfall and runoff.
        d03 = 25400 / crvnum - 254
        return (
            0
            if self.rain <= 0.2 * d03
            else (self.rain - 0.2 * d03) ** 2 / (self.rain + 0.8 * d03)
        )

    def roots_capable_of_uptake(self):
        """Computes the weight of roots capable of uptake for all soil cells."""
        # the indices for the relative capability of uptake (between 0 and 1) of water
        # and nutrients by root age classes.
        cuind = np.array((1, 0.5, 0))
        weights = self.root_weights * cuind
        weights[self.root_weights <= 1e-15] = 0
        self.root_weight_capable_uptake = weights.sum(axis=2)

    cumulative_drained_water: float = 0  # mm

    def gravity_flow(self, applywat):
        """Computes the water redistribution in the soil or surface irrigation (by
        flooding or sprinklers).

        Arguments
        ---------
        applywat
            amount of water applied, mm.
        """
        # Add the applied amount of water to the top soil cell of each column.
        self.soil_water_content[0, :] += 0.10 * applywat / self.layer_depth[0]
        self.cumulative_drained_water += self.drain() * 10 / self.row_space
