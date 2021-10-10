# pylint: disable=too-many-lines
import datetime
from math import atan, cos, log, pi, sqrt

import numpy as np
import numpy.typing as npt
from scipy import constants

from .meteorology import VaporPressure, clearskyemiss
from .soil import form


def canopy_balance(  # pylint: disable=too-many-arguments,too-many-locals
    etp1: float,
    rlzero: float,
    rsv: float,
    c2: float,
    sf: float,
    so: float,
    thet: float,
    tv: float,
) -> float:
    """Solves the energy balance equations at the foliage / air interface, and computes
    the resulting temperature of the foliage.

    Units for all energy fluxes are: ly / sec.

    Parameters
    ----------
    c2
        multiplier for sensible heat transfer at plant surface.
    etp1
        transpiration (mm / sec).
    rlzero
        incoming long wave radiation (ly / sec).
    rsv
        global radiation absorbed by the vegetation
    sf
        fraction of shaded soil area
    so
        temperature of soil surface (k).
    thet
        air temperature (k).
    tv
        temperature of plant canopy.

    Returns
    -------
    float
        temperature of plant canopy.

    Examples
    --------
    >>> canopy_balance(0.1, 1, 10, 1, 0.8, 12, 15, 18)
    20.938243153794296
    """
    # Constants:
    ef = 0.95  # emissivity of the foliage surface
    eg = 0.95  # emissivity of the soil surface
    stefa1 = 1.38e-12  # stefan-boltsman constant.

    # long wave radiation reaching the canopy
    rlv1 = sf * ef * rlzero + sf * ef * eg * stefa1 * so ** 4  # from sky and soil
    # rlv4 is the multiplier of tv**4 for emitted long wave radiation from vegetation,
    # corrected for the amount reflected back from soil surface and absorbed by foliage
    # This is two-sided (note that when eg = ef = 1, the coefficient corr will be 2)
    corr = 1 + eg / (ef + eg - ef * eg)  # coefficient
    rlv4 = stefa1 * sf * ef * corr
    tvex = 0.0  # previous value of tv
    ccadx = 0.0  # previous value of ccadjust
    # Start iterations for tv:
    for mot in range(50):
        # Emitted long wave radiation from vegetation (cclwe)
        cclwe = rlv4 * tv ** 4
        dcclwe = 4 * rlv4 * tv ** 3  # derivative of cclwe
        # Latent heat flux (hvlat) is computed from the transpiration rate.
        hvlat = (75.5255 - 0.05752 * tv) * etp1
        dhvlat = -0.05752 * etp1  # derivative of hvlat
        # Sensible heat transfer from vegetation
        # average air temperature above soil surface (k) in canopy
        tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)
        senfheat = c2 * (tv - tafk)  # sensible heat transfer from foliage
        dsenfheat = c2 * (1 - 0.6 * sf)  # derivative of senfheat
        # Compute the energy balance at the plant surface (cc), and if it is small
        # enough end the computation.
        cc = (
            cclwe  # (a) long wave emission from vegetation
            + hvlat  # (b) latent heat transfer
            + senfheat  # (c) sensible heat transfer from vegetation to air
            - rlv1  # long wave radiation reaching the vegetation
            - rsv  # global radiation on vegetation
        )
        if abs(cc) < 1e-5:
            return tv  # end iterations for tv
        # If cc is not small enough, compute its derivative by tv (ccp).
        # ccp is the derivative of energy balance at the plant surface (by tv)
        ccp = dcclwe + dhvlat + dsenfheat  # (a)  # (b)  # (c)
        # Correct the canopy temperature by  the ratio of cc to ccp.
        ccadjust = cc / ccp  # adjustment of tv before next iteration
        # If adjustment is small enough, no more iterations are needed.
        if abs(ccadjust) < 2e-3:
            return tv
        # If ccadjust is not the same sign as ccadx, reduce fluctuations
        if mot >= 2 and abs(ccadjust - ccadx) > abs(ccadjust + ccadx):
            ccadjust = (ccadjust + ccadx) / 2
            tv = (tv + tvex) / 2
        ccadjust = min(max(ccadjust, -10), 10)
        tv -= ccadjust
        tvex = tv
        ccadx = ccadjust
    # If reached 50 iterations there must be an error somewhere!
    raise RuntimeError


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


def compute_incoming_long_wave_radiation(
    humidity: float,
    temperature: float,
    cloud_cov: float,
    cloud_cor: float,
) -> float:
    """LONG WAVE RADIATION EMITTED FROM SKY"""
    vp = 0.01 * humidity * VaporPressure(temperature)  # air vapor pressure, KPa.
    ea0 = clearskyemiss(
        vp, temperature + constants.zero_Celsius
    )  # sky emissivity from clear portions of the sky.
    # incoming long wave radiation (ly / sec).
    rlzero = (ea0 * (1 - cloud_cov) + cloud_cov) * constants.sigma * (
        temperature + constants.zero_Celsius
    ) ** 4 - cloud_cor
    return rlzero / constants.calorie / 10_000


# pylint: disable=R0912,R0914,R0915
def SensibleHeatTransfer(tsf, tenviron, height, wndcanp) -> float:
    """Computes the sensible heat transfer coefficient, using the friction potential
    (shear) temperature (thstar), and the surface friction (shear) velocity (ustar) at
    the atmospheric boundary. It is called three times from energy_balance(): for
    canopy or soil surface with their environment.

    Parameters
    ----------
    tenviron
        temperature (K) of the environment - air at 200 cm height for columns with no
        canopy, or tafk when canopy is present .
    tsf
        surface temperature, K, of soil or canopy.
    wndcanp
        wind speed in the canopy (if present), cm s-1.
    height
        canopy height, cm, or zero for soil surface.

    Returns
    -------
    float
        raw sensible heat transfer coefficient
    """
    # Constant values used:
    grav = 980  # acceleration due to gravity (980 cm sec-2).
    s40 = 0.13  # calibration constant.
    s42 = 0.63  # calibration constant.
    stmin = 5  # minimal value of ustar.
    vonkar = 0.40  # Von-Karman constant (0.40).
    zalit1 = 0.0962  # parameter .....
    # Wind velocity not allowed to be less than 100 cm s-1.
    u = max(wndcanp, 100)  # wind speed at 200 cm height, cm / s.
    # Assign initial values to z0 and gtop, and set dt.
    z0 = max(s40 * height, 1)  # surface roughness parameter, cm.
    gtop = log(
        (200 - s42 * height) / z0
    )  # logarithm of ratio of height of measurement to surface roughness parameter.
    dt = tsf - tenviron  # temperature difference.
    # Set approximate initial values for ustar and thstar (to reduce iterations).
    thstar: float  # friction potential (shear) temperature.
    ustar: float  # Surface friction (shear) velocity (cm sec-1).
    if dt >= 0:
        ustar = 1.873 + 0.570172 * dt + 0.07438568 * u
        thstar = -0.05573 * dt + 1.987 / u - 6.657 * dt / u
    else:
        ustar = max(-4.4017 + 1.067 * dt + 0.25957 * u - 0.001683 * dt * u, 5)
        thstar = max(-0.0096 - 0.1149 * dt + 0.0000377 * u + 0.0002367 * dt * u, 0.03)
    tbot1 = (
        tsf  # surface temperature corrected for friction (shear) potential temperature.
    )
    g1: float  # temporary derived variable
    ug1chk: float  # previous value of ug1.
    ug1: float  # ratio of ustar to g1.
    ug1res: float  # residual value of ug1.
    # Start iterations.
    for mtest in range(100):
        ug1chk = 0  # previous value of UG1.
        if mtest > 0:
            # Assign values to tbot1, uchek, thekz, and ug1chk.
            tbot1 = tsf + zalit1 * thstar * pow((ustar * z0 / 15), 0.45) / vonkar
            uchek = ustar  # previous value of ustar.
            thekz = thstar  # previous value of thstar.
            if g1 != 0:
                ug1chk = ustar / g1
        # Compute air temperature at 1 cm,, and compute zl and lstar.
        # nondimensional height.
        if abs(thstar) < 1e-30:
            zl = 0
        else:
            thetmn = (
                tenviron + tbot1
            ) * 0.5  # mean temperature (K) of air and surface.
            lstar = (thetmn * ustar * ustar) / (vonkar * grav * thstar)
            zl = min(max((200 - s42 * height) / lstar, -5), 0.5)
        # Compute g1u, and g2 temporary derived variables.
        if zl > 0:
            g1u = -4.7 * zl
            g2 = max(-6.35135 * zl, -1)
        else:
            tmp1 = pow((1 - 15 * zl), 0.25)  # intermediate variable.
            g1u = (
                2 * log((1 + tmp1) / 2)
                + log((1 + tmp1 * tmp1) / 2)
                - 2 * atan(tmp1 + 1.5708)
            )
            g2 = 2 * log((1 + sqrt(1 - 9 * zl)) / 2)
        g2 = min(g2, gtop)
        # Compute ustar and check for minimum value.
        ustar = max(vonkar * u / (gtop - g1u), stmin)
        # Compute g1 and thstar.
        g1 = 0.74 * (gtop - g2) + zalit1 * pow((ustar * z0 / 0.15), 0.45)
        thstar = -dt * vonkar / g1
        # If more than 30 iterations, reduce fluctuations
        if mtest > 30:
            thstar = (thstar + thekz) / 2
            ustar = (ustar + uchek) / 2

        # Compute ug1 and  ug1res to check convergence
        ug1 = ustar / g1
        if abs(ug1chk) <= 1.0e-30:
            ug1res = abs(ug1)
        else:
            ug1res = abs((ug1chk - ug1) / ug1chk)
        # If ug1 did not converge, go to next iteration.
        if abs(ug1 - ug1chk) <= 0.05 or ug1res <= 0.01:
            return ustar * vonkar / g1
    # Stop simulation if no convergence after 100 iterations.
    raise RuntimeError


class Thermology:  # pylint: disable=E0203,E1101,R0902,R0912,R0914,R0915,W0201
    bclay: float
    bsand: float
    cka: float
    ckw: float
    date: datetime.date
    dclay: float
    dsand: float
    field_capacity: npt.NDArray[np.double]
    heat_conductivity_dry_soil: npt.NDArray[np.double]
    layer_depth: npt.NDArray[np.double]
    marginal_water_content: npt.NDArray[np.double]
    pore_space: npt.NDArray[np.double]
    soil_clay_volume_fraction: npt.NDArray[np.double]
    soil_sand_volume_fraction: npt.NDArray[np.double]
    soil_water_content: npt.NDArray[np.double]

    def soil_thermology(self):
        """The main part of the soil temperature sub-model.

        References
        ----------

        Benjamin, J.G., Ghaffarzadeh, M.R. and Cruse, R.M. 1990. Coupled water and heat
        transport in ridged soils. Soil Sci. Soc. Am. J. 54:963-969.

        Chen, J. 1984. Uncoupled multi-layer model for the transfer of sensible and
        latent heat flux densities from vegetation. Boundary-Layer Meteorology 28:213-
        225.

        Chen, J. 1985. A graphical extrapolation method to determine canopy resistance
        from measured temperature and humidity profiles above a crop canopy. Agric.
        For. Meteorol. 37:75-88.

        Clothier, B.E., Clawson, K.L., Pinter, P.J.Jr., Moran, M.S., Reginato, R.J. and
        Jackson, R.D. 1986. Estimation of soil heat flux from net radiation during the
        growth of alfalfa. Agric. For. Meteorol. 37:319-329.

        Costello, T.A. and Braud, H.J. Jr. 1989. Thermal diffusivity of soil by non-
        linear regression analysis of soil temperature data. Trans. ASAE 32:1281-1286.

        De Vries, D.A. 1963. Thermal properties of soils. In: W.R. Van Wijk (ed)
        Physics of plant environment, North Holland, Amsterdam, pp 210-235.

        Deardorff, J.W. 1978. Efficient prediction of ground surface temperature and
        moisture with inclusion of a layer of vegetation. J. Geophys. Res. 83 (C4):
        1889-1903.

        Dong, A., Prashar, C.K. and Grattan, S.R. 1988. Estimation of daily and hourly
        net radiation. CIMIS Final Report June 1988, pp. 58-79.

        Ephrath, J.E., Goudriaan, J. and Marani, A. 1996. Modelling diurnal patterns of
        air temperature, radiation, wind speed and relative humidity by equations from
        daily characteristics. Agricultural Systems 51:377-393.

        Hadas, A. 1974. Problem involved in measuring the soil thermal conductivity and
        diffusivity in a moist soil. Agric. Meteorol. 13:105-113.

        Hadas, A. 1977. Evaluation of theoretically predicted thermal conductivities of
        soils under field and laboratory conditions. Soil Sci. Soc. Am. J. 41:460-466.

        Hanks, R.J., Austin, D.D. and Ondrechen, W.T. 1971. Soil temperature estimation
        by a numerical method. Soil Sci. Soc. Am. Proc. 35:665-667.

        Hares, M.A. and Novak, M.D. 1992. Simulation of surface energy balance and soil
        temperature under strip tillage: I. Model description. Soil Sci. Soc. Am. J.
        56:22-29.

        Hares, M.A. and Novak, M.D. 1992. Simulation of surface energy balance and soil
        temperature under strip tillage: II. Field test. Soil Sci. Soc. Am. J. 56:29-36.

        Horton, E. and Wierenga, P.J. 1983. Estimating the soil heat flux from
        observations of soil temperature near the surface. Soil Sci. Soc. Am. J. 47:
        14-20.

        Horton, E., Wierenga, P.J. and Nielsen, D.R. 1983. Evaluation of methods for
        determining apparent thermal diffusivity of soil near the surface. Soil Sci.
        Soc. Am. J. 47:25-32.

        Horton, R. 1989. Canopy shading effects on soil heat and water flow. Soil Sci.
        Soc. Am. J. 53:669-679.

        Horton, R., and Chung, S-O, 1991. Soil Heat Flow. Ch. 17 in: Hanks, J., and
        Ritchie, J.T., (Eds.) Modeling Plant and Soil Systems. Am. Soc. Agron.,
        Madison, WI, pp 397-438.

        Iqbal, M. 1983. An Introduction to Solar Radiation. Academic Press.

        Kimball, B.A., Jackson, R.D., Reginato, R.J., Nakayama, F.S. and Idso, S.B.
        1976. Comparison of field-measured and calculated soil heat fluxes. Soil Sci.
        Soc. Am. J. 40:18-28.

        Lettau, B. 1971. Determination of the thermal diffusivity in the upper layers
        of a natural ground cover. Soil Sci. 112:173-177.

        Monin, A.S. 1973. Boundary layers in planetary atmospheres. In: P. Morrel
        (ed.), Dynamic meteorology, D. Reidel Publishing Company, Boston, pp. 419-458.

        Spitters, C.J.T., Toussaint, H.A.J.M. and Goudriaan, J. 1986. Separating the
        diffuse and direct component of global radiation and its implications for
        modeling canopy photosynthesis. Part I. Components of incoming radiation.
        Agric. For. Meteorol. 38:217-229.

        Wierenga, P.J. and de Wit, C.T. 1970. Simulation of heat flow in soils. Soil
        Sci. Soc. Am. Proc. 34:845-848.

        Wierenga, P.J., Hagan, R.M. and Nielsen, D.R. 1970. Soil temperature profiles
        during infiltration and redistribution of cool and warm irrigation water. Water
        Resour. Res. 6:230-238.

        Wierenga, P.J., Nielsen, D.R. and Hagan, R.M. 1969. Thermal properties of soil
        based upon field and laboratory measurements. Soil Sci. Soc. Am. Proc. 33:354-
        360.
        """
        # Compute dts, the daily change in deep soil temperature (C), as a site-
        # dependent function of Daynum.
        dts = (
            2
            * pi
            * self.site_parameters[10]
            / 365
            * cos(
                2
                * pi
                * (self.date.timetuple().tm_yday - self.site_parameters[11])
                / 365
            )
        )
        # Define iter1 and dlt for hourly time step.
        iter1 = 24  # number of iterations per day.
        dlt = 3600  # time (seconds) of one iteration.
        kk = 1  # number of soil columns for executing computations.
        # If there is no canopy cover, no horizontal heat flux is assumed, kk = 1.
        # Otherwise it is equal to the number of columns in the slab.
        shadeav = 0  # average shaded area in all shaded soil columns.
        # emerge_switch defines the type of soil temperature computation.
        if self.emerge_switch > 1:
            shadetot = 0  # sum of shaded area in all shaded soil columns.
            nshadedcol = 0  # number of at least partially shaded soil columns.
            kk = 20
            for k in range(20):
                if self.irradiation_soil_surface[k] <= 0.99:
                    shadetot += 1 - self.irradiation_soil_surface[k]
                    nshadedcol += 1

            if nshadedcol > 0:
                shadeav = shadetot / nshadedcol
        # es and ActualSoilEvaporation are computed as the average for the whole soil
        # slab, weighted by column widths.
        es = 0  # potential evaporation rate, mm day-1
        self.actual_soil_evaporation = 0
        # Start hourly loop of iterations.
        for ihr in range(iter1):
            # Update the temperature of the last soil layer (lower boundary conditions).
            self.deep_soil_temperature += dts * dlt / 86400
            etp0 = 0  # actual transpiration (mm s-1) for this hour
            if self.evapotranspiration > 0.000001:
                etp0 = (
                    self.actual_transpiration
                    * self.hours[ihr].ref_et
                    / self.evapotranspiration
                    / dlt
                )
            # Compute vertical transport for each column
            for k in range(kk):
                #  Set hourly_soil_temperature for the lowest soil layer.
                self.hourly_soil_temperature[
                    ihr, 40 - 1, k
                ] = self.deep_soil_temperature
                # Compute transpiration from each column, weighted by its relative
                # shading.
                etp1 = 0  # actual hourly transpiration (mm s-1) for a column.
                if shadeav > 0.000001:
                    etp1 = etp0 * (1 - self.irradiation_soil_surface[k]) / shadeav
                ess = 0  # evaporation rate from surface of a soil column (mm / sec).
                # The potential evaporation rate (escol1k) from a column is the sum of
                # the radiation component of the Penman equation(es1hour), multiplied
                # by the relative radiation reaching this column, and the wind and
                # vapor deficit component of the Penman equation (es2hour).
                # potential evaporation fron soil surface of a column, mm per hour.
                escol1k = (
                    self.hours[ihr].et1 * self.irradiation_soil_surface[k]
                    + self.hours[ihr].et2
                )
                es += escol1k * self.column_width[k]
                # Compute actual evaporation from soil surface. update water content of
                # the soil cell, and add to daily sum of actual evaporation.
                evapmax = (
                    0.9
                    * (self.soil_water_content[0, k] - self.thad[0])
                    * 10
                    * self.layer_depth[0]
                )  # maximum possible evaporatio from a soil cell near the surface.
                escol1k = min(evapmax, escol1k)
                self.soil_water_content[0, k] -= 0.1 * escol1k / self.layer_depth[0]
                self.actual_soil_evaporation += escol1k * self.column_width[k]
                ess = escol1k / dlt
                # Call self.energy_balance to compute soil surface and canopy
                # temperature.
                self.energy_balance(ihr, k, ess, etp1)
            # Compute soil temperature flux in the vertical direction.
            # Assign iv = 1, layer = 0, nn = nl.
            iv = 1  # indicates vertical (=1) or horizontal (=0) flux.
            nn = 40  # number of array members for heat flux.
            layer = 0  # soil layer number
            # Loop over kk columns, and call SoilHeatFlux().
            for k in range(kk):
                self.soil_heat_flux(dlt, iv, nn, layer, k, ihr)
            # If no horizontal heat flux is assumed, make all array members of
            # hourly_soil_temperature equal to the value computed for the first column.
            # Also, do the same for array memebers of soil_water_content.
            if self.emerge_switch <= 1:
                self.hourly_soil_temperature[ihr, :, :] = self.hourly_soil_temperature[
                    ihr, :, 0
                ][:, None].repeat(20, axis=1)
                self.soil_water_content[0, :] = self.soil_water_content[0, 0]
            # Compute horizontal transport for each layer

            # Compute soil temperature flux in the horizontal direction, when
            # emerge_switch = 2.
            # Assign iv = 0 and nn = nk. Start loop for soil layers, and call
            # soil_heat_flux.
            if self.emerge_switch > 1:
                iv = 0
                nn = 20
                for l in range(40):
                    layer = l
                    self.soil_heat_flux(dlt, iv, nn, layer, l, ihr)
            # Compute average temperature of foliage, in degrees C. The average is
            # weighted by the canopy shading of each column, only columns which are
            # shaded 5% or more by canopy are used.
            tfc = 0  # average foliage temperature, weighted by shading in each column
            shading = 0  # sum of shaded area in all shaded columns, to compute TFC
            for k in range(20):
                if self.irradiation_soil_surface[k] <= 0.95:
                    tfc += (self.foliage_temperature[k] - 273.161) * (
                        1 - self.irradiation_soil_surface[k]
                    )
                    shading += 1 - self.irradiation_soil_surface[k]
            if shading >= 0.01:
                tfc /= shading
            # If emergence date is to be simulated, call predict_emergence().
            if self.emerge_switch == 0 and self.date >= self.plant_date:
                emerge_date = self.predict_emergence(
                    self.plant_date, ihr, self.plant_row_column
                )
                if emerge_date is not None:
                    self.emerge_date = emerge_date
                    self.emerge_switch = 2
            if ihr < 23:
                self.hourly_soil_temperature[ihr + 1] = self.hourly_soil_temperature[
                    ihr
                ]
        # At the end of the day compute actual daily evaporation and its cumulative sum
        if kk == 1:
            es /= self.column_width[1]
            self.actual_soil_evaporation /= self.column_width[1]
        else:
            es /= self.row_space
            self.actual_soil_evaporation /= self.row_space
        # compute daily averages.
        self.soil_temperature = self.hourly_soil_temperature.mean(axis=0)

    def energy_balance(self, ihr, k, ess, etp1):
        """Solves the energy balance equations at the soil surface, and at the
        foliage / atmosphere interface. It computes the resulting temperatures of the
        soil surface and the plant canopy.

        Units for all energy fluxes are: cal cm-2 sec-1.
        It is called from SoilTemperature(), on each hourly time step and for each soil
        column.
        It calls functions clearskyemiss(), VaporPressure(), SensibleHeatTransfer(),
        soil_surface_balance() and canopy_balance()

        :param ihr: the time of day in hours.
        :param k: soil column number.
        :param ess: evaporation from surface of a soil column (mm / sec).
        :param etp1: actual transpiration rate (mm / sec).
        :param sf: fraction of shaded soil area
        """
        hour = self.hours[ihr]
        # Constants used:
        wndfac = 0.60  # Ratio of wind speed under partial canopy cover.
        # proportion of short wave radiation (on fully shaded soil surface) intercepted
        # by the canopy.
        cswint = 0.75
        # Set initial values
        sf = 1 - self.irradiation_soil_surface[k]
        thet = hour.temperature + 273.161  # air temperature, K
        so, so2, so3 = self.hourly_soil_temperature[
            ihr, :3, k
        ]  # soil surface temperature, K
        # Compute soil surface albedo (based on Horton and Chung, 1991):
        ag = compute_soil_surface_albedo(
            self.soil_water_content[0, k],
            self.field_capacity[0],
            self.thad[0],
            self.site_parameters[15],
            self.site_parameters[16],
        )

        rzero, rss, rsup = compute_incoming_short_wave_radiation(
            hour.radiation, sf * cswint, ag
        )
        rlzero = compute_incoming_long_wave_radiation(
            hour.humidity, hour.temperature, hour.cloud_cov, hour.cloud_cor
        )

        # Set initial values of canopy temperature and air temperature in canopy.
        tv: float  # temperature of plant foliage (K)
        tafk: float  # temperature (K) of air inside the canopy.
        if sf < 0.05:  # no vegetation
            tv = thet
            tafk = thet
        # Wind velocity in canopy is converted to cm / s.
        wndhr = hour.wind_speed * 100  # wind speed in cm /sec
        # air density * specific heat at constant pressure = 0.24 * 2 * 1013 / 5740
        rocp: float
        # divided by tafk.
        c2: float  # multiplier for sensible heat transfer (at plant surface).
        rsv: float  # global radiation absorbed by the vegetation
        if sf >= 0.05:  # a shaded soil column
            tv = self.foliage_temperature[k]  # vegetation temperature
            # Short wave radiation intercepted by the canopy:
            rsv = (
                rzero * (1 - hour.albedo) * sf * cswint  # from above
                + rsup * (1 - hour.albedo) * sf * cswint  # reflected from soil surface
            )
            # Air temperature inside canopy is the average of soil, air, and plant
            # temperatures, weighted by 0.1, 0.3, and 0.6, respectively.
            tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)

            # Call SensibleHeatTransfer to compute sensible heat transfer coefficient.
            # Factor 2.2 for sensible heat transfer: 2 sides of leaf plus stems and
            # petioles.
            # sensible heat transfer coefficient for soil
            varcc = SensibleHeatTransfer(
                tv, tafk, self.plant_height, wndhr
            )  # canopy to air
            rocp = 0.08471 / tafk
            c2 = 2.2 * sf * rocp * varcc
        soold = so  # previous value of soil surface temperature
        tvold = tv  # previous value of vegetation temperature
        # Starting iterations for soil and canopy energy balance
        for menit in range(30):
            soold = so
            wndcanp = (
                1 - sf * (1 - wndfac)
            ) * wndhr  # estimated wind speed under canopy
            # Call SensibleHeatTransfer() to compute sensible heat transfer for soil
            # surface to air
            tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)
            # sensible heat transfer coefficientS for soil
            varc = SensibleHeatTransfer(so, tafk, 0, wndcanp)
            rocp = 0.08471 / tafk
            hsg = (
                rocp * varc
            )  # multiplier for computing sensible heat transfer soil to air.
            # Call soil_surface_balance() for energy balance in soil surface / air
            # interface.
            so, so2, so3 = self.soil_surface_balance(
                ihr, k, ess, rlzero, rss, sf, hsg, so, so2, so3, thet, tv
            )

            if sf >= 0.05:
                # This section executed for shaded columns only.
                tvold = tv
                # Compute canopy energy balance for shaded columns
                tv = canopy_balance(etp1, rlzero, rsv, c2, sf, so, thet, tv)
                if menit >= 10:
                    # The following is used to reduce fluctuations.
                    so = (so + soold) / 2
                    tv = (tv + tvold) / 2
            if abs(tv - tvold) <= 0.05 and abs(so - soold) <= 0.05:
                break
        else:
            raise RuntimeError  # stop simulation if more than 30 iterations.
        # After convergence - set global variables for the following temperatures:
        if sf >= 0.05:
            self.foliage_temperature[k] = tv
        self.hourly_soil_temperature[ihr, :3, k] = [so, so2, so3]

    soil_heat_flux_numiter: int = 0  # number of this iteration.

    # equal to the dl array in a columnn, or wk in a row.
    dz: npt.NDArray[np.double] = np.zeros(40)
    # heat capacity of soil layer (cal cm-3 oC-1).
    hcap: npt.NDArray[np.double] = np.zeros(40)
    # array of previous soil temperatures.
    ts0: npt.NDArray[np.double] = np.zeros(40)
    # array of soil temperatures.
    ts1: npt.NDArray[np.double] = np.zeros(40)

    def heat_balance(self, nn):
        """Checks and corrects the heat balance in the soil soil cells, within a soil
        layer. It is called by function soil_heat_flux only for horizontal flux.

        The implicit part of the solution may cause some deviation in the total heat
        sum to occur. This module corrects the heat balance if the sum of absolute
        deviations is not zero, so that the total amount of heat in the array does not
        change. The correction is proportional to the difference between the previous
        and present heat amounts.

        Arguments
        ---------
        nn
            the number of soil cells in this layer or column.
        """
        # Sum of absolute value of differences in heat content in the array between
        # beginning and end of this time step.
        dabs = np.abs(self.ts1[:nn] - self.ts0[:nn]).sum()
        dev = 0  # Sum of differences of heat amount in soil.
        for i in range(nn):
            dev += self.dz[i] * self.hcap[i] * (self.ts1[i] - self.ts0[i])
        if dabs > 0:
            for i in range(nn):
                self.ts1[i] = self.ts1[i] - abs(self.ts1[i] - self.ts0[i]) * dev / (
                    dabs * self.dz[i] * self.hcap[i]
                )

    def soil_heat_flux(self, dlt, iv, nn, layer, n0, ihr):  # pylint: disable=R0913
        """Computes heat flux in one direction between soil cells.

        NOTE: the units are:
        thermal conductivity = cal cm-1 s-1 oC-1;
        heat capacity = cal cm-3 oC-1;
        thermal diffusivity = cm2 s-1;
        ckx and cky are dimensionless;

        Arguments
        ---------
        dlt
            time (seconds) of one iteration.
        iv
            = 1 for vertical flux, = 0 for horizontal flux.
        layer
            soil layer number.
        n0
            number of layer or column of this array
        nn
            number of soil cells in the array.
        """
        # Constant parameters:
        beta1 = 0.90  # weighting factor for the implicit method of computation.
        ca = 0.0003  # heat capacity of air (cal cm-3 oC-1).
        # Set soil layer number l (needed to define heat_capacity_soil_solid,
        # pore_space, soil_thermal_conductivity).
        # Compute for each soil cell the heat capacity and heat diffusivity.

        l = layer  # soil layer number.
        q1: npt.NDArray[np.double]  # array of water content.
        # array of thermal diffusivity of soil cells (cm2 s-1).
        asoi: npt.NDArray[np.double]
        if iv == 1:
            q1 = self.soil_water_content[:, n0]
            self.ts1 = self.hourly_soil_temperature[ihr, :, n0]
            self.dz = self.layer_depth[:]
        else:
            q1 = self.soil_water_content[n0]
            self.ts1 = self.hourly_soil_temperature[ihr, n0]
            self.dz = self.column_width[:]
        self.hcap = (
            q1 + self.heat_capacity_soil_solid[l] + (self.pore_space[l] - q1) * ca
        )
        asoi = (
            np.array(
                [
                    self.soil_thermal_conductivity(q1[i], self.ts1[i], l)
                    for i in range(nn)
                ]
            )
            / self.hcap
        )
        # The numerical solution of the flow equation is a combination of the implicit
        # method (weighted by beta1) and the explicit method (weighted by 1-beta1).
        dltt: float  # computed time step required.
        avdif = np.zeros(
            40, dtype=np.double
        )  # average thermal diffusivity between adjacent cells.
        dy = np.zeros(
            40, dtype=np.double
        )  # array of distances between centers of adjacent cells (cm).
        dltmin = dlt  # minimum time step for the explicit solution.
        avdif[0] = 0
        dy[0] = 0
        for i in range(1, nn):
            # Compute average diffusivities avdif between layer i and the previous
            # (i-1), and dy(i), distance (cm) between centers of layer i and the
            # previous (i-1)
            avdif[i] = (asoi[i] + asoi[i - 1]) / 2
            dy[i] = (self.dz[i - 1] + self.dz[i]) / 2
            # Determine the minimum time step required for the explicit solution.
            dltt = 0.2 * dy[i] * self.dz[i] / avdif[i] / (1 - beta1)
            if dltt < dltmin:
                dltmin = dltt
        # Use time step of dlt1 seconds, for iterx iterations
        iterx = int(dlt / dltmin)  # computed number of iterations.
        if dltmin < dlt:
            iterx += 1
        dlt1 = dlt / iterx  # computed time (seconds) of an iteration.
        # start iterations. Store temperature data in array ts0. count iterations.
        for _ in range(iterx):
            for i in range(nn):
                self.ts0[i] = self.ts1[i]
                if iv == 1:
                    l = i
                asoi[i] = (
                    self.soil_thermal_conductivity(q1[i], self.ts1[i], l) / self.hcap[i]
                )
                if i > 0:
                    avdif[i] = (asoi[i] + asoi[i - 1]) / 2
            self.soil_heat_flux_numiter += 1
            # The solution of the simultaneous equations in the implicit method
            # alternates between the two directions along the arrays. The reason for
            # this is because the direction of the solution may cause some cumulative
            # bias. The counter numiter determines the direction of the solution.
            # arrays used for the implicit numerical solution.
            cau = np.zeros(40, dtype=np.double)
            dau = np.zeros(40, dtype=np.double)
            # nondimensional diffusivities to next and previous layers.
            ckx: float
            cky: float
            # used for computing the implicit solution.
            vara: float
            varb: float
            if self.soil_heat_flux_numiter % 2 == 0:
                # 1st direction of computation, for an even iteration number:
                dau[0] = 0
                cau[0] = self.ts1[0]
                # Loop from the second to the last but one soil cells. Compute
                # nondimensional diffusivities to next and previous layers.
                for i in range(1, nn - 1):
                    ckx = avdif[i + 1] * dlt1 / (self.dz[i] * dy[i + 1])
                    cky = avdif[i] * dlt1 / (self.dz[i] * dy[i])
                    # Correct value of layer 1 for explicit heat movement to/from
                    # layer 2
                    if i == 1:
                        cau[0] = (
                            self.ts1[0]
                            - (1 - beta1)
                            * (self.ts1[0] - self.ts1[1])
                            * cky
                            * self.dz[1]
                            / self.dz[0]
                        )
                    vara = 1 + beta1 * (ckx + cky) - beta1 * ckx * dau[i - 1]
                    dau[i] = beta1 * cky / vara
                    varb = self.ts1[i] + (1 - beta1) * (
                        cky * self.ts1[i - 1]
                        + ckx * self.ts1[i + 1]
                        - (cky + ckx) * self.ts1[i]
                    )
                    cau[i] = (varb + beta1 * ckx * cau[i - 1]) / vara
                # Correct value of last layer (nn-1) for explicit heat movement to/from
                # layer nn-2
                self.ts1[nn - 1] = (
                    self.ts1[nn - 1]
                    - (1 - beta1)
                    * (self.ts1[nn - 1] - self.ts1[nn - 2])
                    * ckx
                    * self.dz[nn - 2]
                    / self.dz[nn - 1]
                )
                # Continue with the implicit solution
                for i in range(nn - 2, -1, -1):
                    self.ts1[i] = dau[i] * self.ts1[i + 1] + cau[i]
            else:
                # Alternate direction of computation for odd iteration number
                dau[nn - 1] = 0
                cau[nn - 1] = self.ts1[nn - 1]
                for i in range(nn - 2, 0, -1):
                    ckx = avdif[i + 1] * dlt1 / (self.dz[i] * dy[i + 1])
                    cky = avdif[i] * dlt1 / (self.dz[i] * dy[i])
                    if i == nn - 2:
                        cau[nn - 1] = (
                            self.ts1[nn - 1]
                            - (1 - beta1)
                            * (self.ts1[nn - 1] - self.ts1[nn - 2])
                            * ckx
                            * self.dz[nn - 2]
                            / self.dz[nn - 1]
                        )
                    vara = 1 + beta1 * (ckx + cky) - beta1 * cky * dau[i + 1]
                    dau[i] = beta1 * ckx / vara
                    varb = self.ts1[i] + (1 - beta1) * (
                        ckx * self.ts1[i + 1]
                        + cky * self.ts1[i - 1]
                        - (cky + ckx) * self.ts1[i]
                    )
                    cau[i] = (varb + beta1 * cky * cau[i + 1]) / vara
                self.ts1[0] = (
                    self.ts1[0]
                    - (1 - beta1)
                    * (self.ts1[0] - self.ts1[1])
                    * cky
                    * self.dz[1]
                    / self.dz[0]
                )
                for i in range(1, nn):
                    self.ts1[i] = dau[i] * self.ts1[i - 1] + cau[i]
            # Call heat_balance to correct quantitative deviations caused by the imlicit
            # part of the solution.
            self.heat_balance(nn)
        # Set values of SoiTemp
        if iv == 1:
            self.hourly_soil_temperature[ihr, :, n0] = self.ts1
        else:
            self.hourly_soil_temperature[ihr, n0, :] = self.ts1

    def soil_surface_balance(  # pylint: disable=R0913
        self, ihr, k, ess, rlzero, rss, sf, hsg, so, so2, so3, thet, tv
    ) -> tuple[float, float, float]:
        """Solves the energy balance equations at the soil surface, and computes the
        resulting temperature of the soil surface.

        Units for all energy fluxes are: cal cm-2 sec-1.

        :param ihr: the time in hours.
        :param k: soil column number.
        :param ess: evaporation from soil surface (mm / sec).
        :param rlzero: incoming long wave
        :param rss: global radiation absorbed by soil surface
        :param sf: fraction of shaded soil area
        :param hsg: multiplier for computing sensible heat transfer from soil to air.
        :param thet: air temperature (K).
        :param tv: temperature of plant canopy (K).
        """
        # Constants:
        ef = 0.95  # emissivity of the foliage surface
        eg = 0.95  # emissivity of the soil surface
        stefa1 = 1.38e-12  # Stefan-Boltsman constant.
        # Long wave radiation reaching the soil:
        rls1: float  # long wave energy reaching soil surface
        if sf >= 0.05:  # haded column
            rls1 = (
                # from sky on unshaded soil surface
                (1 - sf) * eg * rlzero
                # from foliage on shaded soil surface
                + sf * eg * ef * stefa1 * tv ** 4
            )
        else:
            rls1 = eg * rlzero  # from sky in unshaded column
        # rls4 is the multiplier of so**4 for emitted long wave radiation from soil
        rls4 = eg * stefa1
        bbex: float  # previous value of bbadjust.
        soex = so  # previous value of so.
        # Start itrations for soil surface enegy balance.
        for mon in range(50):
            # Compute latent heat flux from soil evaporation: convert from mm sec-1 to
            # cal cm-2 sec-1. Compute derivative of hlat
            # hlat is the energy used for evaporation from soil surface (latent heat)
            hlat = (75.5255 - 0.05752 * so) * ess
            dhlat = -0.05752 * ess  # derivative of hlat
            # Compute the thermal conductivity of layers 1 to 3 by
            # soil_thermal_conductivity().
            # heat conductivity of n-th soil layer in cal / (cm sec deg).
            rosoil1 = self.soil_thermal_conductivity(
                self.soil_water_content[0, k], so, 1
            )
            rosoil2 = self.soil_thermal_conductivity(
                self.soil_water_content[1, k], so2, 2
            )
            rosoil3 = self.soil_thermal_conductivity(
                self.soil_water_content[2, k], so3, 3
            )
            surface_layer_depth = self.layer_depth[:3]
            # Compute average rosoil between layers 1 to 3,and heat transfer from soil
            # surface to 3rd soil layer.
            # multiplier for heat flux between 1st and 3rd soil layers.
            rosoil = (
                (np.array([rosoil1, rosoil2, rosoil3]) * surface_layer_depth).sum()
                / surface_layer_depth.sum()
                / (np.array([0.5, 1, 0.5]) * surface_layer_depth).sum()
            )
            # bbsoil is the heat energy transfer by conductance from surface to inner
            bbsoil = rosoil * (so - so3)
            # emtlw is emitted long wave radiation from soil surface
            emtlw = rls4 * pow(so, 4)
            # Sensible heat transfer and its derivative
            # average air temperature above soil surface (K)
            tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)
            senheat = hsg * (so - tafk)  # sensible heat transfer from soil surface
            dsenheat = hsg * (1 - sf * 0.1)  # derivative of senheat
            # Compute the energy balance bb. (positive direction is upward)
            bb = (
                emtlw  # long wave radiation emitted from soil surface
                - rls1  # long wave radiation reaching the soil surface
                + bbsoil  # (b) heat transfer from soil surface to next soil layer
                + hlat  # (c) latent heat transfer
                - rss  # global radiation reaching the soil surface
                + senheat  # (d) heat transfer from soil surface to air
            )

            if abs(bb) < 1e-5:
                return so, so2, so3  # end computation for so
            # If bb is not small enough, compute its derivative by so.

            demtlw = (
                4 * rls4 * so ** 3
            )  # The derivative of emitted long wave radiation (emtlw)
            # Compute derivative of bbsoil
            sop001 = so + 0.001  # soil surface temperature plus 0.001
            # heat conductivity of 1st soil layer for so+0.001
            rosoil1p = self.soil_thermal_conductivity(
                self.soil_water_content[0, k], sop001, 1
            )
            # rosoil for so+0.001
            rosoilp = (
                (np.array([rosoil1p, rosoil2, rosoil3]) * surface_layer_depth).sum()
                / surface_layer_depth.sum()
                / (np.array([0.5, 1, 0.5]) * surface_layer_depth).sum()
            )
            drosoil = (rosoilp - rosoil) / 0.001  # derivative of rosoil
            dbbsoil = rosoil + drosoil * (so - so3)  # derivative of bbsoil
            # The derivative of the energy balance function
            bbp = demtlw + dbbsoil + dhlat + dsenheat  # (a)  # (b)  # (c)  # (d)
            # Correct the upper soil temperature by the ratio of bb to bbp.
            # the adjustment of soil surface temperature before next iteration
            bbadjust = bb / bbp
            # If adjustment is small enough, no more iterations are needed.
            if abs(bbadjust) < 0.002:
                return so, so2, so3
            # If bbadjust is not the same sign as bbex, reduce fluctuations
            if mon <= 1:
                bbex = 0
            elif mon >= 2:
                if abs(bbadjust + bbex) < abs(bbadjust - bbex):
                    bbadjust = (bbadjust + bbex) / 2
                    so = (so + soex) / 2

            bbadjust = min(max(bbadjust, -10), 10)

            so -= bbadjust
            so2 += (so - soex) / 2
            so3 += (so - soex) / 3
            soex = so
            bbex = bbadjust
            mon += 1
        # If (mon >= 50) send message on error and end simulation.
        raise RuntimeError(
            "\n".join(
                (
                    "Infinite loop in soil_surface_balance(). Abnormal stop!!",
                    "Daynum, ihr, k = %s %3d %3d" % (self.date.isoformat(), ihr, k),
                    "so = %10.3g" % so,
                    "so2 = %10.3g" % so2,
                    "so3 = %10.3g" % so3,
                )
            )
        )

    def soil_thermal_conductivity(self, q0: float, t0: float, l0: int) -> float:
        """Computes and returns the thermal conductivity of the soil
        (cal cm-1 s-1 oC-1). It is based on the work of De Vries(1963).

        Arguments
        ---------
        l0
            soil layer.
        q0
            volumetric soil moisture content.
        t0
            soil temperature (K).
        """
        # Convert soil temperature to degrees C.
        tcel = t0 - 273.161  # soil temperature, in C.
        # Compute cpn, the apparent heat conductivity of air in soil pore spaces, when
        # saturated with water vapor, using a function of soil temperature, which
        # changes linearly between 36 and 40 C.
        # effect of temperature on heat conductivity of air saturated with water vapor.
        bb: float
        if tcel <= 36:
            bb = 0.06188
        elif 36 < tcel <= 40:
            bb = 0.06188 + (tcel - 36) * (0.05790 - 0.06188) / (40 - 36)
        else:
            bb = 0.05790
        # apparent heat conductivity of air in soil pore spaces, when it is saturated
        # with water vapor.
        cpn = self.cka + 0.05 * np.exp(bb * tcel)
        # Compute xair, air content of soil per volume, from soil porosity and moisture
        # content.
        # Compute thermal conductivity
        # (a) for wet soil (soil moisture higher than field capacity),
        # (b) for less wet soil.
        # In each case compute first ga, and then dair.
        # air content of soil, per volume.
        xair = max(self.pore_space[l0] - q0, 0)
        dair: float  # aggregation factor for air in soil pore spaces.
        ga: float  # shape factor for air in pore spaces.
        hcond: float  # computed heat conductivity of soil, mcal cm-1 s-1 oc-1.
        if q0 >= self.field_capacity[l0]:
            # (a) Heat conductivity of soil wetter than field capacity.
            ga = 0.333 - 0.061 * xair / self.pore_space[l0]
            dair = form(cpn, self.ckw, ga)
            hcond = (
                q0 * self.ckw
                + self.dsand * self.bsand * self.soil_sand_volume_fraction[l0]
                + self.dclay * self.bclay * self.soil_clay_volume_fraction[l0]
                + dair * cpn * xair
            ) / (
                q0
                + self.dsand * self.soil_sand_volume_fraction[l0]
                + self.dclay * self.soil_clay_volume_fraction[l0]
                + dair * xair
            )
        else:
            # (b) For soil less wet than field capacity, compute also ckn (heat
            # conductivity of air in the soil pores).
            qq: float  # soil water content for computing ckn and ga.
            ckn: float  # heat conductivity of air in pores in soil.
            qq = max(q0, self.marginal_water_content[l0])
            ckn = self.cka + (cpn - self.cka) * qq / self.field_capacity[l0]
            ga = 0.041 + 0.244 * (qq - self.marginal_water_content[l0]) / (
                self.field_capacity[l0] - self.marginal_water_content[l0]
            )
            dair = form(ckn, self.ckw, ga)
            hcond = (
                qq * self.ckw
                + self.dsand * self.bsand * self.soil_sand_volume_fraction[l0]
                + self.dclay * self.bclay * self.soil_clay_volume_fraction[l0]
                + dair * ckn * xair
            ) / (
                qq
                + self.dsand * self.soil_sand_volume_fraction[l0]
                + self.dclay * self.soil_clay_volume_fraction[l0]
                + dair * xair
            )
            # When soil moisture content is less than the limiting value
            # marginal_water_content, modify the value of hcond.
            if qq <= self.marginal_water_content[l0]:
                hcond = (
                    hcond - self.heat_conductivity_dry_soil[l0]
                ) * q0 / self.marginal_water_content[
                    l0
                ] + self.heat_conductivity_dry_soil[
                    l0
                ]
        # The result is hcond converted from mcal to cal.
        return hcond / 1000
