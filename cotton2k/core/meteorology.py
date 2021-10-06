import datetime
from calendar import isleap
from collections import defaultdict
from math import acos, cos, degrees, exp, pi, radians, sin, sqrt, tan
from typing import DefaultDict

import numpy as np
from scipy import constants

from .utils import date2doy

METEOROLOGY: DefaultDict = defaultdict(lambda: defaultdict(dict))
TZ_WIDTH = 15  # timezone width in degree


def compute_day_length(coordinate: tuple[float, float], date: datetime.date) -> dict:
    lat, lon = coordinate
    xday: float = radians(360 * date2doy(date) / (365 + int(isleap(date.year))))
    declination: float = (
        0.006918
        - 0.399912 * cos(xday)
        + 0.070257 * sin(xday)
        - 0.006758 * cos(2 * xday)
        + 0.000907 * sin(2 * xday)
        - 0.002697 * cos(3 * xday)
        + 0.001480 * sin(3 * xday)
    )
    exday: datetime.timedelta = datetime.timedelta(
        hours=degrees(
            0.000075
            + 0.001868 * cos(xday)
            - 0.032077 * sin(xday)
            - 0.014615 * cos(2 * xday)
            - 0.04089 * sin(2 * xday)
        )
        / TZ_WIDTH
    )

    solar_noon: datetime.datetime = (
        datetime.datetime.combine(
            date,
            datetime.time(12),
            datetime.timezone(datetime.timedelta(hours=lon / TZ_WIDTH)),
        ).astimezone(
            datetime.timezone(
                datetime.timedelta(hours=(lon + TZ_WIDTH / 2) // TZ_WIDTH)
            )
        )
        - exday
    )
    ht: float = -tan(radians(lat)) * tan(declination)
    ht = min(max(ht, -1), 1)

    day_length = datetime.timedelta(hours=degrees(2 * acos(ht)) / TZ_WIDTH)
    sunr = solar_noon - day_length / 2
    return {
        "day_length": day_length,
        "solar_noon": solar_noon,
        "sunr": sunr,
        "suns": sunr + day_length,
        "declination": declination,
        "tmpisr": 1367
        * (
            1.00011
            + 0.034221 * cos(xday)
            + 0.00128 * sin(xday)
            + 0.000719 * cos(2 * xday)
            + 0.000077 * sin(2 * xday)
        ),
    }


def compute_hourly_wind_speed(  # pylint: disable=too-many-arguments
    ti: float, wind: float, t1: float, t2: float, t3: float, wnytf: float
) -> float:
    """Computes the hourly values of wind speed (m/sec), estimated from the measured
    total daily wind run.

    The algorithm is described by Ephrath et al. (1996). It is based on the following
    assumptions:

    Although the variability of wind speed during any day is very large, the diurnal
    wind speed curves appear to be characterized by the following repetitive pattern:
    increase in wind speed from time `t1` in the morning to time `t2` in the afternoon,
    decrease from `t2` to `t3` in the evening, and a low constant wind speed at night,
    from `t3` to `t1` in the next day.

    The values of `t1`, `t2`, and `t3` have been determined in the calling routine:
    `t1` is `SitePar(1)` hours after sunrise, `t2` is `SitePar(2)` hours after solar
    noon, and `t3` is `SitePar(3)` hours after sunset. These parameters are site-
    specific. They are 1, 3, and 0, respectively, for the San Joaquin valley of
    California and for Arizona, and 1, 4, and 2, respectively, for the coastal plain of
    israel.

    The wind speed during the night, from `t3` to `t1` next day (`wmin`) is assumed to
    be proportional to the daily total wind run. The ratio `wnytf` is also site-
    specific, `SitePar(4)`, (0.008 for San Joaquin and Arizona, 0.0025 for the coastal
    plain of Israel). wmin is the minimum wind speed from `t1` to `t3`.

    `wtday` is computed by subtracting the daily integral of wmin, after converting it
    from m/sec to km/day, from the total daily wind run (wndt).

    `wmax`, the maximum wind speed at time `t2` (minus `wmin`), is computed from wtday
    and converted to m/sec.

    `daywnd` from t1 to t2 is now computed as an increasing sinusoidal function from
    `wmin` to `wmin + wmax`, and it is computed from `t2` to `t3` as a decreasing
    sinusoidal function from `wmin + wmax` to `wmin`.

    Reference
    ---------
    Ephrath, J.E., Goudriaan, J., Marani, A., 1996. Modelling diurnal patterns of air
    temperature, radiation wind speed and relative humidity by equations from daily
    characteristics. Agricultural Systems 51, 377–393.
    https://doi.org/10.1016/0308-521X(95)00068-G

    Arguments
    ---------
    t1
        the hour at which day-time wind begins to blow.
    t2
        the hour at which day-time wind speed is maximum.
    t3
        the hour at which day-time wind ceases to blow.
    ti
        the hour of the day.
    wind
        wind speed (m / s).
    wnytf
        Factor for estimating night-time wind (from time t3 to time t1 next day).
    """
    # constants related to t1, t2, t3 :
    sf1 = 4 * (t2 - t1)
    sf2 = 4 * (t3 - t2)
    wmin = wind * wnytf  # the constant minimum wind speed during the night (m/sec).
    wtday = wind - wmin  # integral of wind run from t1 to t3, minus wmin (km).
    wmax = (
        wtday * 2 * pi * 24 / (sf1 + sf2)
    )  # the maximum wind speed (m per sec), above wmin.
    if t1 <= ti < t2:
        return wmin + wmax * sin(2 * pi * (ti - t1) / sf1)
    if t2 <= ti < t3:
        return wmin + wmax * sin(2 * pi * (ti - (2 * t2 - t3)) / sf2)
    return wmin


def dayrh(tt: float, tdew: float) -> float:
    """Computes the hourly values of relative humidity, using the hourly air and dew
    point temperatures.

    If the estimated dew point is higher than the actual air temperature, its value is
    taken as the air temperature (relative humidity 100%).

    The relative humidity is calculated as the percentage ratio of the saturated vapor
    pressure at dew point temperature and the saturated vapor pressure at actual air
    temperature.

    Reference:

    Ephrath, J.E., Goudriaan, J. and Marani, A. 1996. Modelling diurnal patterns of air
    temperature, radiation, wind speed and relative humidity by equations from daily
    characteristics. Agricultural Systems 51:377-393.

    :param tt: air temperature C at this time of day.
    :type tt: float
    :param tdew: dew point temperature C at this time of day.
    :type tdew: float
    :return: relative humidity
    :rtype: float
    """
    td = min(
        tt, tdew
    )  # the dew point temperature (C), is assumed to be tt if tt < tdew.
    esvp = VaporPressure(tt)  # the saturated vapor pressure in the air (mbar).
    vpa = VaporPressure(td)  # the actual vapor pressure in the air (mbar).
    relative_humidity = 100 * vpa / esvp  # relative humidity at this time of day, %.
    return min(100, max(1, relative_humidity))


def radiation(radsum: float, sinb: float, c11: float) -> float:
    """
    Function radiation() computes the hourly values of global radiation, in W m-2,
    using the measured daily total global radiation.

    The algorithm follows the paper of Spitters et al. (1986). It assumes
    that atmospheric transmission of radiation is lower near the margins of
    the daylight period, because of an increase in the path length through
    the atmosphere at lower solar heights. Radiation is therefore assumed to be
    proportional to sinb * (1 + c11 * sinb), where the value of c11 is set as 0.4 .

    Input arguments:
    radsum - daily radiation integral.
    sinb - sine of the solar elevation.
    c11 - constant parameter (0.4).

    References:

    Spitters, C.J.T., Toussaint, H.A.J.M. and Goudriaan, J. 1986.
    Separating the diffuse and direct component of global radiation and
    its implications for modeling canopy photosynthesis. Part I.
    Components of incoming radiation. Agric. For. Meteorol. 38:217-229.

    Ephrath, J.E., Goudriaan, J. and Marani, A. 1996. Modelling
    diurnal patterns of air temperature, radiation, wind speed and
    relative humidity by equations from daily characteristics.
    Agricultural Systems 51:377-393.
    """
    return 0 if sinb <= 0 else radsum * sinb * (1 + c11 * sinb)


def delta(tk: float, svp: float) -> float:
    """Computes the slope of the saturation vapor pressure (svp, in mb) versus air
    temperature (tk, in K). This algorithm is the same as used by CIMIS."""
    a = 10 ** (-0.0304 * tk)
    b = tk ** 2
    c = 10 ** (-1302.88 / tk)
    return (6790.5 - 5.02808 * tk + 4916.8 * a * b + 174209 * c) * svp / b


def gamma(elev: float, tt: float) -> float:
    """Computes the psychometric constant at elevation (elev), m above sea level, and
    air temperature, C (tt). This algorithm is the same as used by CIMIS."""
    bp = np.polynomial.Polynomial([101.3, -0.01152, 5.44e-7])(
        elev
    )  # barometric pressure, KPa, at this elevation.
    return 0.000646 * bp * (1 + 0.000946 * tt)


def refalbed(isrhr: float, rad: float, coszhr: float, sunahr: float) -> float:
    """Computes the reference crop albedo, using the CIMIS algorithm.

    This algorithm is described by Dong et al. (1988). Albedo is estimated as a
    function of sun elevation above the horizon (suna) for clear or partly cloudy sky
    (rasi >= 0.375) and when the sun is at least 10 degrees above the horizon.

    For very cloudy sky, or when solar altitude is below 10 degrees, the following
    albedo value is assumed: (p4)+ 0.26

    Reference:
    Dong, A., Prashar, C.K. and Grattan, S.R. 1988. Estimation of daily and hourly net
    radiation. CIMIS Final Report June 1988, pp. 58-79.

    Parameters
    ----------
    isrhr
        hourly extraterrestrial radiation in W m-2 .
    rad
        hourly global radiation in W / m-2 .
    coszhr
        cosine of sun angle from zenith.
    sunahr
        sun angle from horizon, degrees.
    """
    p1 = 0.00158  # p1 ... p4 are constant parameters.
    p2 = 0.386
    p3 = 0.0188
    p4 = 0.26
    rasi = rad / isrhr if isrhr > 0 else 0  # ratio of rad to isrhr
    if coszhr > 0.1736 and rasi >= 0.375:
        refalb = p1 * sunahr + p2 * exp(-p3 * sunahr)  # the reference albedo
        return min(refalb, p4)
    return p4


def clcor(  # pylint: disable=too-many-arguments
    ihr: int,
    ck: float,
    isrhr: float,
    coszhr: float,
    day_length: float,
    rad: float,
    solar_noon: float,
) -> float:
    """Computes cloud type correction, using the CIMIS algorithm.

    Global variables used: DayLength, Radiation[], SolarNoon, pi

    NOTE: This algorithm is described by Dong et al. (1988). ck is the cloud-type
    correction used in the Monteith equation for estimating net radiation. The value of
    this correction depends on site and time of year. Regional ck values for California
    are given by Dong et al. (1988). In the San Joaquin valley of California ck is
    almost constant from April to October, with an average value of 60. The value of ck
    is site-dependant, assumed to be constant during the growing season.

    The daily ck is converted to an hourly value for clear or partly cloudy sky
    (rasi >= 0.375) and when the sun is at least 10 degrees above the horizon.

    Evening, night and early morning cloud type correction is temporarily assigned 0.
    It is later assigned the values of first or last non-zero values (in the calling
    routine).

    Reference:
    Dong, A., Prashar, C.K. and Grattan, S.R. 1988. Estimation of daily and hourly net
    radiation. CIMIS Final Report June 1988, pp. 58-79.

    Parameters
    ----------
    ck
        cloud type correction factor (data for this location).
    coszhr
        cosine of sun angle from zenith.
    ihr
        time of day, hours.
    isrhr
        hourly extraterrestrial radiation in W m-2 .
    """
    # ratio of Radiation to isrhr.
    rasi = rad / isrhr if isrhr > 0 else 0
    if coszhr >= 0.1736 and rasi >= 0.375:
        angle = (
            pi * (ihr - solar_noon + 0.5) / day_length
        )  # hour angle (from solar noon) in radians.
        return ck * pi / 2 * cos(angle)
    return 0


def cloudcov(radihr: float, isr: float, cosz: float) -> float:
    """Computes cloud cover for this hour from radiation data, using the CIMIS
    algorithm. The return value is cloud cover ratio (0 to 1)

    This algorithm is described by Dong et al. (1988). Cloud cover fraction is
    estimated as a function of the ratio of actual solar radiation to extraterrestrial
    radiation. The parameters of this function have been based on California data.

    The equation is for daylight hours, when the sun is not less than 10 degrees above
    the horizon (coszhr > 0.1736).

    Reference:
    Dong, A., Prashar, C.K. and Grattan, S.R. 1988. Estimation of daily and hourly net
    radiation. CIMIS Final Report June 1988, pp. 58-79.

    Parameters
    ----------
    radihr
        hourly global radiation in W m-2 .
    isr
        hourly extraterrestrial radiation in W m-2 .
    cosz
        cosine of sun angle from zenith.
    """
    p1 = 1.333  # p1, p2, p3 are constant parameters.
    p2 = 1.7778
    p3 = 0.294118
    rasi = radihr / isr if isr > 0 else 0  # ratio of radihr to isr.

    if cosz > 0.1736 and rasi <= p1 / p2:
        # computed cloud cover.
        return max((p1 - p2 * max(rasi, 0.375)) ** p3, 0)
    return 0


def sunangle(
    ti: float,
    latitude: float,
    declination: float,
    solar_noon: float,
) -> tuple[float, float]:
    """Computes sun angle for any time of day.

    Parameters
    ----------
    ti = time of day, hours.

    Returns
    -------
    coszhr
        cosine of sun angle from zenith for this hour.
    sunahr
        sun angle from horizon, degrees.
    """
    # The latitude is converted to radians (xlat).
    xlat = radians(latitude)
    # amplitude of the sine of the solar height, computed as the product of cosines of
    # latitude and declination angles.
    cd = cos(xlat) * cos(declination)
    # seasonal offset of the sine of the solar height, computed as the product of sines
    # of latitude and declination angles.
    sd = sin(xlat) * sin(declination)
    hrangle = radians(15.0 * (ti - solar_noon))  # hourly angle converted to radians
    coszhr = sd + cd * cos(hrangle)
    if coszhr <= 0:
        return 0, 0
    if coszhr >= 1:
        return 1, 90
    sunahr = abs(degrees(acos(coszhr)) - 90)
    return coszhr, sunahr


def VaporPressure(tt: float) -> float:
    """Computes the water vapor pressure in the air (in KPa units) as a function of the
    air at temperature tt (C). This equation is widely used.

    Arguments
    ---------
    tt
        temperature in C.

    Returns
    -------
    float
        vapor pressure in KPa.

    Examples
    --------
    >>> VaporPressure(20)
    2.338022964146756
    """
    return 0.61078 * exp(17.269 * tt / (tt + 237.3))


def clearskyemiss(vp: float, tk: float) -> float:
    """
    Estimates clear sky emissivity for long wave radiation.

    References
    ----------

    Idso, S.B., 1981. A set of equations for full spectrum and 8- to 14-μm and 10.5- to
    12.5-μm thermal radiation from cloudless skies. Water Resources Research 17,
    295–304. https://doi.org/10.1029/WR017i002p00295

    Arguments
    ---------
    vp
        vapor pressure of the air in KPa
    tk
        air temperature in K.

    Returns
    -------
    float
        clear sky emissivity for long wave radiation.

    Examples
    --------
    >>> clearskyemiss(2.33, 20+273.15)
    0.9312521744252138
    """
    vp1 = vp * 10  # vapor pressure of the air in mbars.

    return min(0.70 + 5.95e-5 * vp1 * exp(1500 / tk), 1)


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


def tdewest(maxt: float, site5: float, site6: float) -> float:
    """Estimates the approximate daily average dewpoint temperature when it is not
    available.

    Arguments
    ---------
    maxt
        maximum temperature of this day.

    Examples
    --------
    >>> tdewest(22, 10, 18)
    10.8
    >>> tdewest(18, 10, 18)
    10
    >>> tdewest(40, 10, 18)
    18
    """

    if maxt <= 20:
        return site5
    if maxt >= 40:
        return site6
    return ((40 - maxt) * site5 + (maxt - 20) * site6) / 20


class Meteorology:  # pylint: disable=E1101,R0914,W0201
    date: datetime.date
    meteor: dict
    site_parameters: list[float]

    def daily_meteorology(self):
        declination: float  # daily declination angle, in radians
        sunr: float  # time of sunrise, hours.
        suns: float  # time of sunset, hours.
        tmpisr: float  # extraterrestrial radiation, \frac{W}{m^2}
        hour = datetime.timedelta(hours=1)
        result = compute_day_length((self.latitude, self.longitude), self.date)
        declination = result["declination"]
        zero = result["sunr"].replace(hour=0, minute=0, second=0, microsecond=0)
        sunr = (result["sunr"] - zero) / hour
        suns = (result["suns"] - zero) / hour
        tmpisr = result["tmpisr"]
        self.solar_noon = (result["solar_noon"] - zero) / hour
        self.day_length = result["day_length"] / hour

        xlat = self.latitude * pi / 180  # latitude converted to radians.
        cd = cos(xlat) * cos(declination)  # amplitude of the sine of the solar height.
        sd = sin(xlat) * sin(
            declination
        )  # seasonal offset of the sine of the solar height.
        # The computation of the daily integral of global radiation (from sunrise to
        # sunset) is based on Spitters et al. (1986).
        c11 = 0.4  # constant parameter
        radsum: float
        if abs(sd / cd) >= 1:
            radsum = 0
        else:
            # dsbe is the integral of sinb * (1 + c11 * sinb) from sunrise to sunset.
            dsbe = (
                acos(-sd / cd) * 24 / pi * (sd + c11 * sd * sd + 0.5 * c11 * cd * cd)
                + 12 * (cd * (2 + 3 * c11 * sd)) * sqrt(1 - (sd / cd) * (sd / cd)) / pi
            )
            # The daily radiation integral is computed for later use in function
            # Radiation.
            # Daily radiation intedral is converted from langleys to Watt m - 2,
            # and divided by dsbe.
            # 11.630287 = 1000000 / 3600 / 23.884
            radsum = self.meteor[self.date]["irradiation"] * 11.630287 / dsbe
        # Parameters for the daily wind function are now computed:
        # the hour at which wind begins to blow (SitePar(1) hours after sunrise).
        t1 = sunr + self.site_parameters[1]
        # the hour at which wind speed is maximum (SitePar(2) hours after solar noon).
        t2 = self.solar_noon + self.site_parameters[2]
        # the hour at which wind stops to blow (SitePar(3) hours after sunset).
        t3 = suns + self.site_parameters[3]
        wnytf = self.site_parameters[
            4
        ]  # used for estimating night time wind (from time t3 to time t1 next day).

        for ihr in range(24):
            hour = self.hours[ihr]
            ti = ihr + 0.5
            sinb = sd + cd * cos(pi * (ti - self.solar_noon) / 12)
            hour.radiation = radiation(radsum, sinb, c11)
            hour.temperature = self.daytmp(ti, sunr, suns)
            hour.dew_point = self.calculate_hourly_dew_point(
                ti,
                hour.temperature,
                sunr,
                self.solar_noon + self.site_parameters[8],
            )
            hour.humidity = dayrh(hour.temperature, hour.dew_point)
            hour.wind_speed = compute_hourly_wind_speed(
                ti, self.meteor[self.date]["wind"] * 1000 / 86400, t1, t2, t3, wnytf
            )
        # Compute average daily temperature, using function AverageAirTemperatures.
        self.calculate_average_temperatures()
        # Compute potential evapotranspiration.
        self.compute_evapotranspiration(declination, tmpisr)

    def dew_point_range(self, tmax, tmin):
        """range of dew point temperature."""
        return max(
            self.site_parameters[12]
            + self.site_parameters[13] * tmax
            + self.site_parameters[14] * tmin,
            0,
        )

    def calculate_dew_point(self, t, tmax, tmin, tdew):
        tdrange = self.dew_point_range(tmax, tmin)
        tdmin = tdew - tdrange / 2  # minimum of dew point temperature.
        return tdmin + tdrange * (t - tmin) / (tmax - tmin)

    def calculate_hourly_dew_point(
        self,
        time,
        temperature,
        sunrise,
        hmax,
    ):
        """Computes the hourly values of dew point temperature from average dew-point
        and the daily estimated range. This range is computed as a regression on
        maximum and minimum temperatures.

        Arguments
        ---------
        hmax
            time of maximum air temperature
        """
        yesterday = self._sim.meteor[self.date - datetime.timedelta(days=1)]
        today = self._sim.meteor[self.date]
        tomorrow = self._sim.meteor[self.date + datetime.timedelta(days=1)]
        if time <= sunrise:
            # from midnight to sunrise
            tmax = yesterday["tmax"]
            tmin = today["tmin"]
            tdew = yesterday["tdew"]
        elif time <= hmax:
            # from sunrise to hmax
            tmax = today["tmax"]
            tmin = today["tmin"]
            tdew = today["tdew"]
        # from hmax to midnight
        else:
            tmax = today["tmax"]
            tmin = tomorrow["tmin"]
            tdew = tomorrow["tdew"]
        return self.calculate_dew_point(temperature, tmax, tmin, tdew)

    def daytmp(
        self,
        ti,
        sunr,
        suns,
    ) -> float:
        """Computes and returns the hourly values of air temperature, using the
        measured daily maximum and minimum.

        The algorithm is described in Ephrath et al. (1996). It is based on the
        following assumptions:

        1. The time of minimum daily temperature is at sunrise.
        2. The time of maximum daily temperature is SitePar[8] hours after solar
        noon.

        Many models assume a sinusoidal curve of the temperature during the day,
        but actual data deviate from the sinusoidal curve in the following
        characteristic way: a faster increase right after sunrise, a near plateau
        maximum during several hours in the middle of the day, and a rather fast
        decrease by sunset. The physical reason for this is a more efficient mixing
        of heated air from ground level into the atmospheric boundary layer, driven
        by strong lapse temperature gradients buoyancy.

        NOTE: **will be used for "power" as in Fortran notation**.

        A first order approximation is

            daytmp = tmin + (tmax-tmin) * st * tkk / (tkk + daytmp - tmin)

        where

            st = sin(pi * (ti - SolarNoon + dayl / 2) / (dayl + 2 * SitePar[8]))

        Since daytmp appears on both sides of the first equation, it can be solved
        and written explicitly as:

            daytmp = tmin - tkk/2 + 0.5 * sqrt(tkk**2 + 4 * amp * tkk * st)

        where the amplitude of tmin and tmax is calculated as
            amp = (tmax - tmin) * (1 + (tmax - tmin) / tkk)
        This ensures that temperature still passes through tmin and tmax values.
        The value of tkk was determined by calibration as 15.

        This algorithm is used for the period from sunrise to the time of maximum
        temperature, hmax. A similar algorithm is used for the time from hmax to
        sunset, but the value of the minimum temperature of the next day
        (mint_tomorrow) is used instead of mint_today.

        Night air temperature is described by an exponentially declining curve.
        For the time from sunset to mid-night:

            daytmp = (mint_tomorrow - sst * exp((dayl - 24) / tcoef)
                    + (sst - mint_tomorrow) * exp((suns - ti) / tcoef))
                    / (1 - exp((dayl - 24) / tcoef))

        where tcoef is a time coefficient, determined by calibration as 4, sst is
        the sunset temperature, determined by the daytime equation as:

            sst = mint_tomorrow - tkk / 2 + 0.5 * sqrt(tkk**2 + 4 * amp * tkk * sts)

        where

            sts  = sin(pi * dayl / (dayl + 2 * SitePar[8]))
            amp = (tmax - mint_tomorrow) * (1 + (tmax - mint_tomorrow) / tkk)

        For the time from midnight to sunrise, similar equations are used, but the
        minimum temperature of this day (mint_today) is used instead of
        mint_tomorrow, and the maximum temperature of the previous day
        (maxt_yesterday) is used instead of maxt_today. Also, (suns-ti-24) is used
        for the time variable instead of (suns-ti).

        These exponential equations for night-time temperature ensure that the
        curve will be continuous with the daytime equation at sunset, and will pass
        through the minimum temperature at sunrise.

        Reference
        ---------
        Ephrath, J.E., Goudriaan, J. and Marani, A. 1996. Modelling diurnal patterns of
        air temperature, radiation, wind speed and relative humidity by equations from
        daily characteristics. Agricultural Systems 51:377-393.
        """
        # The temperature increase at which the sensible heat flux isdoubled, in
        # comparison with the situation without buoyancy.
        tkk = 15
        tcoef = 4  # time coefficient for the exponential part of the equation.
        hmax = self.solar_noon + self.site_parameters[8]  # hour of maximum temperature
        yesterday = self.meteor[self.date - datetime.timedelta(days=1)]
        today = self.meteor[self.date]
        tomorrow = self.meteor[self.date + datetime.timedelta(days=1)]

        amp: float  # amplitude of temperatures for a period.
        sst: float  # the temperature at sunset.
        st: float  # computed from time of day, used for daytime temperature.
        sts: float  # intermediate variable for computing sst.

        if ti <= sunr:
            # from midnight to sunrise
            amp = (yesterday["tmax"] - today["tmin"]) * (
                1 + (yesterday["tmax"] - today["tmin"]) / tkk
            )
            sts = sin(
                pi * self.day_length / (self.day_length + 2 * self.site_parameters[8])
            )
            # compute temperature at sunset:
            sst = today["tmin"] - tkk / 2 + 0.5 * sqrt(tkk * tkk + 4 * amp * tkk * sts)
            return (
                today["tmin"]
                - sst * exp((self.day_length - 24) / tcoef)
                + (sst - today["tmin"]) * exp((suns - ti - 24) / tcoef)
            ) / (1 - exp((self.day_length - 24) / tcoef))
        if ti <= hmax:
            # from sunrise to hmax
            amp = (today["tmax"] - today["tmin"]) * (
                1 + (today["tmax"] - today["tmin"]) / tkk
            )
            st = sin(
                pi
                * (ti - self.solar_noon + self.day_length / 2)
                / (self.day_length + 2 * self.site_parameters[8])
            )
            return today["tmin"] - tkk / 2 + sqrt(tkk * tkk + 4 * amp * tkk * st) / 2
        if ti <= suns:
            # from hmax to sunset
            amp = (today["tmax"] - tomorrow["tmin"]) * (
                1 + (today["tmax"] - tomorrow["tmin"]) / tkk
            )
            st = sin(
                pi
                * (ti - self.solar_noon + self.day_length / 2)
                / (self.day_length + 2 * self.site_parameters[8])
            )
            return tomorrow["tmin"] - tkk / 2 + sqrt(tkk * tkk + 4 * amp * tkk * st) / 2
        # from sunset to midnight
        amp = (today["tmax"] - tomorrow["tmin"]) * (
            1 + (today["tmax"] - tomorrow["tmin"]) / tkk
        )
        sts = sin(
            pi * self.day_length / (self.day_length + 2 * self.site_parameters[8])
        )
        sst = tomorrow["tmin"] - tkk / 2 + sqrt(tkk * tkk + 4 * amp * tkk * sts) / 2
        return (
            tomorrow["tmin"]
            - sst * exp((self.day_length - 24) / tcoef)
            + (sst - tomorrow["tmin"]) * exp((suns - ti) / tcoef)
        ) / (1 - exp((self.day_length - 24) / tcoef))

    def calculate_average_temperatures(self):
        self.average_temperature = 0
        self.daytime_temperature = 0
        self.nighttime_temperature = 0
        night_hours = 0
        for hour in self.hours:
            if hour.radiation <= 0:
                night_hours += 1
                self.nighttime_temperature += hour.temperature
            else:
                self.daytime_temperature += hour.temperature
            self.average_temperature += hour.temperature
        if night_hours in (0, 24):
            raise RuntimeError("Plant cotton in polar region?")
        self.average_temperature /= 24
        self.nighttime_temperature /= night_hours
        self.daytime_temperature /= 24 - night_hours

    def compute_evapotranspiration(self, declination, tmpisr):
        """computes the rate of reference evapotranspiration and related variables."""
        stefb: np.float64 = 5.77944e-08  # the Stefan-Boltzman constant, in W m-2 K-4
        c12: np.float64 = 0.125  # c12 ... c15 are constant parameters.
        c13: np.float64 = 0.0439
        c14: np.float64 = 0.030
        c15: np.float64 = 0.0576
        iamhr = 0  # earliest time in day for computing cloud cover
        ipmhr = 0  # latest time in day for computing cloud cover
        cosz: np.float64 = 0  # cosine of sun angle from zenith for this hour
        suna: np.float64 = 0  # sun angle from horizon, degrees at this hour
        # Start hourly loop
        for ihr, hour in enumerate(self.hours):
            ti = ihr + 0.5  # middle of the hourly interval
            # The following subroutines and functions are called for each hour:
            # sunangle, cloudcov, clcor, refalbed .
            cosz, suna = sunangle(
                ti,
                self.latitude,
                declination,
                self.solar_noon,
            )
            isr = tmpisr * cosz  # hourly extraterrestrial radiation in W / m**2
            hour.cloud_cov = cloudcov(hour.radiation, isr, cosz)
            # clcor is called to compute cloud-type correction.
            # iamhr and ipmhr are set.
            hour.cloud_cor = clcor(
                ihr,
                self.site_parameters[7],
                isr,
                cosz,
                self.day_length,
                hour.radiation,
                self.solar_noon,
            )
            if cosz >= 0.1736 and iamhr == 0:
                iamhr = ihr
            if ihr >= 12 and cosz <= 0.1736 and ipmhr == 0:
                ipmhr = ihr - 1
            # refalbed is called to compute the reference albedo for each hour.
            hour.albedo = refalbed(isr, hour.radiation, cosz, suna)
        # Zero some variables that will later be used for summation.
        self.evapotranspiration = 0
        self.net_radiation = 0  # daily net radiation
        for ihr, hour in enumerate(self.hours):
            # Compute saturated vapor pressure (svp), using function VaporPressure().
            # The actual vapor pressure (vp) is computed from svp and the relative
            # humidity. Compute vapor pressure deficit (vpd). This procedure is based
            # on the CIMIS algorithm.
            svp = VaporPressure(hour.temperature)  # saturated vapor pressure, mb
            vp = 0.01 * hour.humidity * svp  # vapor pressure, mb
            vpd = svp - vp  # vapor pressure deficit, mb.
            # Get cloud cover and cloud correction for night hours
            if ihr < iamhr or ihr > ipmhr:
                hour.cloud_cov = 0
                hour.cloud_cor = 0
            # The hourly net radiation is computed using the CIMIS algorithm
            # (Dong et al., 1988):
            # rlonin, the hourly incoming long wave radiation, is computed from ea0,
            # cloud cover (CloudCoverRatio), air temperature (tk),  stefb, and cloud
            # type correction (CloudTypeCorr).
            # rnet, the hourly net radiation, W m-2, is computed from the global
            # radiation, the albedo, the incoming long wave radiation, and the outgoing
            # longwave radiation.
            tk = hour.temperature + 273.161  # hourly air temperature in Kelvin.
            ea0 = clearskyemiss(vp, tk)  # clear sky emissivity for long wave radiation
            # Compute incoming long wave radiation:
            rlonin = (
                ea0 * (1 - hour.cloud_cov) + hour.cloud_cov
            ) * stefb * tk ** 4 - hour.cloud_cor
            rnet = (1 - hour.albedo) * hour.radiation + rlonin - stefb * tk ** 4
            self.net_radiation += rnet
            # The hourly reference evapotranspiration ReferenceETP is computed by the
            # CIMIS algorithm using the modified Penman equation:
            # The weighting ratio (w) is computed from the functions del() (the slope
            # of the saturation vapor pressure versus air temperature) and gam() (the
            # psychometric constant).
            w = delta(tk, svp) / (
                delta(tk, svp) + gamma(self.elevation, hour.temperature)
            )  # coefficient of the Penman equation

            # The wind function (fu2) is computed using different sets of parameters
            # for day-time and night-time. The parameter values are as suggested by
            # CIMIS.
            fu2 = (  # wind function for computing evapotranspiration
                c12 + c13 * hour.wind_speed
                if hour.radiation <= 0
                else c14 + c15 * hour.wind_speed
            )

            # hlathr, the latent heat for evaporation of water (W m-2 per mm at this
            # hour) is computed as a function of temperature.
            hlathr = 878.61 - 0.66915 * (hour.temperature + 273.161)
            # ReferenceETP, the hourly reference evapotranspiration, is now computed by
            # the modified Penman equation.
            hour.ref_et = max(w * rnet / hlathr + (1 - w) * vpd * fu2, 0)
            # ReferenceTransp is the sum of ReferenceETP
            self.evapotranspiration += hour.ref_et
            # es1hour and es2hour are computed as the hourly potential evapo-
            # transpiration due to radiative and aerodynamic factors, respectively.
            # es1hour and ReferenceTransp are not computed for periods of negative net
            # radiation.
            hour.et2 = (1 - w) * vpd * fu2
            hour.et1 = max(w * rnet / hlathr, 0)
