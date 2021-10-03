from collections import defaultdict
from datetime import timedelta
from math import acos, cos, pi, sin, sqrt
from typing import TYPE_CHECKING

from .climate import compute_day_length, compute_hourly_wind_speed, dayrh, radiation

if TYPE_CHECKING:
    from typing import DefaultDict

METEOROLOGY: "DefaultDict" = defaultdict(lambda: defaultdict(dict))


class Meteorology:  # pylint: disable=E1101,R0903,R0914,W0201
    def daily_meteorology(self):
        u = (self.date - self.start_date).days
        declination: float  # daily declination angle, in radians
        sunr: float  # time of sunrise, hours.
        suns: float  # time of sunset, hours.
        tmpisr: float  # extraterrestrial radiation, \frac{W}{m^2}
        hour = timedelta(hours=1)
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
            radsum = self.climate[u]["Rad"] * 11.630287 / dsbe
        rainToday = self.climate[u]["Rain"]  # the amount of rain today, mm
        # Set 'pollination switch' for rainy days (as in GOSSYM).
        self.pollination_switch = rainToday < 2.5
        # Call SimulateRunoff() only if the daily rainfall is more than 2 mm.
        # NOTE: this is modified from the original GOSSYM - RRUNOFF routine. It is
        # called here for rainfall only, but it is not activated when irrigation is
        # applied.
        runoffToday = 0  # amount of runoff today, mm
        if rainToday >= 2.0:
            runoffToday = self.simulate_runoff(u)
            if runoffToday < rainToday:
                rainToday -= runoffToday
            else:
                rainToday = 0
            self.climate[u]["Rain"] = rainToday
        self.runoff = runoffToday
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
            hour.temperature = self.daytmp(u, ti, self.site_parameters[8], sunr, suns)
            hour.dew_point = self.tdewhour(
                u,
                ti,
                hour.temperature,
                sunr,
                self.solar_noon,
                self.site_parameters[8],
                self.site_parameters[12],
                self.site_parameters[13],
                self.site_parameters[14],
            )
            hour.humidity = dayrh(hour.temperature, hour.dew_point)
            hour.wind_speed = compute_hourly_wind_speed(
                ti, self.climate[u]["Wind"] * 1000 / 86400, t1, t2, t3, wnytf
            )
        # Compute average daily temperature, using function AverageAirTemperatures.
        self.calculate_average_temperatures()
        # Compute potential evapotranspiration.
        self.compute_evapotranspiration(
            self.latitude, self.elevation, declination, tmpisr, self.site_parameters[7]
        )
