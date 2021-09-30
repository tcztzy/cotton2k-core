# pylint: disable=no-name-in-module
from math import exp

import numpy as np

# parameters used to correct photosynthesis for ambient CO2 concentration.
CO2_PARAMETERS = (
    1.0235,
    1.0264,
    1.0285,
    1.0321,
    1.0335,
    1.0353,
    1.0385,
    1.0403,
    1.0431,
    1.0485,
    1.0538,
    1.0595,
    1.0627,
    1.0663,
    1.0716,
    1.0752,
    1.0784,
    1.0823,
    1.0880,
    1.0923,
    1.0968,
    1.1019,
    1.1087,
    1.1172,
    1.1208,
    1.1243,
    1.1311,
    1.1379,
    1.1435,
    1.1490,
    1.1545,
    1.1601,
    1.1656,
    1.1712,
    1.1767,
    1.1823,
    1.1878,
    1.1934,
    1.1990,
    1.2045,
    1.2101,
    1.2156,
    1.2212,
    1.2267,
    1.2323,
)

START_YEAR = 1960
STOP_YEAR = START_YEAR + len(CO2_PARAMETERS) - 1


def ambient_co2_factor(year):
    """
    Examples
    --------
    >>> ambient_co2_factor(1959)
    1
    >>> ambient_co2_factor(2019)
    1.30526
    >>> ambient_co2_factor(2020)
    1.310124
    >>> ambient_co2_factor(2021)
    1.314988
    """
    if year < START_YEAR:
        return 1
    if year <= STOP_YEAR:
        return CO2_PARAMETERS[year - START_YEAR]
    return CO2_PARAMETERS[-1] + 0.004864 * (year - STOP_YEAR)


# pylint: disable=no-member, too-few-public-methods
class Photosynthesis:
    @property
    def carbon_dioxide_correction_factor(self):
        """Get the CO2 correction factor for photosynthesis, using ambient_co2_factor
        and a factor that may be variety specific."""
        return 1.3 * ambient_co2_factor(self.date.year)

    @property
    def nitrogen_correction_factor(self):
        """Compute the effect of leaf N concentration on photosynthesis, using an
        empirical relationship."""
        vpnet = [0.034, 0.010, 0.32]
        ptnfac = vpnet[2] + (self.leaf_nitrogen_concentration - vpnet[1]) * (
            1 - vpnet[2]
        ) / (
            vpnet[0] - vpnet[1]
        )  # correction factor for low nitrogen content in leaves.
        return max(min(ptnfac, 1), vpnet[2])

    @property
    def photorespiration_ratio(self):
        """Compute the photorespiration factor as a linear function of average day time
        temperature."""
        return np.polynomial.Polynomial([0.0032125, 0.0066875])(
            self.daytime_temperature
        )  # photorespiration factor

    # pylint: disable=too-many-arguments
    def column_shading(
        self,
        row_space,
        plant_row_column,
        column_width,
        max_leaf_area_index,
        relative_radiation_received_by_a_soil_column,
    ):
        zint = 1.0756 * self.plant_height / row_space
        for k in range(20):
            sw = (k + 1) * column_width
            if k <= plant_row_column:
                k0 = plant_row_column - k
                sw1 = sw - column_width / 2
            else:
                sw1 = sw - column_width / 2 - (plant_row_column + 1) * column_width
                k0 = k
            shade = 0
            if sw1 < self.plant_height:
                shade = 1 - (sw1 / self.plant_height) ** 2
                if (
                    self.light_interception < zint
                    and self.leaf_area_index < max_leaf_area_index
                ):
                    shade *= self.light_interception / zint
            relative_radiation_received_by_a_soil_column[k0] = max(0.05, 1 - shade)

    def compute_light_interception(
        self,
        max_leaf_area_index: float,
        row_space: float,
    ):
        if self.version < 0x0500:  # type: ignore[attr-defined]
            zint = 1.0756 * self.plant_height / row_space  # type: ignore[attr-defined]
            lfint = (
                0.80 * self.leaf_area_index  # type: ignore[attr-defined]
                if self.leaf_area_index <= 0.5  # type: ignore[attr-defined]
                else 1 - exp(0.07 - 1.16 * self.leaf_area_index)  # type: ignore
            )
            if lfint > zint:
                light_interception = (zint + lfint) / 2
            elif self.leaf_area_index < max_leaf_area_index:  # type: ignore
                light_interception = lfint
            else:
                light_interception = zint
            return light_interception if light_interception < 1 else 1
        param = max(1.16, -0.1 * self.plant_height + 8)  # type: ignore[attr-defined]
        return 1 - exp(-param * self.leaf_area_index)  # type: ignore[attr-defined]

    def maintenance_respiration(self, old_stem_weight):
        """maintenance respiration, g per plant per day."""
        rsubo = 0.0032  # maintenance respiration factor.
        # Old stems are those more than voldstm = 32 calendar days old.
        # Maintenance respiration is computed on the basis of plant dry weight, minus
        # the old stems and the dry tissue of opened bolls.
        return (self.maintenance_weight - old_stem_weight) * rsubo

    # pylint: disable=too-many-locals
    def get_net_photosynthesis(
        self, radiation, per_plant_area, ptsred, old_stem_weight
    ):
        """
        References:

        Baker et. al. (1972). Simulation of Growth and Yield in Cotton: I. Gross
        photosynthesis, respiration and growth. Crop Sci. 12:431-435.

        Harper et. al. (1973) Carbon dioxide and the photosynthesis of field crops. A
        metered carbon dioxide release in cotton under field conditions. Agron. J.
        65:7-11.

        Baker (1965)  Effects of certain environmental factors on net assimilation in
        cotton. Crop Sci. 5:53-56 (Fig 5).
        """
        # constants
        gsubr = 0.375  # the growth respiration factor.
        # Note: co2parm is for icrease in ambient CO2 concentration changes from 1959
        # (308 ppm).
        # The first 28 values (up to 1987) are from GOSSYM. The other values (up to
        # 2004) are derived from data of the Carbon Dioxide Information Analysis Center
        # (CDIAC).

        # Exit the function and end simulation if there are no leaves
        if self.leaf_area_index <= 0:
            raise RuntimeError

        # Convert the average daily short wave radiation from langley per day, to Watts
        # per square meter (wattsm).
        wattsm = (
            radiation * 697.45 / (self.day_length * 60)
        )  # average daily global radiation, W m^{-2}
        # Compute pstand as an empirical function of wattsm (Baker et al., 1972)

        # gross photosynthesis for a non-stressed full canopy
        pstand = np.polynomial.Polynomial([2.3908, 1.37379, -0.00054136])(wattsm)

        # Convert it to gross photosynthesis per plant (pplant), using per_plant_area
        # and corrections for light interception by canopy, ambient CO2 concentration,
        # water stress and low N in the leaves.

        # actual gross photosynthetic rate, g per plant per day.
        pplant = (
            0.001
            * pstand
            * self.light_interception
            * per_plant_area
            * ptsred
            * self.carbon_dioxide_correction_factor
            * self.nitrogen_correction_factor
        )

        # Net photosynthesis is computed by subtracting photo-respiration and
        # maintenance respiration from the gross rate of photosynthesis.

        # To avoid computational problem, make sure that pts is positive and non-zero.
        pts = max(
            (1 - self.photorespiration_ratio) * pplant
            - self.maintenance_respiration(old_stem_weight),
            0.00001,
        )  # intermediate computation of net_photosynthesis.
        # the growth respiration (gsubr) supplies energy for converting the supplied
        # carbohydrates to plant tissue dry matter.

        # 0.68182 converts CO2 to CH2O. net_photosynthesis is the computed net
        # photosynthesis, in g per plant per day.
        self.net_photosynthesis = pts / (1 + gsubr) * 0.68182  # pylint: disable=W0201
