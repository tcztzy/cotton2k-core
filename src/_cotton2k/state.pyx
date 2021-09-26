import datetime

import numpy as np

from _cotton2k.utils import date2doy, doy2date

cdef class StateBase:
    rlat1 = np.zeros(40, dtype=np.float64)  # lateral root length (cm) to the left of the tap root
    rlat2 = np.zeros(40, dtype=np.float64)   # lateral root length (cm) to the right of the tap root
    actual_root_growth = np.zeros((40, 20), dtype=np.float64)
    _root_potential_growth = np.zeros((40, 20), dtype=np.float64)  # potential root growth in a soil cell (g per day).
    root_age = np.zeros((40, 20), dtype=np.float64)
    soil_temperature = np.zeros((40, 20), dtype=np.float64)  # daily average soil temperature, oK.
    hours = np.empty(24, dtype=object)

    @property
    def day_inc(self):
        """physiological days increment for this day. based on hourlytemperatures."""
        # The threshold value is assumed to be 12 C (p1). One physiological day is
        # equivalent to a day with an average temperature of 26 C, and therefore the heat
        # units are divided by 14 (p2).

        # A linear relationship is assumed between temperature and heat unit accumulation
        # in the range of 12 C (p1) to 33 C (p2*p3+p1). the effect of temperatures higher
        # than 33 C is assumed to be equivalent to that of 33 C.

        # The following constant Parameters are used in this function:
        p1 = 12.0  # threshold temperature, C
        p2 = 14.0  # temperature, C, above p1, for one physiological day.
        p3 = 1.5  # maximum value of a physiological day.

        dayfd = 0.0  # the daily contribution to physiological age (return value).
        for hour in self.hours:
            # add the hourly contribution to physiological age.
            dayfd += min(max((hour.temperature - p1) / p2, 0), p3)
        return dayfd / 24.0

    @property
    def date(self):
        return datetime.date.fromordinal(self._ordinal)

    @date.setter
    def date(self, value):
        if not isinstance(value, datetime.date):
            raise TypeError
        self._ordinal = value.toordinal()

    @property
    def solar_noon(self):
        return self._[0].solar_noon

    @solar_noon.setter
    def solar_noon(self, value):
        self._[0].solar_noon = value

    @property
    def day_length(self):
        return self._[0].day_length

    @day_length.setter
    def day_length(self, value):
        self._[0].day_length = value

    @property
    def leaf_area_index(self):
        return self._[0].leaf_area_index

    @leaf_area_index.setter
    def leaf_area_index(self, value):
        self._[0].leaf_area_index = value

    @property
    def leaf_area(self):
        return self._[0].leaf_area

    @leaf_area.setter
    def leaf_area(self, value):
        self._[0].leaf_area = value

    @property
    def leaf_weight(self):
        return self._[0].leaf_weight

    @leaf_weight.setter
    def leaf_weight(self, value):
        self._[0].leaf_weight = value

    @property
    def leaf_weight_area_ratio(self):
        return self._[0].leaf_weight_area_ratio

    @leaf_weight_area_ratio.setter
    def leaf_weight_area_ratio(self, value):
        self._[0].leaf_weight_area_ratio = value

    @property
    def leaf_nitrogen(self):
        return self._[0].leaf_nitrogen

    @leaf_nitrogen.setter
    def leaf_nitrogen(self, value):
        self._[0].leaf_nitrogen = value

    @property
    def leaf_nitrogen_concentration(self):
        return self._[0].leaf_nitrogen_concentration

    @leaf_nitrogen_concentration.setter
    def leaf_nitrogen_concentration(self, value):
        self._[0].leaf_nitrogen_concentration = value

    @property
    def number_of_vegetative_branches(self):
        return self._[0].number_of_vegetative_branches

    @number_of_vegetative_branches.setter
    def number_of_vegetative_branches(self, value):
        self._[0].number_of_vegetative_branches = value

    @property
    def number_of_squares(self):
        return self._[0].number_of_squares

    @number_of_squares.setter
    def number_of_squares(self, value):
        self._[0].number_of_squares = value

    @property
    def number_of_green_bolls(self):
        return self._[0].number_of_green_bolls

    @number_of_green_bolls.setter
    def number_of_green_bolls(self, value):
        self._[0].number_of_green_bolls = value

    @property
    def number_of_open_bolls(self):
        return self._[0].number_of_open_bolls

    @number_of_open_bolls.setter
    def number_of_open_bolls(self, value):
        self._[0].number_of_open_bolls = value

    @property
    def nitrogen_stress(self):
        return self._[0].nitrogen_stress

    @nitrogen_stress.setter
    def nitrogen_stress(self, value):
        self._[0].nitrogen_stress = value

    @property
    def nitrogen_stress_vegetative(self):
        return self._[0].nitrogen_stress_vegetative

    @nitrogen_stress_vegetative.setter
    def nitrogen_stress_vegetative(self, value):
        self._[0].nitrogen_stress_vegetative = value

    @property
    def nitrogen_stress_fruiting(self):
        return self._[0].nitrogen_stress_fruiting

    @nitrogen_stress_fruiting.setter
    def nitrogen_stress_fruiting(self, value):
        self._[0].nitrogen_stress_fruiting = value

    @property
    def nitrogen_stress_root(self):
        return self._[0].nitrogen_stress_root

    @nitrogen_stress_root.setter
    def nitrogen_stress_root(self, value):
        self._[0].nitrogen_stress_root = value

    @property
    def total_required_nitrogen(self):
        return self._[0].total_required_nitrogen

    @total_required_nitrogen.setter
    def total_required_nitrogen(self, value):
        self._[0].total_required_nitrogen = value

    @property
    def petiole_nitrogen_concentration(self):
        return self._[0].petiole_nitrogen_concentration

    @petiole_nitrogen_concentration.setter
    def petiole_nitrogen_concentration(self, value):
        self._[0].petiole_nitrogen_concentration = value

    @property
    def seed_nitrogen(self):
        return self._[0].seed_nitrogen

    @seed_nitrogen.setter
    def seed_nitrogen(self, value):
        self._[0].seed_nitrogen = value

    @property
    def seed_nitrogen_concentration(self):
        return self._[0].seed_nitrogen_concentration

    @seed_nitrogen_concentration.setter
    def seed_nitrogen_concentration(self, value):
        self._[0].seed_nitrogen_concentration = value

    @property
    def burr_nitrogen(self):
        return self._[0].burr_nitrogen

    @burr_nitrogen.setter
    def burr_nitrogen(self, value):
        self._[0].burr_nitrogen = value

    @property
    def burr_nitrogen_concentration(self):
        return self._[0].burr_nitrogen_concentration

    @burr_nitrogen_concentration.setter
    def burr_nitrogen_concentration(self, value):
        self._[0].burr_nitrogen_concentration = value

    @property
    def root_nitrogen_concentration(self):
        return self._[0].root_nitrogen_concentration

    @root_nitrogen_concentration.setter
    def root_nitrogen_concentration(self, value):
        self._[0].root_nitrogen_concentration = value

    @property
    def root_nitrogen(self):
        return self._[0].root_nitrogen

    @root_nitrogen.setter
    def root_nitrogen(self, value):
        self._[0].root_nitrogen = value

    @property
    def square_nitrogen(self):
        return self._[0].square_nitrogen

    @square_nitrogen.setter
    def square_nitrogen(self, value):
        self._[0].square_nitrogen = value

    @property
    def square_nitrogen_concentration(self):
        return self._[0].square_nitrogen_concentration

    @square_nitrogen_concentration.setter
    def square_nitrogen_concentration(self, value):
        self._[0].square_nitrogen_concentration = value

    @property
    def stem_nitrogen_concentration(self):
        return self._[0].stem_nitrogen_concentration

    @stem_nitrogen_concentration.setter
    def stem_nitrogen_concentration(self, value):
        self._[0].stem_nitrogen_concentration = value

    @property
    def stem_nitrogen(self):
        return self._[0].stem_nitrogen

    @stem_nitrogen.setter
    def stem_nitrogen(self, value):
        self._[0].stem_nitrogen = value

    @property
    def fruit_growth_ratio(self):
        return self._[0].fruit_growth_ratio

    @fruit_growth_ratio.setter
    def fruit_growth_ratio(self, value):
        self._[0].fruit_growth_ratio = value

    @property
    def ginning_percent(self):
        return self._[0].ginning_percent

    @ginning_percent.setter
    def ginning_percent(self, value):
        self._[0].ginning_percent = value

    @property
    def number_of_pre_fruiting_nodes(self):
        return self._[0].number_of_pre_fruiting_nodes

    @number_of_pre_fruiting_nodes.setter
    def number_of_pre_fruiting_nodes(self, value):
        self._[0].number_of_pre_fruiting_nodes = value

    @property
    def age_of_pre_fruiting_nodes(self):
        return self._[0].age_of_pre_fruiting_nodes

    @property
    def leaf_area_pre_fruiting(self):
        return self._[0].leaf_area_pre_fruiting

    @property
    def leaf_weight_pre_fruiting(self):
        return self._[0].leaf_weight_pre_fruiting

    @property
    def actual_transpiration(self):
        return self._[0].actual_transpiration

    @actual_transpiration.setter
    def actual_transpiration(self, value):
        self._[0].actual_transpiration = value

    @property
    def actual_soil_evaporation(self):
        return self._[0].actual_soil_evaporation

    @actual_soil_evaporation.setter
    def actual_soil_evaporation(self, value):
        self._[0].actual_soil_evaporation = value

    @property
    def potential_evaporation(self):
        return self._[0].potential_evaporation

    @potential_evaporation.setter
    def potential_evaporation(self, value):
        self._[0].potential_evaporation = value

    @property
    def total_actual_leaf_growth(self):
        return self._[0].total_actual_leaf_growth

    @total_actual_leaf_growth.setter
    def total_actual_leaf_growth(self, value):
        self._[0].total_actual_leaf_growth = value

    @property
    def total_actual_petiole_growth(self):
        return self._[0].total_actual_petiole_growth

    @total_actual_petiole_growth.setter
    def total_actual_petiole_growth(self, value):
        self._[0].total_actual_petiole_growth = value

    @property
    def actual_square_growth(self):
        return self._[0].actual_square_growth

    @actual_square_growth.setter
    def actual_square_growth(self, value):
        self._[0].actual_square_growth = value

    @property
    def actual_stem_growth(self):
        return self._[0].actual_stem_growth

    @actual_stem_growth.setter
    def actual_stem_growth(self, value):
        self._[0].actual_stem_growth = value

    @property
    def actual_boll_growth(self):
        return self._[0].actual_boll_growth

    @actual_boll_growth.setter
    def actual_boll_growth(self, value):
        self._[0].actual_boll_growth = value

    @property
    def actual_burr_growth(self):
        return self._[0].actual_burr_growth

    @actual_burr_growth.setter
    def actual_burr_growth(self, value):
        self._[0].actual_burr_growth = value

    @property
    def supplied_nitrate_nitrogen(self):
        return self._[0].supplied_nitrate_nitrogen

    @supplied_nitrate_nitrogen.setter
    def supplied_nitrate_nitrogen(self, value):
        self._[0].supplied_nitrate_nitrogen = value

    @property
    def supplied_ammonium_nitrogen(self):
        return self._[0].supplied_ammonium_nitrogen

    @supplied_ammonium_nitrogen.setter
    def supplied_ammonium_nitrogen(self, value):
        self._[0].supplied_ammonium_nitrogen = value

    @property
    def deep_soil_temperature(self):
        return self._[0].deep_soil_temperature

    @deep_soil_temperature.setter
    def deep_soil_temperature(self, value):
        self._[0].deep_soil_temperature = value

    @property
    def petiole_nitrogen(self):
        return self._[0].petiole_nitrogen

    @petiole_nitrogen.setter
    def petiole_nitrogen(self, value):
        self._[0].petiole_nitrogen = value

    @property
    def petiole_nitrate_nitrogen_concentration(self):
        return self._[0].petiole_nitrate_nitrogen_concentration

    @petiole_nitrate_nitrogen_concentration.setter
    def petiole_nitrate_nitrogen_concentration(self, value):
        self._[0].petiole_nitrate_nitrogen_concentration = value

    @property
    def pollination_switch(self):
        return self._[0].pollination_switch

    @pollination_switch.setter
    def pollination_switch(self, value):
        self._[0].pollination_switch = value

    @property
    def phenological_delay_by_nitrogen_stress(self):
        """the delay caused by nitrogen stress, is assumed to be a function of the vegetative nitrogen stress."""
        return min(max(0.65 * (1 - self.nitrogen_stress_vegetative), 0), 1)

    def __getitem__(self, item):
        return getattr(self, item)
