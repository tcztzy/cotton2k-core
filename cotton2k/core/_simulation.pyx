# cython: language_level=3
# distutils: language = c++
from enum import Enum, auto
from datetime import date, timedelta
from math import sin, cos, acos, sqrt, pi, atan
from pathlib import Path

from libc.math cimport exp, log
from libc.stdlib cimport malloc
from libc.stdint cimport uint32_t
from libcpp cimport bool as bool_t

cimport numpy
import numpy as np
from scipy.interpolate import interp2d

from .meteorology import tdewest
from .fruit import TemperatureOnFruitGrowthRate
from .phenology import Stage
from .leaf import temperature_on_leaf_growth_rate, leaf_resistance_for_transpiration
from .soil import root_psi, SoilTemOnRootGrowth, SoilAirOnRootGrowth, SoilNitrateOnRootGrowth, PsiOnTranspiration, SoilTemperatureEffect, SoilWaterEffect, wcond, qpsi, psiq, SoilMechanicResistance, PsiOsmotic, form
from .utils import date2doy, doy2date
from .thermology import canopy_balance


ctypedef struct cState:
    double leaf_weight_area_ratio  # temperature dependent factor for converting leaf area to leaf weight during the day, g dm-1
    double petiole_nitrogen_concentration  # average nitrogen concentration in petioles.
    double seed_nitrogen_concentration  # average nitrogen concentration in seeds.
    double seed_nitrogen  # total seed nitrogen, g per plant.
    double root_nitrogen_concentration  # average nitrogen concentration in roots.
    double root_nitrogen  # total root nitrogen, g per plant.
    double square_nitrogen_concentration  # average concentration of nitrogen in the squares.
    double burr_nitrogen_concentration  # average nitrogen concentration in burrs.
    double burr_nitrogen  # nitrogen in burrs, g per plant.
    double square_nitrogen  # total nitrogen in the squares, g per plant
    double stem_nitrogen_concentration  # ratio of stem nitrogen to dry matter.
    double stem_nitrogen  # total stem nitrogen, g per plant
    double fruit_growth_ratio  # ratio between actual and potential square and boll growth.
    double deep_soil_temperature  # boundary soil temperature of deepest layer (K)
    double total_actual_leaf_growth  # actual growth rate of all the leaves, g plant-1 day-1.
    double total_actual_petiole_growth  # actual growth rate of all the petioles, g plant-1 day-1.
    double actual_burr_growth  # total actual growth of burrs in bolls, g plant-1 day-1.
    double supplied_nitrate_nitrogen  # uptake of nitrate by the plant from the soil, mg N per slab per day.
    double supplied_ammonium_nitrogen  # uptake of ammonia N by the plant from the soil, mg N per slab per day.
    double petiole_nitrogen  # total petiole nitrogen, g per plant.
    double petiole_nitrate_nitrogen_concentration  # average nitrate nitrogen concentration in petioles.
    int number_of_pre_fruiting_nodes  # number of prefruiting nodes, per plant.
    double delay_for_new_branch[3]

ctypedef struct NitrogenFertilizer:  # nitrogen fertilizer application information for each day.
    int day  # date of application (DOY)
    int mthfrt  # method of application ( 0 = broadcast; 1 = sidedress; 2 = foliar; 3 = drip fertigation);
    int ksdr  # horizontal placement of side-dressed fertilizer, cm.
    int lsdr  # vertical placement of side-dressed fertilizer, cm.
    double amtamm  # ammonium N applied, kg N per ha;
    double amtnit  # nitrate N applied, kg N per ha;
    double amtura  # urea N applied, kg N per ha;
cdef int maxl = 40
cdef int maxk = 20
cdef int nl
cdef int nk
cdef double PotGroAllSquares  # sum of potential growth rates of all squares, g plant-1 day-1.
cdef double PotGroAllBolls  # sum of potential growth rates of seedcotton in all bolls, g plant-1 day-1.
cdef double PotGroAllBurrs  # sum of potential growth rates of burrs in all bolls, g plant-1 day-1.
cdef NitrogenFertilizer NFertilizer[150]
cdef int NumNitApps  # number of applications of nitrogen fertilizer.
cdef double ElCondSatSoilToday  # electrical conductivity of saturated extract (mmho/cm) on this day.
cdef double HumusNitrogen[40][20]  # N in stable humic fraction material in a soil cells, mg/cm3.

cdef int LateralRootFlag[40] # flags indicating presence of lateral roots in soil layers: 0 = no lateral roots are possible. 1 = lateral roots may be initiated. 2 = lateral roots have been initiated.

AbscissionLag = np.zeros(20)  # the time (in physiological days) from tagging fruiting sites for shedding.
ShedByWaterStress = np.zeros(20)  # the effect of moisture stress on shedding.
ShedByNitrogenStress = np.zeros(20)  # the effect of nitrogen stress on shedding.
ShedByCarbonStress = np.zeros(20)  # the effect of carbohydrate stress on shedding
NumSheddingTags = 0  # number of 'box-car' units used for moving values in arrays defining fruit shedding (AbscissionLag, ShedByCarbonStress, ShedByNitrogenStress and ShedByWaterStress).

# defined by input for consecutive 15 cm soil layers.
cdef double LayerDepth = 15
cdef double AverageLwp = 0  # running average of state.min_leaf_water_potential + state.max_leaf_water_potential for the last 3 days.
PercentDefoliation = 0

LwpMinX = np.zeros(3, dtype=np.double)  # array of values of min_leaf_water_potential for the last 3 days.
LwpX = np.zeros(3, dtype=np.double)  # array of values of min_leaf_water_potential + max_leaf_water_potential for the last 3 days.

DefoliationDate = np.zeros(5, dtype=np.int_)  # Dates (DOY) of defoliant applications.
DefoliationMethod = np.zeros(5, dtype=np.int_)  # code number of method of application of defoliants:  0 = 'banded'; 1 = 'sprinkler'; 2 = 'broaddcast'.
DefoliantAppRate = np.zeros(5, dtype=np.double)  # rate of defoliant application in pints per acre.


PetioleWeightPreFru = np.zeros(9, dtype=np.double)  # weight of prefruiting node petioles, g.
PotGroLeafAreaPreFru = np.zeros(9, dtype=np.double)  # potentially added area of a prefruiting node leaf, dm2 day-1.
PotGroLeafWeightPreFru = np.zeros(9, dtype=np.double)  # potentially added weight of a prefruiting node leaf, g day-1.
PotGroPetioleWeightPreFru = np.zeros(9, dtype=np.double)  # potentially added weight of a prefruiting node petiole, g day-1.

FreshOrganicNitrogen = np.zeros((40, 20), dtype=np.double)  # N in fresh organic matter in a soil cell, mg cm-3.


cdef class Hour:
    cdef public double albedo  # hourly albedo of a reference crop.
    cdef public double cloud_cor  # hourly cloud type correction.
    cdef public double cloud_cov  # cloud cover ratio (0 to 1).
    cdef public double dew_point  # hourly dew point temperatures, C.
    cdef public double et1  # part of hourly Penman evapotranspiration affected by net radiation, in mm per hour.
    cdef public double et2  # part of hourly Penman evapotranspiration affected by wind and vapor pressure deficit, in mm per hour.
    cdef public double humidity  # hourly values of relative humidity (%).
    cdef public double radiation  # hourly global radiation, W / m2.
    cdef public double ref_et  # reference evapotranspiration, mm per hour.
    cdef public double temperature  # hourly air temperatures, C.
    cdef public double wind_speed  # Hourly wind velocity, m per second.

cdef double[3] cgind = [1, 1, 0.10]  # the index for the capability of growth of class I roots (0 to 1).


cdef void NitrogenFlow(int nn, double q01[], double q1[], double dd[], double nit[], double nur[]):
    """Computes the movement of nitrate and urea between the soil cells, within a soil column or within a soil layer, as a result of water flux.

    It is assumed that there is only a passive movement of nitrate and urea (i.e., with the movement of water).

    Arguments
    ---------
    dd
        one dimensional array of layer or column widths.
    nit
        one dimensional array of a layer or a column of VolNo3NContent.
    nn
        the number of cells in this layer or column.
    nur
        one dimensional array of a layer or a column of soil_urea_content.
    q01
        one dimensional array of a layer or a column of the previous values of cell.water_content.
    q1
        one dimensional array of a layer or a column of cell.water_content."""
    # Zeroise very small values to prevent underflow.
    for i in range(nn):
        if nur[i] < 1e-20:
            nur[i] = 0
        if nit[i] < 1e-20:
            nit[i] = 0
    # Declare and zeroise arrays.
    qdn = np.zeros(40, dtype=np.double)  # amount of nitrate N moving to the previous cell.
    qup = np.zeros(40, dtype=np.double)  # amount of nitrate N moving to the following cell.
    udn = np.zeros(40, dtype=np.double)  # amount of urea N moving to the previous cell.
    uup = np.zeros(40, dtype=np.double)  # amount of urea N moving to the following cell.
    for i in range(nn):
        # The amout of water in each soil cell before (aq0) and after (aq1) water movement is computed from the previous values of water content (q01), the present values (q1), and layer thickness. The associated transfer of soluble nitrate N (qup and qdn) and urea N (uup and udn) is now computed. qup and uup are upward movement (from cell i+1 to i), qdn and udn are downward movement (from cell i-1 to i).
        aq0 = q01[i] * dd[i]  # previous amount of water in cell i
        aq1 = q1[i] * dd[i]  # amount of water in cell i now
        if i == 0:
            qup[i] = 0
            uup[i] = 0
        else:
            qup[i] = -qdn[i - 1]
            uup[i] = -udn[i - 1]

        if i == nn - 1:
            qdn[i] = 0
            udn[i] = 0
        else:
            qdn[i] = min(max((aq1 - aq0) * nit[i + 1] / q01[i + 1], -0.2 * nit[i] * dd[i]), 0.2 * nit[i + 1] * dd[i + 1])
            udn[i] = min(max((aq1 - aq0) * nur[i + 1] / q01[i + 1], -0.2 * nur[i] * dd[i]), 0.2 * nur[i + 1] * dd[i + 1])
    # Loop over all cells to update nit and nur arrays.
    for i in range(nn):
        nit[i] += (qdn[i] + qup[i]) / dd[i]
        nur[i] += (udn[i] + uup[i]) / dd[i]


cdef class State:
    cdef cState _
    cdef Simulation _sim
    cdef numpy.ndarray root_impedance  # root mechanical impedance for a soil cell, kg cm-2.
    cdef unsigned int _ordinal
    cdef public numpy.ndarray foliage_temperature  # average foliage temperature (oK).
    cdef public numpy.ndarray soil_temperature  # daily average soil temperature, oK.
    cdef public numpy.ndarray hourly_soil_temperature  # hourly soil temperature oK.
    cdef public numpy.ndarray root_growth_factor  # root growth correction factor in a soil cell (0 to 1).
    cdef public numpy.ndarray root_weights
    cdef public numpy.ndarray root_weight_capable_uptake  # root weight capable of uptake, in g per soil cell.
    cdef public numpy.ndarray burr_weight  # weight of burrs for each site, g per plant.
    cdef public numpy.ndarray burr_potential_growth  # potential growth rate of burrs in an individual boll, g day-1.
    cdef public numpy.ndarray node_petiole_weight  # petiole weight at each fruiting site, g.
    cdef public numpy.ndarray node_petiole_potential_growth  # potential growth in weight of an individual fruiting node petiole, g day-1.
    cdef public numpy.ndarray main_stem_leaf_area
    cdef public numpy.ndarray main_stem_leaf_area_potential_growth  # potential growth in area of an individual main stem node leaf, dm2 day-1.
    cdef public numpy.ndarray main_stem_leaf_weight  # mainstem leaf weight at each node, g.
    cdef public numpy.ndarray main_stem_leaf_potential_growth  # potential growth in weight of an individual main stem node leaf, g day-1.
    cdef public numpy.ndarray main_stem_leaf_petiole_potential_growth  # potential growth in weight of an individual main stem node petiole, g day-1.
    cdef public numpy.ndarray main_stem_leaf_petiole_weight  # weight of mainstem leaf petiole at each node, g.
    cdef public numpy.ndarray square_weights  # weight of each square, g per plant.
    cdef public numpy.ndarray square_potential_growth  # potential growth in weight of an individual fruiting node squares, g day-1.
    cdef public numpy.ndarray node_delay  # cumulative effect of stresses on delaying the formation of a new node on a fruiting branch.
    cdef public numpy.ndarray node_leaf_age  # leaf age at each fruiting site, physiological days.
    cdef public numpy.ndarray node_leaf_area  # leaf area at each fruiting site, dm2.
    cdef public numpy.ndarray node_leaf_weight  # leaf weight at each fruiting site, g.
    cdef public numpy.ndarray node_leaf_area_potential_growth  # potential growth in area of an individual fruiting node leaf, dm2 day-1.
    cdef public numpy.ndarray fruiting_nodes_age  # age of each fruiting site, physiological days from square initiation.
    cdef public numpy.ndarray fruiting_nodes_average_temperature  # running average temperature of each node.
    cdef public numpy.ndarray fruiting_nodes_boll_age  # age of each boll, physiological days from flowering.
    cdef public numpy.ndarray fruiting_nodes_boll_potential_growth  # potential growth in weight of an individual fruiting node bolls, g day-1.
    cdef public numpy.ndarray fruiting_nodes_boll_weight  # weight of seedcotton for each site, g per plant.
    cdef public numpy.ndarray fruiting_nodes_fraction  # fraction of fruit remaining at each fruiting site (0 to 1).
    cdef public numpy.ndarray fruiting_nodes_stage
    cdef public numpy.ndarray fruiting_nodes_ginning_percent
    cdef public numpy.ndarray soil_water_content  # volumetric water content of a soil cell, cm3 cm-3.
    cdef public numpy.ndarray soil_urea_content  # volumetric urea nitrogen content of a soil cell, mg N cm-3.
    cdef public numpy.ndarray soil_fresh_organic_matter  # fresh organic matter in the soil, mg / cm3.
    cdef public numpy.ndarray soil_nitrate_content  # volumetric nitrate nitrogen content of a soil cell, mg N cm-3.
    cdef public numpy.ndarray soil_ammonium_content  # volumetric ammonium nitrogen content of a soil cell, mg N cm-3.
    cdef public numpy.ndarray soil_humus_organic_matter  # humus fraction of soil organic matter, mg/cm3.
    cdef public numpy.ndarray soil_psi  # matric water potential of a soil cell, bars.
    cdef public object date
    cdef public unsigned int seed_layer_number  # layer number where the seeds are located.
    cdef public unsigned int taproot_layer_number  # last soil layer with taproot.
    cdef public unsigned int year
    cdef public unsigned int version
    cdef public unsigned int kday
    cdef public unsigned int drip_x  # number of column in which the drip emitter is located
    cdef public unsigned int drip_y  # number of layer in which the drip emitter is located.
    cdef public double actual_boll_growth  # total actual growth of seedcotton in bolls, g plant-1 day-1.
    cdef public double actual_soil_evaporation  # actual evaporation from soil surface, mm day-1.
    cdef public double actual_stem_growth  # actual growth rate of stems, g plant-1 day-1.
    cdef public double actual_transpiration  # actual transpiration from plants, mm day-1.
    cdef public double average_soil_psi  # average soil matric water potential, bars, computed as the weighted average of the root zone.
    cdef public double day_length  # day length, in hours
    cdef public double pre_fruiting_nodes_age[9]  # age of each prefruiting node, physiological days.
    cdef public double pre_fruiting_leaf_area[9]  # area of prefruiting node leaves, dm2.
    cdef public double average_min_leaf_water_potential  #running average of min_leaf_water_potential for the last 3 days.
    cdef public double average_temperature  # average daily temperature, C, for 24 hours.
    cdef public double carbon_allocated_for_root_growth  # available carbon allocated for root growth, g per plant.
    cdef public double carbon_stress  # carbohydrate stress factor.
    cdef public double daytime_temperature  # average day-time temperature, C.
    cdef public double delay_of_emergence  # effect of negative values of xt on germination rate.
    cdef public double delay_of_new_fruiting_branch[3]  # cumulative effect of stresses on delaying the formation of a new fruiting branch.
    cdef public double evapotranspiration  # daily sum of hourly reference evapotranspiration, mm per day.
    cdef public double extra_carbon  # Extra carbon, not used for plant potential growth requirements, assumed to accumulate in taproot.
    cdef public double fiber_length
    cdef public double fiber_strength
    cdef public double ginning_percent  # weighted average ginning percentage of all open bolls.
    cdef public double green_bolls_burr_weight  # total weight of burrs in green bolls, g plant-1.
    cdef public double hypocotyl_length  # length of hypocotyl, cm.
    cdef public double leaf_area_index
    cdef public double leaf_nitrogen
    cdef public double leaf_potential_growth  # sum of potential growth rates of all leaves, g plant-1 day-1.
    cdef public double leaf_weight
    cdef public double leaf_weight_pre_fruiting[9]  # weight of prefruiting node leaves, g.
    cdef public double light_interception  # ratio of light interception by plant canopy.
    cdef public double max_leaf_water_potential  # maximum (dawn) leaf water potential, MPa.
    cdef public double min_leaf_water_potential  # minimum (noon) leaf water potential, MPa.
    cdef public double net_photosynthesis  # net photosynthetic rate, g per plant per day.
    cdef public double net_radiation  # daily total net radiation, W m-2.
    cdef public double nighttime_temperature  # average night-time temperature, C.
    cdef public double nitrogen_stress  # the average nitrogen stress coefficient for vegetative and reproductive organs
    cdef public double nitrogen_stress_vegetative  # nitrogen stress limiting vegetative development.
    cdef public double nitrogen_stress_fruiting  # nitrogen stress limiting fruit development.
    cdef public double nitrogen_stress_root  # nitrogen stress limiting root development.
    cdef public double number_of_green_bolls  # average number of retained green bolls, per plant.
    cdef public double open_bolls_burr_weight
    cdef public double pavail  # residual available carbon for root growth from previous day.
    cdef public double petiole_potential_growth  # sum of potential growth rates of all petioles, g plant-1 day-1.
    cdef public double petiole_weight  # total petiole weight, g per plant.
    cdef public double plant_height
    cdef public double reserve_carbohydrate  # reserve carbohydrates in leaves, g per plant.
    cdef public double root_potential_growth  # potential growth rate of roots, g plant-1 day-1
    cdef public double seed_moisture  # moisture content of germinating seeds, percent.
    cdef public double stem_potential_growth  # potential growth rate of stems, g plant-1 day-1.
    cdef public double stem_weight  # total stem weight, g per plant.
    cdef public double taproot_length  # the length of the taproot, in cm.
    cdef public double total_required_nitrogen  # total nitrogen required for plant growth, g per plant.
    cdef public double water_stress  # general water stress index (0 to 1).
    cdef public double water_stress_stem  # water stress index for stem growth (0 to 1).
    cdef public double last_layer_with_root_depth  # the depth to the end of the last layer with roots (cm).
    rlat1 = np.zeros(40, dtype=np.float64)  # lateral root length (cm) to the left of the tap root
    rlat2 = np.zeros(40, dtype=np.float64)   # lateral root length (cm) to the right of the tap root
    actual_root_growth = np.zeros((40, 20), dtype=np.float64)
    _root_potential_growth = np.zeros((40, 20), dtype=np.float64)  # potential root growth in a soil cell (g per day).
    root_age = np.zeros((40, 20), dtype=np.float64)
    hours = np.empty(24, dtype=object)

    def __init__(self, sim, version):
        self._sim = sim
        self.version = version
        for i in range(24):
            self.hours[i] = Hour()

    def __getattr__(self, name):
        try:
            return getattr(self._sim, name)
        except AttributeError:
            return getattr(self, name)

    def __getitem__(self, item):
        return getattr(self, item)

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
    def leaf_area(self):
        # It is assumed that cotyledons fall off at time
        # of first square.
        area = 0
        if self._sim.first_square_date is None:
            cotylwt = 0.20  # weight of cotyledons dry matter.
            area = 0.6 * cotylwt
        for j in range(self.number_of_pre_fruiting_nodes):
            area += self.pre_fruiting_leaf_area[j]
        area += self.main_stem_leaf_area.sum()
        area += self.node_leaf_area.sum()
        return area

    @property
    def number_of_open_bolls(self) -> float:
        return self.fruiting_nodes_fraction[self.fruiting_nodes_stage == Stage.MatureBoll].sum()

    cdef public double _leaf_nitrogen_concentration

    @property
    def leaf_nitrogen_concentration(self):
        if self.leaf_weight > 0.00001:
            self._leaf_nitrogen_concentration = self.leaf_nitrogen / self.leaf_weight
        return self._leaf_nitrogen_concentration

    @property
    def leaf_weight_area_ratio(self):
        return self._.leaf_weight_area_ratio

    @leaf_weight_area_ratio.setter
    def leaf_weight_area_ratio(self, value):
        self._.leaf_weight_area_ratio = value

    @property
    def petiole_nitrogen_concentration(self):
        return self._.petiole_nitrogen_concentration

    @petiole_nitrogen_concentration.setter
    def petiole_nitrogen_concentration(self, value):
        self._.petiole_nitrogen_concentration = value

    @property
    def seed_nitrogen(self):
        return self._.seed_nitrogen

    @seed_nitrogen.setter
    def seed_nitrogen(self, value):
        self._.seed_nitrogen = value

    @property
    def seed_nitrogen_concentration(self):
        return self._.seed_nitrogen_concentration

    @seed_nitrogen_concentration.setter
    def seed_nitrogen_concentration(self, value):
        self._.seed_nitrogen_concentration = value

    @property
    def burr_nitrogen(self):
        return self._.burr_nitrogen

    @burr_nitrogen.setter
    def burr_nitrogen(self, value):
        self._.burr_nitrogen = value

    @property
    def burr_nitrogen_concentration(self):
        return self._.burr_nitrogen_concentration

    @burr_nitrogen_concentration.setter
    def burr_nitrogen_concentration(self, value):
        self._.burr_nitrogen_concentration = value

    @property
    def root_nitrogen_concentration(self):
        return self._.root_nitrogen_concentration

    @root_nitrogen_concentration.setter
    def root_nitrogen_concentration(self, value):
        self._.root_nitrogen_concentration = value

    @property
    def root_nitrogen(self):
        return self._.root_nitrogen

    @root_nitrogen.setter
    def root_nitrogen(self, value):
        self._.root_nitrogen = value

    @property
    def square_nitrogen(self):
        return self._.square_nitrogen

    @square_nitrogen.setter
    def square_nitrogen(self, value):
        self._.square_nitrogen = value

    @property
    def square_nitrogen_concentration(self):
        return self._.square_nitrogen_concentration

    @square_nitrogen_concentration.setter
    def square_nitrogen_concentration(self, value):
        self._.square_nitrogen_concentration = value

    @property
    def stem_nitrogen_concentration(self):
        return self._.stem_nitrogen_concentration

    @stem_nitrogen_concentration.setter
    def stem_nitrogen_concentration(self, value):
        self._.stem_nitrogen_concentration = value

    @property
    def stem_nitrogen(self):
        return self._.stem_nitrogen

    @stem_nitrogen.setter
    def stem_nitrogen(self, value):
        self._.stem_nitrogen = value

    @property
    def fruit_growth_ratio(self):
        return self._.fruit_growth_ratio

    @fruit_growth_ratio.setter
    def fruit_growth_ratio(self, value):
        self._.fruit_growth_ratio = value

    @property
    def number_of_pre_fruiting_nodes(self):
        return self._.number_of_pre_fruiting_nodes

    @number_of_pre_fruiting_nodes.setter
    def number_of_pre_fruiting_nodes(self, value):
        self._.number_of_pre_fruiting_nodes = value

    @property
    def total_actual_leaf_growth(self):
        return self._.total_actual_leaf_growth

    @total_actual_leaf_growth.setter
    def total_actual_leaf_growth(self, value):
        self._.total_actual_leaf_growth = value

    @property
    def total_actual_petiole_growth(self):
        return self._.total_actual_petiole_growth

    @total_actual_petiole_growth.setter
    def total_actual_petiole_growth(self, value):
        self._.total_actual_petiole_growth = value

    @property
    def actual_burr_growth(self):
        return self._.actual_burr_growth

    @actual_burr_growth.setter
    def actual_burr_growth(self, value):
        self._.actual_burr_growth = value

    @property
    def supplied_nitrate_nitrogen(self):
        return self._.supplied_nitrate_nitrogen

    @supplied_nitrate_nitrogen.setter
    def supplied_nitrate_nitrogen(self, value):
        self._.supplied_nitrate_nitrogen = value

    @property
    def supplied_ammonium_nitrogen(self):
        return self._.supplied_ammonium_nitrogen

    @supplied_ammonium_nitrogen.setter
    def supplied_ammonium_nitrogen(self, value):
        self._.supplied_ammonium_nitrogen = value

    @property
    def deep_soil_temperature(self):
        return self._.deep_soil_temperature

    @deep_soil_temperature.setter
    def deep_soil_temperature(self, value):
        self._.deep_soil_temperature = value

    @property
    def petiole_nitrogen(self):
        return self._.petiole_nitrogen

    @petiole_nitrogen.setter
    def petiole_nitrogen(self, value):
        self._.petiole_nitrogen = value

    @property
    def petiole_nitrate_nitrogen_concentration(self):
        return self._.petiole_nitrate_nitrogen_concentration

    @petiole_nitrate_nitrogen_concentration.setter
    def petiole_nitrate_nitrogen_concentration(self, value):
        self._.petiole_nitrate_nitrogen_concentration = value

    @property
    def phenological_delay_by_nitrogen_stress(self):
        """the delay caused by nitrogen stress, is assumed to be a function of the vegetative nitrogen stress."""
        return min(max(0.65 * (1 - self.nitrogen_stress_vegetative), 0), 1)

    def predict_emergence(self, plant_date, hour, plant_row_column):
        """This function predicts date of emergence."""
        cdef double dpl = 5  # depth of planting, cm (assumed 5).
        # Define some initial values on day of planting.
        if self.date == plant_date and hour == 0:
            self.delay_of_emergence = 0
            self.hypocotyl_length = 0.3
            self.seed_moisture = 8
            # Compute soil layer number for seed depth.
            self.seed_layer_number = np.searchsorted(self.layer_depth_cumsum, dpl)
        # Compute matric soil moisture potential at seed location.
        # Define te as soil temperature at seed location, C.
        cdef double psi  # matric soil moisture potential at seed location.
        cdef double te  # soil temperature at seed depth, C.
        psi = self.soil_psi[self.seed_layer_number, plant_row_column]
        te = self.hourly_soil_temperature[hour, self.seed_layer_number, plant_row_column] - 273.161
        te = max(te, 10)

        # Phase 1 of of germination - imbibition. This phase is executed when the moisture content of germinating seeds is not more than 80%.
        cdef double dw  # rate of moisture addition to germinating seeds, percent per hour.
        cdef double xkl  # a function of temperature and moisture, used to calculate dw.
        if self.seed_moisture <= 80:
            xkl = .0338 + .0000855 * te * te - 0.003479 * psi
            if xkl < 0:
                xkl = 0
            # Compute the rate of moisture addition to germinating seeds, percent per hour.
            dw = xkl * (80 - self.seed_moisture)
            # Compute delw, the marginal value of dw, as a function of soil temperature and soil water potential.
            delw = 0  # marginal value of dw.
            if te < 21.2:
                delw = -0.1133 + .000705 * te ** 2 - .001348 * psi + .001177 * psi ** 2
            elif te < 26.66:
                delw = -.3584 + .001383 * te ** 2 - .03509 * psi + .003507 * psi ** 2
            elif te < 32.3:
                delw = -.6955 + .001962 * te ** 2 - .08335 * psi + .007627 * psi ** 2 - .006411 * psi * te
            else:
                delw = 3.3929 - .00197 * te ** 2 - .36935 * psi + .00865 * psi ** 2 + .007306 * psi * te
            if delw < 0.01:
                delw = 0.01
            # Add dw to tw, or if dw is less than delw assign 100% to tw.
            if dw > delw:
                self.seed_moisture += dw
            else:
                self.seed_moisture = 100
            return

        # Phase 2 of of germination - hypocotyl elongation.
        cdef double xt  # a function of temperature, used to calculate de.
        if te > 39.9:
            xt = 0
        else:
            xt = 0.0853 - 0.0057 * (te - 34.44) * (te - 34.44) / (41.9 - te)
        # At low soil temperatures, when negative values of xt occur, compute the delay in germination rate.
        if xt < 0 and te < 14:
            self.delay_of_emergence += xt / 2
            return
        else:
            if self.delay_of_emergence < 0:
                if self.delay_of_emergence + xt < 0:
                    self.delay_of_emergence += xt
                    return
                else:
                    xt += self.delay_of_emergence
                    self.delay_of_emergence = 0
        # Compute elongation rate of hypocotyl, de, as a sigmoid function of HypocotylLength. Add de to HypocotylLength.
        cdef double de  # rate of hypocotyl growth, cm per hour.
        de = 0.0567 * xt * self.hypocotyl_length * (10 - self.hypocotyl_length)
        self.hypocotyl_length += de
        # Check for completion of emergence (when HypocotylLength exceeds planting depth) and report germination to output.
        if self.hypocotyl_length > dpl:
            self.kday = 1
            return self.date

    def pre_fruiting_node(self, stemNRatio, time_to_next_pre_fruiting_node, time_factor_for_first_two_pre_fruiting_nodes, time_factor_for_third_pre_fruiting_node, initial_pre_fruiting_nodes_leaf_area):
        """This function checks if a new prefruiting node is to be added, and then sets it."""
        # The following constant parameter is used:
        cdef double MaxAgePreFrNode = 66  # maximum age of a prefruiting node (constant)
        # When the age of the last prefruiting node exceeds MaxAgePreFrNode, this function is not activated.
        if self.pre_fruiting_nodes_age[self.number_of_pre_fruiting_nodes - 1] > MaxAgePreFrNode:
            return
        # Loop over all existing prefruiting nodes.
        # Increment the age of each prefruiting node in physiological days.
        for i in range(self.number_of_pre_fruiting_nodes):
            self.pre_fruiting_nodes_age[i] += self.day_inc
        # For the last prefruiting node (if there are less than 9 prefruiting nodes):
        # The period (timeToNextPreFruNode) until the formation of the next node is VarPar(31), but it is modified for the first three nodes.
        # If the physiological age of the last prefruiting node is more than timeToNextPreFruNode, form a new prefruiting node - increase state.number_of_pre_fruiting_nodes, assign the initial average temperature for the new node, and initiate a new leaf on this node.
        if self.number_of_pre_fruiting_nodes >= 9:
            return
        # time, in physiological days, for the next prefruiting node to be formed.
        if self.number_of_pre_fruiting_nodes <= 2:
            time_to_next_pre_fruiting_node *= time_factor_for_first_two_pre_fruiting_nodes
        elif self.number_of_pre_fruiting_nodes == 3:
            time_to_next_pre_fruiting_node *= time_factor_for_third_pre_fruiting_node

        if self.pre_fruiting_nodes_age[self.number_of_pre_fruiting_nodes - 1] >= time_to_next_pre_fruiting_node:
            if self.version >= 0x500:
                leaf_weight = min(initial_pre_fruiting_nodes_leaf_area * self.leaf_weight_area_ratio, self.stem_weight - 0.2)
                if leaf_weight <= 0:
                    return
                leaf_area = leaf_weight / self.leaf_weight_area_ratio
            else:
                leaf_area = initial_pre_fruiting_nodes_leaf_area
                leaf_weight = leaf_area * self.leaf_weight_area_ratio
            self.number_of_pre_fruiting_nodes += 1
            self.pre_fruiting_leaf_area[self.number_of_pre_fruiting_nodes - 1] = leaf_area
            self.leaf_weight_pre_fruiting[self.number_of_pre_fruiting_nodes - 1] = leaf_weight
            self.leaf_weight += leaf_weight
            self.stem_weight -= leaf_weight
            self.leaf_nitrogen += leaf_weight * stemNRatio
            self.stem_nitrogen -= leaf_weight * stemNRatio

    def leaf_water_potential(self, double row_space):
        """This function simulates the leaf water potential of cotton plants.

        It has been adapted from the model of Moshe Meron (The relation of cotton leaf water potential to soil water content in the irrigated management range. PhD dissertation, UC Davis, 1984).
        """
        # Constant parameters used:
        cdef double cmg = 3200  # length in cm per g dry weight of roots, based on an average
        # root diameter of 0.06 cm, and a specific weight of 0.11 g dw per cubic cm.
        cdef double psild0 = -1.32  # maximum values of min_leaf_water_potential
        cdef double psiln0 = -0.40  # maximum values of self.max_leaf_water_potential.
        cdef double rtdiam = 0.06  # average root diameter in cm.
        cdef double[13] vpsil = [0.48, -5.0, 27000., 4000., 9200., 920., 0.000012, -0.15, -1.70, -3.5, 0.1e-9, 0.025, 0.80]
        # Leaf water potential is not computed during 10 days after emergence. Constant values are assumed for this period.
        if self.kday <= 10:
            self.max_leaf_water_potential = psiln0
            self.min_leaf_water_potential = psild0
            return
        # Compute shoot resistance (rshoot) as a function of plant height.
        cdef double rshoot  # shoot resistance, Mpa hours per cm.
        rshoot = vpsil[0] * self.plant_height / 100
        # Assign zero to summation variables
        cdef double psinum = 0  # sum of RootWtCapblUptake for all soil cells with roots.
        cdef double rootvol = 0  # sum of volume of all soil cells with roots.
        cdef double rrlsum = 0  # weighted sum of reciprocals of rrl.
        cdef double rroot = 0  # root resistance, Mpa hours per cm.
        cdef double sumlv = 0  # weighted sum of root length, cm, for all soil cells with roots.
        cdef double vh2sum = 0  # weighted sum of soil water content, for all soil cells with roots.
        # Loop over all soil cells with roots. Check if RootWtCapblUptake is greater than vpsil[10].
        # All average values computed for the root zone, are weighted by RootWtCapblUptake (root weight capable of uptake), but the weight assigned will not be greater than vpsil[11].
        cdef double rrl  # root resistance per g of active roots.
        for l in range(40):
            for k in range(20):
                if self.root_weight_capable_uptake[l, k] >= vpsil[10]:
                    psinum += min(self.root_weight_capable_uptake[l, k], vpsil[11])
                    sumlv += min(self.root_weight_capable_uptake[l, k], vpsil[11]) * cmg
                    rootvol += self.layer_depth[l] * self._sim.column_width[k]
                    if self.soil_psi[l, k] <= vpsil[1]:
                        rrl = vpsil[2] / cmg
                    else:
                        rrl = (vpsil[3] - self.soil_psi[l, k] * (vpsil[4] + vpsil[5] * self.soil_psi[l, k])) / cmg
                    rrlsum += min(self.root_weight_capable_uptake[l, k], vpsil[11]) / rrl
                    vh2sum += self.soil_water_content[l, k] * min(self.root_weight_capable_uptake[l, k], vpsil[11])
        # Compute average root resistance (rroot) and average soil water content (vh2).
        cdef double dumyrs  # intermediate variable for computing cond.
        cdef double vh2  # average of soil water content, for all soil soil cells with roots.
        if psinum > 0 and sumlv > 0:
            rroot = psinum / rrlsum
            vh2 = vh2sum / psinum
            dumyrs = max(sqrt(1 / (pi * sumlv / rootvol)) / rtdiam, 1.001)
        else:
            rroot = 0
            vh2 = self.thad[0]
            dumyrs = 1.001
        # Compute hydraulic conductivity (cond), and soil resistance near the root surface  (rsoil).
        cdef double cond  # soil hydraulic conductivity near the root surface.
        cond = wcond(vh2, self.thad[0], self.soil_saturated_water_content[0], self.beta[0], self.soil_saturated_hydraulic_conductivity[0], self.pore_space[0]) / 24
        cond = cond * 2 * sumlv / rootvol / log(dumyrs)
        cond = max(cond, vpsil[6])
        cdef double rsoil = 0.0001 / (2 * pi * cond)  # soil resistance, Mpa hours per cm.
        # Compute leaf resistance (leaf_resistance_for_transpiration) as the average of the resistances of all existing leaves.
        # The resistance of an individual leaf is a function of its age.
        # Function leaf_resistance_for_transpiration is called to compute it. This is executed for all the leaves of the plant.
        cdef int numl = 0  # number of leaves.
        cdef double sumrl = 0  # sum of leaf resistances for all the plant.
        for j in range(self.number_of_pre_fruiting_nodes):  # loop prefruiting nodes
            numl += 1
            sumrl += leaf_resistance_for_transpiration(self.pre_fruiting_nodes_age[j])

        for i, s in np.ndenumerate(self.fruiting_nodes_stage):
            if s != Stage.NotYetFormed:
                numl += 1
                sumrl += leaf_resistance_for_transpiration(self.node_leaf_age[i])
        cdef double rleaf = sumrl / numl  # leaf resistance, Mpa hours per cm.

        cdef double rtotal = rsoil + rroot + rshoot + rleaf  # The total resistance to transpiration, MPa hours per cm, (rtotal) is computed.
        # Compute maximum (early morning) leaf water potential, max_leaf_water_potential, from soil water potential (average_soil_psi, converted from bars to MPa).
        # Check for minimum and maximum values.
        self.max_leaf_water_potential = min(max(vpsil[7] + 0.1 * self.average_soil_psi, vpsil[8]), psiln0)
        # Compute minimum (at time of maximum transpiration rate) leaf water potential, min_leaf_water_potential, from maximum transpiration rate (etmax) and total resistance to transpiration (rtotal).
        cdef double etmax = 0  # the maximum hourly rate of evapotranspiration for this day.
        for ihr in range(24):  # hourly loop
            if self.hours[ihr].ref_et > etmax:
                etmax = self.hours[ihr].ref_et
        self.min_leaf_water_potential = min(max(self.max_leaf_water_potential - 0.1 * max(etmax, vpsil[12]) * rtotal, vpsil[9]), psild0)

    def actual_leaf_growth(self, vratio):
        """This function simulates the actual growth of leaves of cotton plants. It is called from PlantGrowth()."""
        # Loop for all prefruiting node leaves. Added dry weight to each leaf is proportional to PotGroLeafWeightPreFru. Update leaf weight (state.leaf_weight_pre_fruiting) and leaf area (state.pre_fruiting_leaf_area) for each prefruiting node leaf. added dry weight to each petiole is proportional to PotGroPetioleWeightPreFru. update petiole weight (PetioleWeightPreFru) for each prefruiting node leaf.
        # Compute total leaf weight (state.leaf_weight), total petiole weight (PetioleWeightNodes).
        for j in range(self.number_of_pre_fruiting_nodes): # loop by prefruiting node.
            self.leaf_weight_pre_fruiting[j] += PotGroLeafWeightPreFru[j] * vratio
            self.leaf_weight += self.leaf_weight_pre_fruiting[j]
            PetioleWeightPreFru[j] += PotGroPetioleWeightPreFru[j] * vratio
            self.petiole_weight += PetioleWeightPreFru[j]
            self.pre_fruiting_leaf_area[j] += PotGroLeafAreaPreFru[j] * vratio
        # Loop for all fruiting branches on each vegetative branch, to compute actual growth of mainstem leaves.
        # Added dry weight to each leaf is proportional to PotGroLeafWeightMainStem, added dry weight to each petiole is proportional to PotGroPetioleWeightMainStem, and added area to each leaf is proportional to PotGroLeafAreaMainStem.
        # Update leaf weight (LeafWeightMainStem), petiole weight (PetioleWeightMainStem) and leaf area(LeafAreaMainStem) for each main stem node leaf.
        # Update the total leaf weight (state.leaf_weight), total petiole weight (state.petiole_weight).
        self.main_stem_leaf_weight += self.main_stem_leaf_potential_growth * vratio
        self.leaf_weight += self.main_stem_leaf_weight.sum()
        self.main_stem_leaf_petiole_weight += self.main_stem_leaf_petiole_potential_growth * vratio
        self.petiole_weight += self.main_stem_leaf_petiole_weight.sum()
        self.main_stem_leaf_area += self.main_stem_leaf_area_potential_growth * vratio
        self.node_leaf_weight += self.node_leaf_area_potential_growth * self.leaf_weight_area_ratio * vratio
        self.leaf_weight += self.node_leaf_weight.sum()
        self.node_petiole_weight += self.node_petiole_potential_growth * vratio
        self.petiole_weight += self.node_petiole_weight.sum()
        self.node_leaf_area += self.node_leaf_area_potential_growth * vratio

    def actual_fruit_growth(self):
        """This function simulates the actual growth of squares and bolls of cotton plants."""
        # Assign zero to all the sums to be computed.
        self.green_bolls_burr_weight = 0
        self.actual_boll_growth = 0
        self.actual_burr_growth = 0
        # Begin loops over all fruiting sites.
        for i, stage in np.ndenumerate(self.fruiting_nodes_stage):
            # If this site is a square, the actual dry weight added to it (dwsq) is proportional to its potential growth.
            if stage == Stage.Square:
                dwsq = self.square_potential_growth[i] * self.fruit_growth_ratio  # dry weight added to square.

                self.square_weights[i] += dwsq
            # If this site is a green boll, the actual dry weight added to seedcotton and burrs is proportional to their respective potential growth.
            if stage in [Stage.GreenBoll, Stage.YoungGreenBoll]:
                # dry weight added to seedcotton in a boll.
                dwboll = self.fruiting_nodes_boll_potential_growth[i] * self.fruit_growth_ratio
                self.fruiting_nodes_boll_weight[i] += dwboll
                self.actual_boll_growth += dwboll
                # dry weight added to the burrs in a boll.
                dwburr = self.burr_potential_growth[i] * self.fruit_growth_ratio
                self.burr_weight[i] += dwburr
                self.actual_burr_growth += dwburr
                self.green_bolls_burr_weight += self.burr_weight[i]

    def dry_matter_balance(self, per_plant_area) -> float:
        """This function computes the cotton plant dry matter (carbon) balance, its allocation to growing plant parts, and carbon stress. It is called from PlantGrowth()."""
        # The following constant parameters are used:
        vchbal = [6.0, 2.5, 1.0, 5.0, 0.20, 0.80, 0.48, 0.40, 0.2072, 0.60651, 0.0065, 1.10, 4.0, 0.25, 4.0]
        # Assign values for carbohydrate requirements for growth of stems, roots, leaves, petioles, squares and bolls. Potential growth of all plant parts is modified by nitrogen stresses.
        # carbohydrate requirement for square growth, g per plant per day.
        cdsqar = PotGroAllSquares * (self.nitrogen_stress_fruiting + vchbal[0]) / (vchbal[0] + 1)
        # carbohydrate requirement for boll and burr growth, g per plant per day.
        cdboll = (PotGroAllBolls + PotGroAllBurrs) * (self.nitrogen_stress_fruiting + vchbal[0]) / (vchbal[0] + 1)
        # cdleaf is carbohydrate requirement for leaf growth, g per plant per day.
        cdleaf = self.leaf_potential_growth * (self.nitrogen_stress_vegetative + vchbal[1]) / (vchbal[1] + 1)
        # cdstem is carbohydrate requirement for stem growth, g per plant per day.
        cdstem = self.stem_potential_growth * (self.nitrogen_stress_vegetative + vchbal[2]) / (vchbal[2] + 1)
        # cdroot is carbohydrate requirement for root growth, g per plant per day.
        cdroot = self.root_potential_growth * (self.nitrogen_stress_root + vchbal[3]) / (vchbal[3] + 1)
        # cdpet is carbohydrate requirement for petiole growth, g per plant per day.
        cdpet = self.petiole_potential_growth * (self.nitrogen_stress_vegetative + vchbal[14]) / (vchbal[14] + 1)
        # total carbohydrate requirement for plant growth, g per plant per day.
        cdsum = cdstem + cdleaf + cdpet + cdroot + cdsqar + cdboll
        # Compute CarbonStress as the ratio of available to required carbohydrates.
        if cdsum <= 0:
            self.carbon_stress = 1
            return 0  # Exit function if cdsum is 0.
        # total available carbohydrates for growth (cpool, g per plant).
        # cpool is computed as: net photosynthesis plus a fraction (vchbal(13) ) of the stored reserves (reserve_carbohydrate).
        cpool = self.net_photosynthesis + self.reserve_carbohydrate * vchbal[13]
        self.carbon_stress = min(1, cpool / cdsum)
        # When carbohydrate supply is sufficient for growth requirements, CarbonStress will be assigned 1, and the carbohydrates actually supplied for plant growth (total_actual_leaf_growth, total_actual_petiole_growth, actual_stem_growth, carbon_allocated_for_root_growth, pdboll, pdsq) will be equal to the required amounts.
        # pdboll is amount of carbohydrates allocated to boll growth.
        # pdsq is amount of carbohydrates allocated to square growth.
        if self.carbon_stress == 1:
            self.total_actual_leaf_growth = cdleaf
            self.total_actual_petiole_growth = cdpet
            self.actual_stem_growth = cdstem
            self.carbon_allocated_for_root_growth = cdroot
            pdboll = cdboll
            pdsq = cdsqar
            xtrac1 = 0
        # When carbohydrate supply is less than the growth requirements, set priorities for allocation of carbohydrates.
        else:
            # cavail remaining available carbohydrates.
            # First priority is for fruit growth. Compute the ratio of available carbohydrates to the requirements for boll and square growth (bsratio).
            if cdboll + cdsqar > 0:
                # ratio of available carbohydrates to the requirements for boll and square growth.
                bsratio = cpool / (cdboll + cdsqar)
                # ffr is ratio of actual supply of carbohydrates to the requirement for boll and square growth.
                # The factor ffr is a function of bsratio and WaterStress. It is assumed that water stress increases allocation of carbohydrates to bolls. Check that ffr is not less than zero, or greater than 1 or than bsratio.
                ffr = min(max((vchbal[5] + vchbal[6] * (1 - self.water_stress)) * bsratio, 0), 1)
                ffr = min(bsratio, ffr)
                # Now compute the actual carbohydrates used for boll and square growth, and the remaining available carbohydrates.
                pdboll = cdboll * ffr
                pdsq = cdsqar * ffr
                cavail = cpool - pdboll - pdsq
            else:
                cavail = cpool
                pdboll = 0
                pdsq = 0
            # The next priority is for leaf and petiole growth. Compute the factor flf for leaf growth allocation, and check that it is not less than zero or greater than 1.
            if cdleaf + cdpet > 0:
                # ratio of actual supply of carbohydrates to the requirement for leaf growth.
                flf = min(max(vchbal[7] * cavail / (cdleaf + cdpet), 0), 1)
                # Compute the actual carbohydrates used for leaf and petiole growth, and the
                # remaining available carbohydrates.
                self.total_actual_leaf_growth = cdleaf * flf
                self.total_actual_petiole_growth = cdpet * flf
                cavail -= self.total_actual_leaf_growth + self.total_actual_petiole_growth
            else:
                self.total_actual_leaf_growth = 0
                self.total_actual_petiole_growth = 0
            # The next priority is for root growth.
            if cdroot > 0:
                # ratio between carbohydrate supply to root and to stem growth.
                # At no water stress conditions, ratio is an exponential function of dry weight of vegetative shoot (stem + leaves). This equation is based on data from Avi Ben-Porath's PhD thesis.
                # ratio is modified (calibrated) by vchbal[11].
                ratio = vchbal[8] + vchbal[9] * exp(-vchbal[10] * (self.stem_weight + self.leaf_weight + self.petiole_weight) *
                                                    per_plant_area)
                ratio *= vchbal[11]
                # rtmax is the proportion of remaining available carbohydrates that can be supplied to root growth. This is increased by water stress.
                rtmax = ratio / (ratio + 1)
                rtmax = rtmax * (1 + vchbal[12] * (1 - self.water_stress))
                rtmax = min(rtmax, 1)
                # Compute the factor frt for root growth allocation, as a function of rtmax, and check that it is not less than zero or greater than 1.
                # ratio of actual supply of carbohydrates to the requirement for root growth.
                frt = min(max(rtmax * cavail / cdroot, 0), 1)
                # Compute the actual carbohydrates used for root growth, and the remaining available carbohydrates.
                self.carbon_allocated_for_root_growth = max((cdroot * frt), (cavail - cdstem))
                cavail -= self.carbon_allocated_for_root_growth
            else:
                self.carbon_allocated_for_root_growth = 0
            # The remaining available carbohydrates are used for stem growth. Compute thefactor fst and the actual carbohydrates used for stem growth.
            if cdstem > 0:
                # ratio of actual supply of carbohydrates to the requirement for stem growth.
                fst = min(max(cavail / cdstem, 0), 1)
                self.actual_stem_growth = cdstem * fst
            else:
                self.actual_stem_growth = 0
            # If there are any remaining available unused carbohydrates, define them as xtrac1.
            xtrac1 = max(cavail - self.actual_stem_growth, 0)
        # Check that the amounts of carbohydrates supplied to each organ will not be less than zero.
        self.actual_stem_growth = max(self.actual_stem_growth, 0)
        self.total_actual_leaf_growth = max(self.total_actual_leaf_growth, 0)
        self.total_actual_petiole_growth = max(self.total_actual_petiole_growth, 0)
        self.carbon_allocated_for_root_growth = max(self.carbon_allocated_for_root_growth, 0)
        pdboll = max(pdboll, 0)
        pdsq = max(pdsq, 0)
        # Update the amount of reserve carbohydrates (reserve_carbohydrate) in the leaves.
        self.reserve_carbohydrate += self.net_photosynthesis - (self.actual_stem_growth + self.total_actual_leaf_growth + self.total_actual_petiole_growth + self.carbon_allocated_for_root_growth + pdboll + pdsq)
        # maximum possible amount of carbohydrate reserves that can be stored in the leaves.
        # resmax is a fraction (vchbal[4])) of leaf weight. Excessive reserves are defined as xtrac2.
        resmax = vchbal[4] * self.leaf_weight
        if self.reserve_carbohydrate > resmax:
            xtrac2 = self.reserve_carbohydrate - resmax
            self.reserve_carbohydrate = resmax
        else:
            xtrac2 = 0
        # ExtraCarbon is computed as total excessive carbohydrates.
        self.extra_carbon = xtrac1 + xtrac2
        # Compute state.fruit_growth_ratio as the ratio of carbohydrates supplied to square and boll growth to their carbohydrate requirements.
        if PotGroAllSquares + PotGroAllBolls + PotGroAllBurrs > 0:
            self.fruit_growth_ratio = (pdsq + pdboll) / (PotGroAllSquares + PotGroAllBolls + PotGroAllBurrs)
        else:
            self.fruit_growth_ratio = 1
        # Compute vratio as the ratio of carbohydrates supplied to leaf and petiole growth to their carbohydrate requirements.
        if self.leaf_potential_growth + self.petiole_potential_growth > 0:
            vratio = (self.total_actual_leaf_growth + self.total_actual_petiole_growth) / (self.leaf_potential_growth + self.petiole_potential_growth)
        else:
            vratio = 1
        return vratio

    def init_root_data(self, uint32_t plant_row_column, double mul):
        self.root_growth_factor = np.ones((40, 20), dtype=np.double)
        self.root_weight_capable_uptake = np.ones((40, 20), dtype=np.double)
        # FIXME: I consider the value is incorrect
        self.root_weights = np.zeros((40, 20, 3), dtype=np.float64)
        self.root_weights[0,(plant_row_column - 1, plant_row_column + 2),0] = 0.0020
        self.root_weights[0,(plant_row_column, plant_row_column + 1),0] = 0.0070
        self.root_weights[1,(plant_row_column - 1, plant_row_column + 2),0] = 0.0040
        self.root_weights[1,(plant_row_column, plant_row_column + 1),0] = 0.0140
        self.root_weights[2,(plant_row_column - 1, plant_row_column + 2),0] = 0.0060
        self.root_weights[2,(plant_row_column, plant_row_column + 1),0] = 0.0210
        for l, w in zip(range(3, 7), (0.0200, 0.0150, 0.0100, 0.0050)):
            self.root_weights[l, (plant_row_column, plant_row_column + 1), 0] = w
        self.root_weights[:] *= mul
        self.root_age[:3,plant_row_column - 1:plant_row_column + 3] = 0.01
        self.root_age[3:7,plant_row_column:plant_row_column + 2] = 0.01

    def compute_root_impedance(self, bulk_density):
        """Calculates soil mechanical impedance to root growth, rtimpd(l,k), for all
        soil cells.
        The impedance is a function of bulk density and water content in each soil soil
        cell. No changes have been made in the original GOSSYM code."""
        water_content = self.soil_water_content / bulk_density[:, None]
        soil_imp = np.genfromtxt(Path(__file__).parent / "soil_imp.csv", delimiter=",")
        f = interp2d(soil_imp[0, 1:], soil_imp[1:, 0], soil_imp[1:, 1:])
        self.root_impedance = np.array([
            [f(bulk_density[l], water_content[l][k]) for k in range(20)]
            for l in range(40)
        ], dtype=np.double)

    def root_death(self, l, k):
        """This function computes the death of root tissue in each soil cell containing roots.

        When root age reaches a threshold thdth(i), a proportion dth(i) of the roots in class i dies. The mass of dead roots is added to DailyRootLoss.

        It has been adapted from GOSSYM, but the threshold age for this process is based on the time from when the roots first grew into each soil cell.

        It is assumed that root death rate is greater in dry soil, for all root classes except class 1. Root death rate is increased to the maximum value in soil saturated with water.
        """
        cdef double aa = 0.008  # a parameter in the equation for computing dthfac.
        cdef double[3] dth = [0.0001, 0.0002, 0.0001]  # the daily proportion of death of root tissue.
        cdef double dthmax = 0.10  # a parameter in the equation for computing dthfac.
        cdef double psi0 = -14.5  # a parameter in the equation for computing dthfac.
        cdef double[3] thdth = [30.0, 50.0, 100.0]  # the time threshold, from the initial
        # penetration of roots to a soil cell, after which death of root tissue of class i may occur.

        result = 0
        for i in range(3):
            if self.root_age[l][k] > thdth[i]:
                # the computed proportion of roots dying in each class.
                dthfac = dth[i]
                if self.soil_water_content[l, k] >= self.pore_space[l]:
                    dthfac = dthmax
                else:
                    if i <= 1 and self.soil_psi[l, k] <= psi0:
                        dthfac += aa * (psi0 - self.soil_psi[l, k])
                    if dthfac > dthmax:
                        dthfac = dthmax
                result += self.root_weights[l][k][i] * dthfac
                self.root_weights[l][k][i] -= self.root_weights[l][k][i] * dthfac
        return result

    def tap_root_growth(self, int NumRootAgeGroups, unsigned int plant_row_column):
        """This function computes the elongation of the taproot."""
        # Call function TapRootGrowth() for taproot elongation, if the taproot has not already reached the bottom of the slab.
        if self.taproot_layer_number >= nl - 1 and self.taproot_length >= self.last_layer_with_root_depth:
            return
        # The following constant parameters are used:
        cdef double p1 = 0.10  # constant parameter.
        cdef double rtapr = 4  # potential growth rate of the taproot, cm/day.
        # It is assumed that taproot elongation takes place irrespective of the supply of carbon to the roots. This elongation occurs in the two columns of the slab where the plant is located.
        # Tap root elongation does not occur in water logged soil (water table).
        cdef int klocp1 = plant_row_column + 1  # the second column in which taproot growth occurs.
        if self.soil_water_content[self.taproot_layer_number, plant_row_column] >= self.pore_space[self.taproot_layer_number] or self.soil_water_content[self.taproot_layer_number, klocp1] >= self.pore_space[self.taproot_layer_number]:
            return
        # Average soil resistance (avres) is computed at the root tip.
        # avres = average value of RootGroFactor for the two soil cells at the tip of the taproot.
        cdef double avres = 0.5 * (self.root_growth_factor[self.taproot_layer_number, plant_row_column] + self.root_growth_factor[self.taproot_layer_number, klocp1])
        # It is assumed that a linear empirical function of avres controls the rate of taproot elongation. The potential elongation rate of the taproot is also modified by soil temperature (SoilTemOnRootGrowth function), soil resistance, and soil moisture near the root tip.
        # Actual growth is added to the taproot_length.
        cdef double stday  # daily average soil temperature (C) at root tip.
        stday = 0.5 * (self.soil_temperature[self.taproot_layer_number][plant_row_column] + self.soil_temperature[self.taproot_layer_number][klocp1]) - 273.161
        cdef double addtaprt  # added taproot length, cm
        addtaprt = rtapr * (1 - p1 + avres * p1) * SoilTemOnRootGrowth(stday)
        self.taproot_length += addtaprt
        # last_layer_with_root_depth, the depth (in cm) to the end of the last layer with roots, is used to check if the taproot reaches a new soil layer.
        # When the new value of taproot_length is greater than last_layer_with_root_depth - it means that the roots penetrate to a new soil layer.
        # In this case, and if this is not the last layer in the slab, the following is executed:
        # self.taproot_layer_number and last_layer_with_root_depth are incremented. If this is a new layer with roots, state.soil.number_of_layers_with_root is also redefined and two soil cells of the new layer are defined as containing roots (by initializing RootColNumLeft and RootColNumRight).
        if self.taproot_layer_number > nl - 2 or self.taproot_length <= self.last_layer_with_root_depth:
            return
        # The following is executed when the taproot reaches a new soil layer.
        self.taproot_layer_number += 1
        self.last_layer_with_root_depth += self.layer_depth[self.taproot_layer_number]
        # RootAge is initialized for these soil cells.
        self.root_age[self.taproot_layer_number][plant_row_column] = 0.01
        self.root_age[self.taproot_layer_number][klocp1] = 0.01
        # Some of the mass of class 1 roots is transferred downwards to the new cells.
        # The transferred mass is proportional to 2 cm of layer width, but it is not more than half the existing mass in the last layer.
        for i in range(NumRootAgeGroups):
            # root mass transferred to the cell below when the elongating taproot
            # reaches a new soil layer.
            # first column
            tran = self.root_weights[self.taproot_layer_number - 1][plant_row_column][i] * 2 / self.layer_depth[self.taproot_layer_number - 1]
            if tran > 0.5 * self.root_weights[self.taproot_layer_number - 1][plant_row_column][i]:
                tran = 0.5 * self.root_weights[self.taproot_layer_number - 1][plant_row_column][i]
            self.root_weights[self.taproot_layer_number][plant_row_column][i] += tran
            self.root_weights[self.taproot_layer_number - 1][plant_row_column][i] -= tran
            # second column
            tran = self.root_weights[self.taproot_layer_number - 1][klocp1][i] * 2 / self.layer_depth[self.taproot_layer_number - 1]
            if tran > 0.5 * self.root_weights[self.taproot_layer_number - 1][klocp1][i]:
                tran = 0.5 * self.root_weights[self.taproot_layer_number - 1][klocp1][i]
            self.root_weights[self.taproot_layer_number][klocp1][i] += tran
            self.root_weights[self.taproot_layer_number - 1][klocp1][i] -= tran

    #begin root
    #                  THE COTTON ROOT SUB-MODEL.
    # The following is a documentation of the root sub-model used in COTTON2K. It is
    # derived from the principles of RHIZOS, as implemented in GOSSYM and in GLYCIM,
    # and from some principles of ROOTSIMU (Hoogenboom and Huck, 1986). It is devised
    # to be generally applicable, and may be used with root systems of different crops by
    # redefining the parameters, which are set here as constants, and some of them are
    # set in function InitializeRootData(). These parameters are of course specific for the
    # crop species, and perhaps also for cultivars or cultivar groups.
    #
    # This is a two-dimensional model and it may be used with soil cells of different
    # sizes. The grid can be defined by the modeler. The maximum numbers of layers
    # and columns are given by the parameters maxl and maxk, respectively. These are set
    # to 40 and 20, in this version of COTTON2K. The grid is set in function InitializeGrid().
    #
    # The whole slab is being simulated. Thus, non-symmetrical processes (such as
    # side-dressing of fertilizers or drip-irrigation) can be handled. The plant is assumed to
    # be situated at the center of the soil slab, or off-center for skip-rows. Adjoining soil
    # slabs are considered as mirror-images of each other. Alternate-row drip systems (or any
    # other agricultural input similarly situated) are located at one edge of the slab.
    #
    # The root mass in each cell is made up of NumRootAgeGroups classes, whose number is to be
    # defined by the modeler. The maximum number of classes is 3 in this version of COTTON2K.
    def lateral_root_growth_left(self, int l, int NumRootAgeGroups, unsigned int plant_row_column, double row_space):
        """This function computes the elongation of the lateral roots in a soil layer(l) to the left."""
        # The following constant parameters are used:
        cdef double p1 = 0.10  # constant parameter.
        cdef double rlatr = 3.6  # potential growth rate of lateral roots, cm/day.
        cdef double rtran = 0.2  # the ratio of root mass transferred to a new soil
        # soil cell, when a lateral root grows into it.
        # On its initiation, lateral root length is assumed to be equal to the width of a soil column soil cell at the location of the taproot.
        if self.rlat1[l] <= 0:
            self.rlat1[l] = self._sim.column_width[plant_row_column]
        cdef double stday  # daily average soil temperature (C) at root tip.
        stday = self.soil_temperature[l][plant_row_column] - 273.161
        cdef double temprg  # the effect of soil temperature on root growth.
        temprg = SoilTemOnRootGrowth(stday)
        # Define the column with the tip of this lateral root (ktip)
        cdef int ktip = 0  # column with the tips of the laterals to the left
        cdef double sumwk = 0  # summation of columns width
        for k in reversed(range(plant_row_column + 1)):
            sumwk += self._sim.column_width[k]
            if sumwk >= self.rlat1[l]:
                ktip = k
                break
        # Compute growth of the lateral root to the left.
        # Potential growth rate (u) is modified by the soil temperature function,
        # and the linearly modified effect of soil resistance (RootGroFactor).
        # Lateral root elongation does not occur in water logged soil.
        if self.soil_water_content[l, ktip] < self.pore_space[l]:
            self.rlat1[l] += rlatr * temprg * (1 - p1 + self.root_growth_factor[l, ktip] * p1)
            # If the lateral reaches a new soil soil cell: a proportion (tran) of mass of roots is transferred to the new soil cell.
            if self.rlat1[l] > sumwk and ktip > 0:
                # column into which the tip of the lateral grows to left.
                newktip = ktip - 1
                for i in range(NumRootAgeGroups):
                    tran = self.root_weights[l][ktip][i] * rtran
                    self.root_weights[l][ktip][i] -= tran
                    self.root_weights[l][newktip][i] += tran
                # RootAge is initialized for this soil cell.
                # RootColNumLeft of this layer idefi
                if self.root_age[l][newktip] == 0:
                    self.root_age[l][newktip] = 0.01

    def lateral_root_growth_right(self, int l, int NumRootAgeGroups, unsigned int plant_row_column, double row_space):
        # The following constant parameters are used:
        cdef double p1 = 0.10  # constant parameter.
        cdef double rlatr = 3.6  # potential growth rate of lateral roots, cm/day.
        cdef double rtran = 0.2  # the ratio of root mass transferred to a new soil
        # soil cell, when a lateral root grows into it.
        # On its initiation, lateral root length is assumed to be equal to the width of a soil column soil cell at the location of the taproot.
        cdef int klocp1 = plant_row_column + 1
        if self.rlat2[l] <= 0:
            self.rlat2[l] = self._sim.column_width[klocp1]
        cdef double stday  # daily average soil temperature (C) at root tip.
        stday = self.soil_temperature[l][klocp1] - 273.161
        cdef double temprg  # the effect of soil temperature on root growth.
        temprg = SoilTemOnRootGrowth(stday)
        # define the column with the tip of this lateral root (ktip)
        cdef int ktip = 0  # column with the tips of the laterals to the right
        cdef double sumwk = 0
        for k in range(klocp1, nk):
            sumwk += self._sim.column_width[k]
            if sumwk >= self.rlat2[l]:
                ktip = k
                break
        # Compute growth of the lateral root to the right. Potential growth rate is modified by the soil temperature function, and the linearly modified effect of soil resistance (RootGroFactor).
        # Lateral root elongation does not occur in water logged soil.
        if self.soil_water_content[l, ktip] < self.pore_space[l]:
            self.rlat2[l] += rlatr * temprg * (1 - p1 + self.root_growth_factor[l, ktip] * p1)
            # If the lateral reaches a new soil soil cell: a proportion (tran) of mass of roots is transferred to the new soil cell.
            if self.rlat2[l] > sumwk and ktip < nk - 1:
                # column into which the tip of the lateral grows to left.
                newktip = ktip + 1  # column into which the tip of the lateral grows to left.
                for i in range(NumRootAgeGroups):
                    tran = self.root_weights[l][ktip][i] * rtran
                    self.root_weights[l][ktip][i] -= tran
                    self.root_weights[l][newktip][i] += tran
                # RootAge is initialized for this soil cell.
                # RootColNumLeft of this layer is redefined.
                if self.root_age[l][newktip] == 0:
                    self.root_age[l][newktip] = 0.01

    def potential_root_growth(self, NumRootAgeGroups, per_plant_area):
        """
        This function calculates the potential root growth rate.
        The return value is the sum of potential root growth rates for the whole slab.
        It is called from PlantGrowth().
        It calls: RootImpedance(), SoilNitrateOnRootGrowth(), SoilAirOnRootGrowth(), SoilMechanicResistance(), SoilTemOnRootGrowth() and root_psi().
        """
        # The following constant parameter is used:
        cdef double rgfac = 0.36  # potential relative growth rate of the roots (g/g/day).
        # Initialize to zero the PotGroRoots array.
        self._root_potential_growth[:] = 0
        self.compute_root_impedance(self.soil_bulk_density)
        for l in range(40):
            for k in range(nk):
                # Check if this soil cell contains roots (if RootAge is greater than 0), and execute the following if this is true.
                # In each soil cell with roots, the root weight capable of growth rtwtcg is computed as the sum of RootWeight[l][k][i] * cgind[i] for all root classes.
                if self.root_age[l][k] > 0:
                    rtwtcg = 0  # root weight capable of growth in a soil soil cell.
                    for i in range(NumRootAgeGroups):
                        rtwtcg += self.root_weights[l][k][i] * cgind[i]
                    # Compute the temperature factor for root growth by calling function SoilTemOnRootGrowth() for this layer.
                    stday = self.soil_temperature[l][k] - 273.161  # soil temperature, C, this day's average for this cell.
                    temprg = SoilTemOnRootGrowth(stday)  # effect of soil temperature on root growth.
                    # Compute soil mechanical resistance for each soil cell by calling SoilMechanicResistance{}, the effect of soil aeration on root growth by calling SoilAirOnRootGrowth(), and the effect of soil nitrate on root growth by calling SoilNitrateOnRootGrowth().

                    lp1 = l if l == nl - 1 else l + 1  # layer below l.

                    # columns to the left and to the right of k.
                    kp1 = min(k + 1, nk - 1)
                    km1 = max(k - 1, 0)

                    rtimpd0 = self.root_impedance[l][k]
                    rtimpdkm1 = self.root_impedance[l][km1]
                    rtimpdkp1 = self.root_impedance[l][kp1]
                    rtimpdlp1 = self.root_impedance[lp1][k]
                    rtimpdmin = min(rtimpd0, rtimpdkm1, rtimpdkp1, rtimpdlp1)  # minimum value of rtimpd
                    rtpct = SoilMechanicResistance(rtimpdmin)  # effect of soil mechanical resistance on root growth (returned from SoilMechanicResistance).
                    # effect of oxygen deficiency on root growth (returned from SoilAirOnRootGrowth).
                    rtrdo = SoilAirOnRootGrowth(self.soil_psi[l, k], self.pore_space[l], self.soil_water_content[l, k])
                    # effect of nitrate deficiency on root growth (returned from SoilNitrateOnRootGrowth).
                    rtrdn = SoilNitrateOnRootGrowth(self.soil_nitrate_content[l, k])
                    # The root growth resistance factor RootGroFactor(l,k), which can take a value between 0 and 1, is computed as the minimum of these resistance factors. It is further modified by multiplying it by the soil moisture function root_psi().
                    # Potential root growth PotGroRoots(l,k) in each cell is computed as a product of rtwtcg, rgfac, the temperature function temprg, and RootGroFactor(l,k). It is also multiplied by per_plant_area / 19.6, for the effect of plant population density on root growth: it is made comparable to a population of 5 plants per m in 38" rows.
                    self.root_growth_factor[l, k] = root_psi(self.soil_psi[l, k]) * min(rtrdo, rtpct, rtrdn)
                    self._root_potential_growth[l][k] = rtwtcg * rgfac * temprg * self.root_growth_factor[l, k] * per_plant_area / 19.6
        return self._root_potential_growth.sum()

    def redist_root_new_growth(self, int l, int k, double addwt, double column_width, unsigned int plant_row_column):
        """This function computes the redistribution of new growth of roots into adjacent soil cells. It is called from ActualRootGrowth().

        Redistribution is affected by the factors rgfdn, rgfsd, rgfup.
        And the values of RootGroFactor(l,k) in this soil cell and in the adjacent cells.
        The values of ActualRootGrowth(l,k) for this and for the adjacent soil cells are computed.
        The code of this module is based, with major changes, on the code of GOSSYM."""
        # The following constant parameters are used. These are relative factors for root growth to adjoining cells, downwards, sideways, and upwards, respectively. These factors are relative to the volume of the soil cell from which growth originates.
        cdef double rgfdn = 900
        cdef double rgfsd = 600
        cdef double rgfup = 10
        # Set the number of layer above and below this layer, and the number of columns to the right and to the left of this column.
        cdef int lm1, lp1  # layer above and below layer l.
        lp1 = min(nl - 1, l + 1)
        lm1 = max(0, l - 1)

        cdef int km1, kp1  # column to the left and to the right of column k.
        kp1 = min(nk - 1, k + 1)
        km1 = max(0, k - 1)
        # Compute proportionality factors (efac1, efacl, efacr, efacu, efacd) as the product of RootGroFactor and the geotropic factors in the respective soil cells.
        # Note that the geotropic factors are relative to the volume of the soil cell.
        # Compute the sum srwp of the proportionality factors.
        cdef double efac1  # product of RootGroFactor and geotropic factor for this cell.
        cdef double efacd  # as efac1 for the cell below this cell.
        cdef double efacl  # as efac1 for the cell to the left of this cell.
        cdef double efacr  # as efac1 for the cell to the right of this cell.
        cdef double efacu  # as efac1 for the cell above this cell.
        cdef double srwp  # sum of all efac values.
        efac1 = self.layer_depth[l] * column_width * self.root_growth_factor[l, k]
        efacl = rgfsd * self.root_growth_factor[l, km1]
        efacr = rgfsd * self.root_growth_factor[l, kp1]
        efacu = rgfup * self.root_growth_factor[lm1, k]
        efacd = rgfdn * self.root_growth_factor[lp1, k]
        srwp = efac1 + efacl + efacr + efacu + efacd
        # If srwp is very small, all the added weight will be in the same soil soil cell, and execution of this function is ended.
        if srwp < 1e-10:
            self.actual_root_growth[l][k] = addwt
            return
        # Allocate the added dry matter to this and the adjoining soil cells in proportion to the EFAC factors.
        self.actual_root_growth[l][k] += addwt * efac1 / srwp
        self.actual_root_growth[l][km1] += addwt * efacl / srwp
        self.actual_root_growth[l][kp1] += addwt * efacr / srwp
        self.actual_root_growth[lm1][k] += addwt * efacu / srwp
        self.actual_root_growth[lp1][k] += addwt * efacd / srwp
        # If roots are growing into new soil soil cells, initialize their RootAge to 0.01.
        if self.root_age[l][km1] == 0:
            self.root_age[l][km1] = 0.01
        if self.root_age[l][kp1] == 0:
            self.root_age[l][kp1] = 0.01
        if self.root_age[lm1][k] == 0:
            self.root_age[lm1][k] = 0.01
        # If this new compartmment is in a new layer with roots, also initialize its RootColNumLeft and RootColNumRight values.
        if self.root_age[lp1][k] == 0 and efacd > 0:
            self.root_age[lp1][k] = 0.01
        # If this is in the location of the taproot, and the roots reach a new soil layer, update the taproot parameters taproot_length, self.last_layer_with_root_depth, and self.taproot_layer_number.
        if k == plant_row_column or k == plant_row_column + 1:
            if lp1 > self.taproot_layer_number and efacd > 0:
                self.taproot_length = self.last_layer_with_root_depth + 0.01
                self.last_layer_with_root_depth += self.layer_depth[lp1]
                self.taproot_layer_number = lp1

    def initialize_lateral_roots(self):
        """This function initiates lateral root growth."""
        cdef double distlr = 12  # the minimum distance, in cm, from the tip of the taproot, for a lateral root to be able to grow.
        # Loop on soil layers, from the lowest layer with roots upward:
        for i, depth in enumerate(np.cumsum(self.layer_depth[self.taproot_layer_number:-1:-1])):
            # Compute distance from tip of taproot.
            # If a layer is marked for a lateral (LateralRootFlag[l] = 1) and its distance from the tip is larger than distlr - initiate a lateral (LateralRootFlag[l] = 2).
            if self.taproot_length - self.last_layer_with_root_depth + depth > distlr and LateralRootFlag[self.taproot_layer_number - i] == 1:
                LateralRootFlag[self.taproot_layer_number - i] = 2

    def lateral_root_growth(self, NumRootAgeGroups, plant_row_column, row_space):
        # Call functions for growth of lateral roots
        self.initialize_lateral_roots()
        for l in range(self.taproot_layer_number):
            if LateralRootFlag[l] == 2:
                self.lateral_root_growth_left(l, NumRootAgeGroups, plant_row_column, row_space)
                self.lateral_root_growth_right(l, NumRootAgeGroups, plant_row_column, row_space)
    #end root

    def pre_fruiting_node_leaf_abscission(self, droplf, first_square_date, defoliate_date):
        """This function simulates the abscission of prefruiting node leaves.

        Arguments
        ---------
        droplf
            leaf age until it is abscised.
        """
        # Loop over all prefruiting nodes. If it is after first square, node age is updated here.
        for j in range(self.number_of_pre_fruiting_nodes):
            if first_square_date is not None:
                self.pre_fruiting_nodes_age[j] += self.day_inc
            # The leaf on this node is abscised if its age has reached droplf, and if there is a leaf here, and if LeafAreaIndex is not too small:
            # Update AbscisedLeafWeight, state.leaf_weight, state.petiole_weight, CumPlantNLoss.
            # Assign zero to state.pre_fruiting_leaf_area, PetioleWeightPreFru and state.leaf_weight_pre_fruiting of this leaf.
            # If a defoliation was applied.
            if self.pre_fruiting_nodes_age[j] >= droplf and self.pre_fruiting_leaf_area[j] > 0 and self.leaf_area_index > 0.1:
                self.leaf_nitrogen -= self.leaf_weight_pre_fruiting[j] * self.leaf_nitrogen_concentration
                self.leaf_weight -= self.leaf_weight_pre_fruiting[j]
                self.petiole_nitrogen -= PetioleWeightPreFru[j] * self.petiole_nitrogen_concentration
                self.petiole_weight -= PetioleWeightPreFru[j]
                self.pre_fruiting_leaf_area[j] = 0
                self.leaf_weight_pre_fruiting[j] = 0
                PetioleWeightPreFru[j] = 0

    def defoliation_leaf_abscission(self, defoliate_date):
        """Simulates leaf abscission caused by defoliants."""
        # When this is the first day of defoliation - if there are any leaves left on the prefruiting nodes, they will be shed at this stage.
        if self.date == defoliate_date:
            for j in range(self.number_of_pre_fruiting_nodes):
                if self.pre_fruiting_leaf_area[j] > 0:
                    self.pre_fruiting_leaf_area[j] = 0
                    self.leaf_nitrogen -= self.leaf_weight_pre_fruiting[j] * self.leaf_nitrogen_concentration
                    self.leaf_weight -= self.leaf_weight_pre_fruiting[j]
                    self.petiole_nitrogen -= PetioleWeightPreFru[j] * self.petiole_nitrogen_concentration
                    self.petiole_weight -= PetioleWeightPreFru[j]
                    self.leaf_weight_pre_fruiting[j] = 0
                    PetioleWeightPreFru[j] = 0
        # When this is after the first day of defoliation - count the number of existing leaves and sort them by age
        if self.date == defoliate_date:
            return
        leaves = []
        for (k, l), w in np.ndenumerate(self.main_stem_leaf_weight):
            if w > 0:
                leaves.append((self.fruiting_nodes_age[k, l, 0], k, l, 66))
            for m in range(5):
                if self.node_leaf_weight[k, l, m] > 0:
                    leaves.append((self.fruiting_nodes_age[k, l, m], k, l, m))
        # Compute the number of leaves to be shed on this day (numLeavesToShed).
        numLeavesToShed = int(len(leaves) * PercentDefoliation / 100)  # the computed number of leaves to be shed.
        # Execute leaf shedding according to leaf age.
        for leaf in sorted(leaves, reverse=True):
            if numLeavesToShed <= 0:
                break
            if numLeavesToShed > 0 and leaf[0] > 0:
                k, l, m = leaf[1:]
                if m == 66:  # main stem leaves
                    self.leaf_nitrogen -= self.main_stem_leaf_weight[k, l] * self.leaf_nitrogen_concentration
                    self.leaf_weight -= self.main_stem_leaf_weight[k, l]
                    self.petiole_nitrogen -= self.main_stem_leaf_petiole_weight[k, l] * self.petiole_nitrogen_concentration
                    self.petiole_weight -= self.main_stem_leaf_petiole_weight[k, l]
                    self.main_stem_leaf_area[k, l] = 0
                    self.main_stem_leaf_weight[k, l] = 0
                    self.main_stem_leaf_petiole_weight[k, l] = 0
                else:  # leaves on fruit nodes
                    self.leaf_nitrogen -= self.node_leaf_weight[k, l, m] * self.leaf_nitrogen_concentration
                    self.leaf_weight -= self.node_leaf_weight[k, l, m]
                    self.petiole_nitrogen -= self.node_petiole_weight[k, l, m] * self.petiole_nitrogen_concentration
                    self.petiole_weight -= self.node_petiole_weight[k, l, m]
                    self.node_leaf_area[k, l, m] = 0
                    self.node_leaf_weight[k, l, m] = 0
                    self.node_petiole_weight[k, l, m] = 0
                numLeavesToShed -= 1

    #begin phenology
    def fruiting_sites_abscission(self):
        """This function simulates the abscission of squares and bolls."""
        global NumSheddingTags
        # The following constant parameters are used:
        cdef double[9] vabsfr = [21.0, 0.42, 30.0, 0.05, 6.0, 2.25, 0.60, 5.0, 0.20]
        # Update tags for shedding: Increment NumSheddingTags by 1, and move the array members of ShedByCarbonStress, ShedByNitrogenStress, ShedByWaterStress, and AbscissionLag.
        NumSheddingTags += 1
        if NumSheddingTags > 1:
            for lt in range(NumSheddingTags - 1, 0, -1):
                ltm1 = lt - 1
                ShedByCarbonStress[lt] = ShedByCarbonStress[ltm1]
                ShedByNitrogenStress[lt] = ShedByNitrogenStress[ltm1]
                ShedByWaterStress[lt] = ShedByWaterStress[ltm1]
                AbscissionLag[lt] = AbscissionLag[ltm1]
        # Calculate shedding intensity: The shedding intensity due to stresses of this day is assigned to the first members of the arrays ShedByCarbonStress, ShedByNitrogenStress, and ShedByWaterStress.
        if self.carbon_stress < self._sim.cultivar_parameters[43]:
            ShedByCarbonStress[0] = (self._sim.cultivar_parameters[43] - self.carbon_stress) / self._sim.cultivar_parameters[43]
        else:
            ShedByCarbonStress[0] = 0
        if self.nitrogen_stress < vabsfr[1]:
            ShedByNitrogenStress[0] = (vabsfr[1] - self.nitrogen_stress) / vabsfr[1]
        else:
            ShedByNitrogenStress[0] = 0
        if self.water_stress < self._sim.cultivar_parameters[44]:
            ShedByWaterStress[0] = (self._sim.cultivar_parameters[44] - self.water_stress) / self._sim.cultivar_parameters[44]
        else:
            ShedByWaterStress[0] = 0
        # Assign 0.01 to the first member of AbscissionLag.
        AbscissionLag[0] = 0.01
        # Updating age of tags for shedding: Each member of array AbscissionLag is incremented by physiological age of today. It is further increased (i.e., shedding will occur sooner) when maximum temperatures are high.
        cdef double tmax = self._sim.meteor[self.date]["tmax"]
        for lt in range(NumSheddingTags):
            AbscissionLag[lt] += max(self.day_inc, 0.40)
            if tmax > vabsfr[2]:
                AbscissionLag[lt] += (tmax - vabsfr[2]) * vabsfr[3]
        # Assign zero to idecr, and start do loop over all days since first relevant tagging. If AbscissionLag reaches a value of vabsfr[4], calculate actual shedding of each site:
        idecr = 0  # decrease in NumSheddingTags after shedding has been executed.
        for lt in range(NumSheddingTags):
            if AbscissionLag[lt] >= vabsfr[4] or lt >= 20:
            # Start loop over all possible fruiting sites. The abscission functions will be called for sites that are squares or green bolls.
                for i, stage in np.ndenumerate(self.fruiting_nodes_stage):
                    if stage in (Stage.Square, Stage.YoungGreenBoll, Stage.GreenBoll):
                        # ratio of abscission for a fruiting site.
                        abscissionRatio = self.site_abscission_ratio(*i, lt)
                        if abscissionRatio > 0:
                            if stage == Stage.Square:
                                self.square_abscission(i, abscissionRatio)
                            else:
                                self.boll_abscission(i, abscissionRatio, self.ginning_percent if self.ginning_percent > 0 else self.fruiting_nodes_ginning_percent[i])
                # Assign zero to the array members for this day.
                ShedByCarbonStress[lt] = 0
                ShedByNitrogenStress[lt] = 0
                ShedByWaterStress[lt] = 0
                AbscissionLag[lt] = 0
                idecr += 1
        # Decrease NumSheddingTags. If plantmap adjustments are necessary for square number, or green boll number, or open boll number - call AdjustAbscission().
        NumSheddingTags -= idecr

        self.compute_site_numbers()

    def site_abscission_ratio(self, k, l, m, lt):
        """This function computes and returns the probability of abscission of a single site (k, l, m).

        Arguments
        ---------
        k, l, m
            indices defining position of this site.
        lt
            lag index for this node.
        """
        # The following constant parameters are used:
        cdef double[5] vabsc = [21.0, 2.25, 0.60, 5.0, 0.20]
        VarPar = self._sim.cultivar_parameters

        # For each site, compute the probability of its abscission (pabs) as afunction of site age, and the total shedding ratio (shedt) as a function of plant stresses that occurred when abscission was triggered.
        pabs = 0  # probability of abscission of a fruiting site.
        shedt = 0  # total shedding ratio, caused by various stresses.
        # (1) Squares (FruitingCode = 1).
        if self.fruiting_nodes_stage[k, l, m] == Stage.Square:
            if self.fruiting_nodes_age[k, l, m] < vabsc[3]:
                pabs = 0  # No abscission of very young squares (AgeOfSite less than vabsc(3))
            else:
                # square age after becoming susceptible to shedding.
                xsqage = self.fruiting_nodes_age[k, l, m] - vabsc[3]
                if xsqage >= vabsc[0]:
                    pabs = VarPar[46]  # Old squares have a constant probability of shedding.
                else:
                    # Between these limits, pabs is a function of xsqage.
                    pabs = VarPar[46] + (VarPar[45] - VarPar[46]) * pow(((vabsc[0] - xsqage) / vabsc[0]), vabsc[1])
            # Total shedding ratio (shedt) is a product of the effects of carbohydrate stress and nitrogen stress.
            shedt = 1 - (1 - ShedByCarbonStress[lt]) * (1 - ShedByNitrogenStress[lt])
        # (2) Very young bolls (FruitingCode = 7, and AgeOfBoll less than VarPar[47]).
        elif self.fruiting_nodes_stage[k, l, m] == Stage.YoungGreenBoll and self.fruiting_nodes_boll_age[k, l, m] <= VarPar[47]:
            # There is a constant probability of shedding (VarPar[48]), and shedt is a product of the effects carbohydrate, and nitrogen stresses. Note that nitrogen stress has only a partial effect in this case, as modified by vabsc[2].
            pabs = VarPar[48]
            shedt = 1 - (1 - ShedByCarbonStress[lt]) * (1 - vabsc[2] * ShedByNitrogenStress[lt])
        # (3) Medium age bolls (AgeOfBoll between VarPar[47] and VarPar[47] + VarPar[49]).
        elif VarPar[47] < self.fruiting_nodes_boll_age[k, l, m] <= (VarPar[47] + VarPar[49]):
            # pabs is linearly decreasing with age, and shedt is a product of the effects carbohydrate, nitrogen and water stresses.  Note that nitrogen stress has only a partial effect in this case, as modified by vabsc[4].
            pabs = VarPar[48] - (VarPar[48] - VarPar[50]) * (self.fruiting_nodes_boll_age[k, l, m] - VarPar[47]) / VarPar[49]
            shedt = 1 - (1 - ShedByCarbonStress[lt]) * (1 - vabsc[4] * ShedByNitrogenStress[lt]) * (1 - ShedByWaterStress[lt])
        # (4) Older bolls (AgeOfBoll between VarPar[47] + VarPar[49] and VarPar[47] + 2*VarPar[49]).
        elif (VarPar[47] + VarPar[49]) < self.fruiting_nodes_boll_age[k, l, m] <= (VarPar[47] + 2 * VarPar[49]):
            # pabs is linearly decreasing with age, and shedt is affected only by water stress.
            pabs = VarPar[50] / VarPar[49] * (VarPar[47] + 2 * VarPar[49] - self.fruiting_nodes_boll_age[k, l, m])
            shedt = ShedByWaterStress[lt]
        # (5) bolls older than VarPar[47] + 2*VarPar[49]
        elif self.fruiting_nodes_boll_age[k, l, m] > (VarPar[47] + 2 * VarPar[49]):
            pabs = 0  # no abscission
        # Actual abscission of tagged sites (abscissionRatio) is a product of pabs, shedt and DayInc for this day. It can not be greater than 1.
        return min(pabs * shedt * self.day_inc, 1)

    def square_abscission(self, index, abscissionRatio):
        """Simulates the abscission of a single square at site (k, l, m).

        Arguments
        ---------
        abscissionRatio
            ratio of abscission of a fruiting site."""
        # Compute the square weight lost by shedding (wtlos) as a proportion of SquareWeight of this site. Update state.square_nitrogen, CumPlantNLoss, SquareWeight[k][l][m], BloomWeightLoss and FruitFraction[k][l][m].
        cdef double wtlos = self.square_weights[index] * abscissionRatio  # weight lost by shedding at this site.
        self.square_nitrogen -= wtlos * self.square_nitrogen_concentration
        self.square_weights[index] -= wtlos
        self.fruiting_nodes_fraction[index] *= (1 - abscissionRatio)
        # If FruitFraction[k][l][m] is less than 0.001 make it zero, and update state.square_nitrogen, CumPlantNLoss, BloomWeightLoss, SquareWeight[k][l][m], and assign 5 to FruitingCode.
        if self.fruiting_nodes_fraction[index] <= 0.001:
            self.fruiting_nodes_fraction[index] = 0
            self.square_nitrogen -= self.square_weights[index] * self.square_nitrogen_concentration
            self.square_weights[index] = 0
            self.fruiting_nodes_stage[index] = Stage.AbscisedAsSquare

    def boll_abscission(self, index, abscissionRatio, gin1):
        """This function simulates the abscission of a single green boll at site (k, l, m). It is called from function FruitingSitesAbscission() if this site is a green boll.

        Arguments
        ---------
        abscissionRatio
            ratio of abscission of a fruiting site.
        gin1
            percent of seeds in seedcotton, used to compute lost nitrogen.
        """
        # Update state.seed_nitrogen, state.burr_nitrogen, CumPlantNLoss, state.green_bolls_burr_weight, boll_weight, burr_weight, and FruitFraction[k][l][m].
        self.seed_nitrogen -= self.fruiting_nodes_boll_weight[index] * abscissionRatio * (1 - gin1) * self.seed_nitrogen_concentration
        self.burr_nitrogen -= self.burr_weight[index] * abscissionRatio * self.burr_nitrogen_concentration
        self.green_bolls_burr_weight -= self.burr_weight[index] * abscissionRatio
        self.fruiting_nodes_boll_weight[index] *= (1 - abscissionRatio)
        self.burr_weight[index] *= (1 - abscissionRatio)
        self.fruiting_nodes_fraction[index] *= (1 - abscissionRatio)

        # If FruitFraction[k][l][m] is less than 0.001 make it zero, update state.seed_nitrogen, state.burr_nitrogen, CumPlantNLoss, state.green_bolls_burr_weight, boll_weight, burr_weight, and assign 4 to FruitingCode.

        if self.fruiting_nodes_fraction[index] <= 0.001:
            self.fruiting_nodes_stage[index] = Stage.AbscisedAsBoll
            self.seed_nitrogen -= self.fruiting_nodes_boll_weight[index] * (1 - gin1) * self.seed_nitrogen_concentration
            self.burr_nitrogen -= self.burr_weight[index] * self.burr_nitrogen_concentration
            self.fruiting_nodes_fraction[index] = 0
            self.green_bolls_burr_weight -= self.burr_weight[index]
            self.fruiting_nodes_boll_weight[index] = 0
            self.burr_weight[index] = 0

    def compute_site_numbers(self):
        """Calculates square, green boll, open boll, and abscised site numbers (NumSquares, NumGreenBolls, NumOpenBolls, and AbscisedFruitSites, respectively), as the sums of FruitFraction in all sites with appropriate FruitingCode."""
        self.number_of_green_bolls = 0
        for i, stage in np.ndenumerate(self.fruiting_nodes_stage):
            if stage in [Stage.YoungGreenBoll, Stage.GreenBoll]:
                self.number_of_green_bolls += self.fruiting_nodes_fraction[i]
    #end phenology

    #begin soil
    """\
    References for soil nitrogen routines:
    ======================================
            Godwin, D.C. and Jones, C.A. 1991. Nitrogen dynamics
    in soil - plant systems. In: J. Hanks and J.T. Ritchie (ed.)
    Modeling Plant and Soil Systems, American Society of Agronomy,
    Madison, WI, USA, pp 287-321.
            Quemada, M., and Cabrera, M.L. 1995. CERES-N model predictions
    of nitrogen mineralized from cover crop residues. Soil  Sci. Soc.
    Am. J. 59:1059-1065.
            Rolston, D.E., Sharpley, A.N., Toy, D.W., Hoffman, D.L., and
    Broadbent, F.E. 1980. Denitrification as affected by irrigation
    frequency of a field soil. EPA-600/2-80-06. U.S. Environmental
    Protection Agency, Ada, OK.
            Vigil, M.F., and Kissel, D.E. 1995. Rate of nitrogen mineralized
    from incorporated crop residues as influenced by temperarure. Soil
    Sci. Soc. Am. J. 59:1636-1644.
            Vigil, M.F., Kissel, D.E., and Smith, S.J. 1991. Field crop
    recovery and modeling of nitrogen mineralized from labeled sorghum
    residues. Soil Sci. Soc. Am. J. 55:1031-1037."""
    def soil_nitrogen(self):
        """This function computes the transformations of the nitrogen compounds in the soil."""
        # For each soil cell: call method urea_hydrolysis(), mineralize_nitrogen(), Nitrification() and denitrification().
        for l in range(40):
            for k in range(20):
                if self.soil_urea_content[l, k] > 0:
                    self.urea_hydrolysis((l, k))
                self.mineralize_nitrogen((l, k), self._sim.start_date, self._sim.row_space)
                if self.soil_ammonium_content[l, k] > 0.00001:
                    self.nitrification((l, k))
                # denitrification() is called if there are enough water and nitrates in the soil cell. cparmin is the minimum temperature C for denitrification.
                cparmin = 5
                if self.soil_nitrate_content[l, k] > 0.001 and self.soil_water_content[l, k] > self.field_capacity[l] and self.soil_temperature[l][k] >= (cparmin + 273.161):
                    self.denitrification((l, k))

    def urea_hydrolysis(self, index):
        """Computes the hydrolysis of urea to ammonium in the soil.

        It is called by function SoilNitrogen(). It calls the function SoilWaterEffect().

        The following procedure is based on the CERES routine, as documented by Godwin and Jones (1991).


        NOTE: Since COTTON2K does not require soil pH in the input, the CERES rate equation was modified as follows:

        ak is a function of soil organic matter, with two site-dependent parameters cak1 and cak2. Their values are functions of the prevalent pH:
            cak1 = -1.12 + 0.203 * pH
            cak2 = 0.524 - 0.062 * pH
        Some examples of these values:
                pH      cak1     cak2
            6.8      .2604    .1024
            7.2      .3416    .0776
            7.6      .4228    .0528
        The values for pH 7.2 are used.
        """
        soil_temperature = self.soil_temperature[index]
        l, k = index
        water_content = self.soil_water_content[l, k]
        fresh_organic_matter = self.soil_fresh_organic_matter[l, k]
        # The following constant parameters are used:
        cdef double cak1 = 0.3416
        cdef double cak2 = 0.0776  # constant parameters for computing ak from organic carbon.
        cdef double stf1 = 40.0
        cdef double stf2 = 0.20  # constant parameters for computing stf.
        cdef double swf1 = 0.20  # constant parameter for computing swf.
        # Compute the organic carbon in the soil (converted from mg / cm3 to % by weight) for the sum of stable and fresh organic matter, assuming 0.4 carbon content in soil organic matter.
        cdef double oc  # organic carbon in the soil (% by weight).
        oc = 0.4 * (fresh_organic_matter + self.soil_humus_organic_matter[l, k]) * 0.1 / self.soil_bulk_density[l]
        # Compute the potential rate of hydrolysis of urea. It is assumed that the potential rate will not be lower than ak0 = 0.25 .
        cdef double ak  # potential rate of urea hydrolysis (day-1).
        ak = max(cak1 + cak2 * oc, 0.25)
        # Compute the effect of soil moisture using function SoilWaterEffect on the rate of urea hydrolysis. The constant swf1 is added to the soil moisture function for mineralization,
        cdef double swf  # soil moisture effect on rate of urea hydrolysis.
        swf = min(max(SoilWaterEffect(water_content, self.field_capacity[l], self.thetar[l], self.soil_saturated_water_content[l], 0.5) + swf1, 0), 1)
        # Compute the effect of soil temperature. The following parameters are used for the temperature function: stf1, stf2.
        cdef double stf  # soil temperature effect on rate of urea hydrolysis.
        stf = min(max((soil_temperature - 273.161) / stf1 + stf2, 0), 1)
        # Compute the actual amount of urea hydrolized, and update soil_urea_content and soil_ammonium_content.
        cdef double hydrur  # amount of urea hydrolized, mg N cm-3 day-1.
        hydrur = ak * swf * stf * self.soil_urea_content[l, k]
        if hydrur > self.soil_urea_content[l, k]:
            hydrur = self.soil_urea_content[l, k]
        self.soil_urea_content[l, k] -= hydrur
        self.soil_ammonium_content[l, k] += hydrur

    def mineralize_nitrogen(self, index, start_date, row_space):
        """Computes the mineralization of organic nitrogen in the soil, and the immobilization of mineral nitrogen by soil microorganisms. It is called by function SoilNitrogen().

        It calls the following functions: SoilTemperatureEffect(), SoilWaterEffect().

        The procedure is based on the CERES routines, as documented by Godwin and Jones (1991).

        NOTE: CERES routines assume freshly incorporated organic matter consists of 20% carbohydrates, 70% cellulose, and 10% lignin, with maximum decay rates of 0.2, 0.05 and 0.0095, respectively. Quemada and Cabrera (1995) suggested decay rates of 0.14, 0.0034, and 0.00095 per day for carbohydrates, cellulose and lignin, respectively.

        Assuming cotton stalks consist of 20% carbohydrates, 50% cellulose and 30% lignin - the average maximum decay rate of FreshOrganicMatter decay rate = 0.03 will be used here.
        """
        l, k = index
        soil_temperature = self.soil_temperature[index]
        fresh_organic_matter = self.soil_fresh_organic_matter[index]
        nitrate_nitrogen_content = self.soil_nitrate_content[index]
        water_content = self.soil_water_content[index]
        # The following constant parameters are used:
        cdef double cnfresh = 25  # C/N ratio in fresh organic matter.
        cdef double cnhum = 10  # C/N ratio in stabilized organic matter (humus).
        cdef double cnmax = 13  # C/N ratio higher than this reduces rate of mineralization.
        cdef double cparcnrf = 0.693  # constant parameter for computing cnRatioEffect.
        cdef double cparHumusN = 0.20  # ratio of N released from fresh OM incorporated in the humus.
        cdef double cparMinNH4 = 0.00025  # mimimum NH4 N remaining after mineralization.
        cdef double decayRateFresh = 0.03  # decay rate constant for fresh organic matter.
        cdef double decayRateHumus = 0.000083  # decay rate constant for humic organic matter.
        # On the first day of simulation set initial values for N in fresh organic matter and in humus, assuming C/N ratios of cnfresh = 25 and cnhum = 10, respectively. Carbon in soil organic matter is 0.4 of its dry weight.
        if self.date <= start_date:
            FreshOrganicNitrogen[l][k] = fresh_organic_matter * 0.4 / cnfresh
            HumusNitrogen[l][k] = self.soil_humus_organic_matter[l, k] * 0.4 / cnhum
        # This function will not be executed for soil cells with no organic matter in them.
        if fresh_organic_matter <= 0 and self.soil_humus_organic_matter[l, k] <= 0:
            return

        # **  C/N ratio in soil **
        # The C/N ratio (cnRatio) is computed for the fresh organic matter and the nitrate and ammonium nitrogen in the soil. It is assumed that C/N ratios higher than cnmax reduce the rate of mineralization. Following the findings of Vigil et al. (1991) the value of cnmax is set to 13.
        cdef double cnRatio = 1000  # C/N ratio in fresh organic matter and mineral N in soil.
        cdef double cnRatioEffect = 1  # the effect of C/N ratio on rate of mineralization.
        cdef double totalSoilN  # total N in the soil cell, excluding the stable humus fraction, mg/cm3
        totalSoilN = FreshOrganicNitrogen[l][k] + nitrate_nitrogen_content + self.soil_ammonium_content[l, k]
        if totalSoilN > 0:
            cnRatio = fresh_organic_matter * 0.4 / totalSoilN
            if cnRatio >= 1000:
                cnRatioEffect = 0
            elif cnRatio > cnmax:
                cnRatioEffect = exp(-cparcnrf * (cnRatio - cnmax) / cnmax)
            else:
                cnRatioEffect = 1

        # **  Mineralization of fresh organic matter **
        # The effects of soil moisture (wf) and of soil temperature (tfac) are computed.
        cdef double wf = SoilWaterEffect(water_content, self.field_capacity[l], self.thetar[l], self.soil_saturated_water_content[l], 0.5)
        cdef double tfac = SoilTemperatureEffect(soil_temperature - 273.161)
        # The gross release of dry weight and of N from decomposition of fresh organic matter is computed.
        cdef double grossReleaseN  # gross release of N from decomposition, mg/cm3
        cdef double immobilizationRateN  # immobilization rate of N associated with decay of residues, mg/cm3 .
        if fresh_organic_matter > 0.00001:
        # The decayRateFresh constant (= 0.03) is modified by soil temperature, soil moisture, and the C/N ratio effect.
            # the actual decay rate of fresh organic matter, day-1.
            g1: float = tfac * wf * cnRatioEffect * decayRateFresh
            # the gross release of dry weight from decomposition, mg/cm3
            grossReleaseDW: float = g1 * fresh_organic_matter
            grossReleaseN = g1 * FreshOrganicNitrogen[l][k]
            # The amount of N required for microbial decay of a unit of fresh organic matter suggested in CERES is 0.02 (derived from:  C fraction in FreshOrganicMatter (=0.4) * biological efficiency of C turnover by microbes (=0.4) * N/C ratio in microbes (=0.125) ). However, Vigil et al. (1991) suggested that this value is 0.0165.
            cparnreq = 0.0165  # The amount of N required for decay of fresh organic matter
            # Substract from this the N ratio in the decaying FreshOrganicMatter, and multiply by grossReleaseDW to get the amount needed (immobilizationRateN) in mg cm-3. Negative value indicates that there is enough N for microbial decay.
            immobilizationRateN = grossReleaseDW * (cparnreq - FreshOrganicNitrogen[l][k] / fresh_organic_matter)
            # All computations assume that the amounts of soil_ammonium_content and VNO3C will each not become lower than cparMinNH4 (= 0.00025) .
            # the maximum possible value of immobilizationRateN, mg/cm3.
            rnac1: float = self.soil_ammonium_content[l, k] + nitrate_nitrogen_content - 2 * cparMinNH4
            immobilizationRateN = min(max(immobilizationRateN, 0), rnac1)
            # FreshOrganicMatter and FreshOrganicNitrogen (the N in it) are now updated.
            self.soil_fresh_organic_matter[index] -= grossReleaseDW
            FreshOrganicNitrogen[l][k] += immobilizationRateN - grossReleaseN
        else:
            grossReleaseN = 0
            immobilizationRateN = 0

        # **  Mineralization of humic organic matter **
        # The mineralization of the humic fraction (rhmin) is now computed. decayRateHumus = 0.000083 is the humic fraction decay rate (day-1). It is modified by soil temperature and soil moisture.
        # N mineralized from the stable humic fraction, mg/cm3 .
        rhmin: float = HumusNitrogen[l][k] * decayRateHumus * tfac * wf
        # rhmin is substacted from HumusNitrogen, and a corresponding amount of dry matter is substracted from soil_humus_organic_matter (assuming C/N = cnhum = 10).
        # It is assumed that 20% (=cparHumusN) of the N released from the fresh organic matter is incorporated in the humus, and a parallel amount of dry matter is also incorporated in it (assuming C/N = cnfresh = 25).
        HumusNitrogen[l][k] -= rhmin + cparHumusN * grossReleaseN
        self.soil_humus_organic_matter[l, k] -= cnhum * rhmin / 0.4 + cparHumusN * cnfresh * grossReleaseN / 0.4
        # 80% (1 - cparHumusN) of the N released from the fresh organic matter , the N released from the decay of the humus, and the immobilized N are used to compute netNReleased. Negative value of netNReleased indicates net N immobilization.
        # the net N released from all organic sources (mg/cm3).
        netNReleased: float = (1 - cparHumusN) * grossReleaseN + rhmin - immobilizationRateN
        # If the net N released is positive, it is added to the NH4 fraction.
        if netNReleased > 0:
            self.soil_ammonium_content[l, k] += netNReleased
        # If net N released is negative (net immobilization), the NH4 fraction is reduced, but at least 0.25 ppm (=cparMinNH4 in mg cm-3) of NH4 N should remain. A matching amount of N is added to the organic N fraction.
        # MineralizedOrganicN, the accumulated nitrogen released by mineralization in the slab is updated.
        else:
            addvnc: float = 0  # immobilised N added to the organic fraction.
            nnom1: float = 0  # temporary storage of netNReleased (if N is also immobilized from NO3).
            if self.soil_ammonium_content[l, k] > cparMinNH4:
                if abs(netNReleased) < (self.soil_ammonium_content[l, k] - cparMinNH4):
                    addvnc = -netNReleased
                else:
                    addvnc = self.soil_ammonium_content[l, k] - cparMinNH4
                self.soil_ammonium_content[l, k] -= addvnc
                FreshOrganicNitrogen[l][k] += addvnc
                nnom1 = netNReleased + addvnc
        # If immobilization is larger than the use of NH4 nitrogen, the NO3 fraction is reduced in a similar procedure.
            if nnom1 < 0 and nitrate_nitrogen_content > cparMinNH4:
                if abs(nnom1) < (nitrate_nitrogen_content - cparMinNH4):
                    addvnc = -nnom1
                else:
                    addvnc = nitrate_nitrogen_content - cparMinNH4
                self.soil_nitrate_content[index] -= addvnc
                FreshOrganicNitrogen[l][k] += addvnc

    def denitrification(self, index):
        """Computes the denitrification of nitrate N in the soil.

        The procedure is based on the CERES routine, as documented by Godwin and Jones (1991).
        """
        soil_temperature = self.soil_temperature[index]
        l, k = index
        # The following constant parameters are used:
        cpar01: float = 24.5
        cpar02: float = 3.1
        cpardenit: float = 0.00006
        cparft: float = 0.046
        cparhum: float = 0.58
        vno3min: float = 0.00025

        # soil carbon content, mg/cm3. soilc is calculated as 0.58 (cparhum) of the stable humic fraction (following CERES), and cw is estimated following Rolston et al. (1980).
        soilc: float = cparhum * self.soil_humus_organic_matter[l, k]
        # water soluble carbon content of soil, ppm.
        cw: float = cpar01 + cpar02 * soilc
        # The effects of soil moisture (fw) and soil temperature (ft) are computed as 0 to 1 factors.
        # effect of soil moisture on denitrification rate.
        fw: float = max((self.soil_water_content[l, k] - self.field_capacity[l]) / (self.soil_saturated_water_content[l] - self.field_capacity[l]), 0)
        # effect of soil temperature on denitrification rate.
        ft: float = min(0.1 * exp(cparft * (soil_temperature - 273.161)), 1)
        # The actual rate of denitrification is calculated. The equation is modified from CERES to units of mg/cm3/day.
        # actual rate of denitrification, mg N per cm3 of soil per day.
        dnrate: float = min(max(cpardenit * cw * self.soil_nitrate_content[index] * fw * ft, 0), self.soil_nitrate_content[index] - vno3min)
        # Update VolNo3NContent, and add the amount of nitrogen lost to SoilNitrogenLoss.
        self.soil_nitrate_content[index] -= dnrate

    def nitrification(self, index):
        """This function computes the transformation of soil ammonia nitrogen to nitrate.
        """
        l, k = index
        soil_temperature = self.soil_temperature[index]
        # The following constant parameters are used:
        cpardepth: float = 0.45
        cparnit1: float = 24.635
        cparnit2: float = 8227
        cparsanc: float = 204  # this constant parameter is modified from kg/ha units in CERES to mg/cm3 units of soil_ammonium_content (assuming 15 cm layers)
        sanc: float  # effect of NH4 N in the soil on nitrification rate (0 to 1).
        if self.soil_ammonium_content[l, k] < 0.1:
            sanc = 1 - exp(-cparsanc * self.soil_ammonium_content[l, k])
        else:
            sanc = 1
        # The rate of nitrification, con1, is a function of soil temperature. It is slightly modified from GOSSYM. it is transformed from immediate rate to a daily time step ratenit.
        # The rate is modified by soil depth, assuming that for an increment of 30 cm depth, the rate is decreased by 55% (multiply by a power of cpardepth). It is also multiplied by the environmental limiting factors (sanc, SoilWaterEffect) to get the actual rate of nitrification.
        # The maximum rate is assumed not higher than 10%.
        # rate of nitrification as a function of temperature.
        con1: float = exp(cparnit1 - cparnit2 / soil_temperature)
        ratenit: float = 1 - exp(-con1)  # actual rate of nitrification (day-1).
        # effect of soil depth on nitrification rate.
        tff: float = max((self.layer_depth_cumsum[l] - 30) / 30, 0)
        # Add the effects of NH4 in soil, soil water content, and depth of soil layer.
        ratenit *= sanc * SoilWaterEffect(self.soil_water_content[l, k], self.field_capacity[l], self.thetar[l], self.soil_saturated_water_content[l], 1) * pow(cpardepth, tff)
        ratenit = min(max(ratenit, 0), 0.10)
        # Compute the actual amount of N nitrified, and update soil_ammonium_content and VolNo3NContent.
        # actual nitrification (mg n cm-3 day-1).
        dnit: float = ratenit * self.soil_ammonium_content[l, k]
        self.soil_ammonium_content[l, k] -= dnit
        self.soil_nitrate_content[index] += dnit

    cdef public long numiter  # counter used for water_flux() calls.

    def capillary_flow(self, noitr):
        """This function computes the capillary water flow between soil cells. It is called by SoilProcedures(), noitr times per day. The number of iterations (noitr) has been computed in SoilProcedures() as a function of the amount of water applied. It is executed only once per day if no water is applied by rain or irrigation."""
        cdef double wk1[40]  # dummy array for passing values of array wk.
        cdef double _dl[40]
        # Set initial values in first day.
        if self.date == self._sim.start_date:
            self.numiter = 0
            for l in range(40):
                wk1[l] = 0
        # Increase the counter numiter, and compute the updated values of SoilPsi in each soil cell by calling functions psiq() and PsiOsmotic().
        self.numiter += 1
        for l in range(40):
            for k in range(20):
                self.soil_psi[l, k] = psiq(self.soil_water_content[l, k], self.thad[l], self.soil_saturated_water_content[l], self.alpha[l], self.beta[l]) - PsiOsmotic(self.soil_water_content[l, k], self.soil_saturated_water_content[l], ElCondSatSoilToday)

        cdef double q01[40]  # one dimensional array of a layer or a column of previous values of cell.water_content.
        cdef double q1[40]  # one dimensional array of a layer or a column of cell.water_content.
        cdef double psi1[40]  # one dimensional array of a layer or a column of SoilPsi.
        cdef double nit[40]  # one dimensional array of a layer or a column of VolNo3NContent.
        cdef double nur[40]  # one dimensional array of a layer or a column of soil_urea_content.
        # direction indicator: iv = 1 for vertical flow in each column; iv = 0 for horizontal flow in each layer.
        # VERTICAL FLOW in each column. the direction indicator iv is set to 1.
        iv: int = 1
        # Loop over all columns. Temporary one-dimensional arrays are defined for each column: assign the cell.water_content[] values to temporary one-dimensional arrays q1 and q01. Assign SoilPsi, VolNo3NContent and soil_urea_content values to arrays psi1, nit and nur, respectively.
        for k in range(20):
            for l in range(40):
                q1[l] = self.soil_water_content[l, k]
                q01[l] = self.soil_water_content[l, k]
                psi1[l] = self.soil_psi[l, k] + PsiOsmotic(self.soil_water_content[l, k], self.soil_saturated_water_content[l], ElCondSatSoilToday)
                nit[l] = self.soil_nitrate_content[l, k]
                nur[l] = self.soil_urea_content[l, k]
                _dl[l] = self.layer_depth[l]
            # Call the following functions: water_flux() calculates the water flow caused by potential gradients; NitrogenFlow() computes the movement of nitrates caused by the flow of water.
            self.water_flux(q1, psi1, _dl, self.thad, self.soil_saturated_water_content, self.pore_space, 40, iv, 0, self.numiter, noitr)
            NitrogenFlow(nl, q01, q1, _dl, nit, nur)
            # Reassign the updated values of q1, nit, nur and psi1 back to cell.water_content, VolNo3NContent, soil_urea_content and SoilPsi.
            for l in range(40):
                self.soil_water_content[l, k] = q1[l]
                self.soil_nitrate_content[l, k] = nit[l]
                self.soil_urea_content[l, k] = nur[l]
                self.soil_psi[l, k] = psi1[l] - PsiOsmotic(self.soil_water_content[l, k], self.soil_saturated_water_content[l], ElCondSatSoilToday)
        cdef numpy.ndarray pp1  # one dimensional array of a layer or a column of PP.
        cdef numpy.ndarray qr1  # one dimensional array of a layer or a column of THAD.
        cdef numpy.ndarray qs1  # one dimensional array of a layer or a column of THTS.

        # HORIZONTAL FLUX in each layer. The direction indicator iv is set to 0.
        iv = 0
        # Loop over all layers. Define the horizon number j for this layer. Temporary one-dimensional arrays are defined for each layer: assign the cell.water_content values to  q1 and q01. Assign SoilPsi, VolNo3NContent, soil_urea_content, thad and soil_saturated_water_content values of the soil cells to arrays psi1, nit, nur, qr1 and qs1, respectively.
        for l in range(40):
            qr1 = np.repeat(self.thad[l], 20)
            pp1 = np.repeat(self.pore_space[l], 20)
            qs1 = np.repeat(self.soil_saturated_water_content[l], 20)
            for k in range(20):
                q1[k] = self.soil_water_content[l, k]
                q01[k] = self.soil_water_content[l, k]
                psi1[k] = self.soil_psi[l][k] + PsiOsmotic(self.soil_water_content[l, k], self.soil_saturated_water_content[l], ElCondSatSoilToday)
                qs1[k] = self.soil_saturated_water_content[l]
                nit[k] = self.soil_nitrate_content[l, k]
                nur[k] = self.soil_urea_content[l, k]
                wk1[k] = self._sim.column_width[k]
            # Call subroutines water_flux(), and NitrogenFlow() to compute water nitrate and urea transport in the layer.
            self.water_flux(q1, psi1, wk1, qr1, qs1, pp1, nk, iv, l, self.numiter, noitr)
            NitrogenFlow(nk, q01, q1, wk1, nit, nur)
            # Reassign the updated values of q1, nit, nur and psi1 back to cell.water_content, VolNo3NContent, soil_urea_content and SoilPsi.
            for k in range(20):
                self.soil_water_content[l, k] = q1[k]
                self.soil_psi[l][k] = psi1[k] - PsiOsmotic(self.soil_water_content[l, k], self.soil_saturated_water_content[l], ElCondSatSoilToday)
                self.soil_nitrate_content[l, k] = nit[k]
                self.soil_urea_content[l, k] = nur[k]
        # Call drain to move excess water down in the column and compute drainage out of the column. Update cumulative drainage.
        cdef double WaterDrainedOut = 0  # water drained out of the slab, mm.
        WaterDrainedOut += self.drain()
        # Compute the soil water potential for all soil cells.
        for l in range(40):
            for k in range(20):
                self.soil_psi[l][k] = psiq(self.soil_water_content[l, k], self.thad[l], self.soil_saturated_water_content[l], self.alpha[l], self.beta[l]) - PsiOsmotic(self.soil_water_content[l, k], self.soil_saturated_water_content[l], ElCondSatSoilToday)

    def drain(self) -> float:
        """the gravity flow of water in the slab, and returns the drainage of water out of the slab. It is called from capillary_flow()."""
        nlx: int = 40  # last soil layer for computing drainage.
        cdef double oldvh2oc[20]  # stores previous values of cell.water_content.
        cdef double nitconc  # nitrate N concentration in the soil solution.
        cdef double nurconc  # urea N concentration in the soil solution.
        # The following is executed if this is not the bottom layer.
        for l in range(nlx - 1):
            layer_depth_ratio: float = self.layer_depth[l] / self.layer_depth[l + 1]
            # Compute the average water content (avwl) of layer l. Store the water content in array oldvh2oc.
            avwl: float = 0  # average water content in a soil layer
            for k in range(20):
                avwl += self.soil_water_content[l, k] * self._sim.column_width[k] / self._sim.row_space
                oldvh2oc[k] = self.soil_water_content[l, k]
            # Upper limit of water content in free drainage..
            uplimit: float = self.max_water_capacity[l]

            # Check if the average water content exceeds uplimit for this layer, and if it does, compute amount (wmov) to be moved to the next layer from each cell.
            wmov: float  # amount of water moving out of a cell.
            if avwl > uplimit:
                wmov = avwl - uplimit
                wmov = wmov * layer_depth_ratio
                for k in range(20):
                    # Water content of all soil cells in this layer will be uplimit. the amount (qmv) to be added to each cell of the next layer is computed (corrected for non uniform column widths). The water content in the next layer is computed.
                    self.soil_water_content[l, k] = uplimit
                    self.soil_water_content[l + 1, k] += wmov * self._sim.column_width[k] * nk / self._sim.row_space
                    # The concentrations of nitrate and urea N in the soil solution are computed and their amounts in this layer and in the next one are updated.
                    qvout: float = (oldvh2oc[k] - uplimit)  # amount of water moving out of a cell.
                    if qvout > 0:
                        nitconc = self.soil_nitrate_content[l, k] / oldvh2oc[k]
                        if nitconc < 1.e-30:
                            nitconc = 0
                        nurconc = self.soil_urea_content[l, k] / oldvh2oc[k]
                        if nurconc < 1.e-30:
                            nurconc = 0
                        self.soil_nitrate_content[l, k] = self.soil_water_content[l, k] * nitconc
                        self.soil_urea_content[l, k] = self.soil_water_content[l, k] * nurconc
                        # Only a part ( NO3FlowFraction ) of N is moved with water draining.
                        vno3mov: float = qvout * nitconc  # amount of nitrate N moving out of a cell.
                        self.soil_nitrate_content[l + 1, k] += self.soil_nitrate_flow_fraction[l] * vno3mov * layer_depth_ratio
                        self.soil_nitrate_content[l, k] += (1 - self.soil_nitrate_flow_fraction[l]) * vno3mov
                        vnurmov: float = qvout * nurconc  # amount of urea N moving out of a cell.
                        self.soil_urea_content[l + 1, k] += self.soil_nitrate_flow_fraction[l] * vnurmov * layer_depth_ratio
                        self.soil_urea_content[l, k] += (1 - self.soil_nitrate_flow_fraction[l]) * vnurmov
            else:  # If the average water content is not higher than uplimit, start another loop over columns.
                for k in range(20):
                    # Check each soil cell if water content exceeds uplimit,
                    if self.soil_water_content[l, k] > uplimit:
                        wmov = self.soil_water_content[l, k] - uplimit
                        self.soil_water_content[l, k] = uplimit
                        self.soil_water_content[l + 1, k] += wmov * layer_depth_ratio
                        nitconc = self.soil_nitrate_content[l, k] / oldvh2oc[k]
                        if nitconc < 1.e-30:
                            nitconc = 0
                        nurconc = self.soil_urea_content[l, k] / oldvh2oc[k]
                        if nurconc < 1.e-30:
                            nurconc = 0
                        self.soil_nitrate_content[l, k] = self.soil_water_content[l, k] * nitconc
                        self.soil_urea_content[l, k] = self.soil_water_content[l, k] * nurconc

                        self.soil_nitrate_content[l + 1, k] += self.soil_nitrate_flow_fraction[l] * wmov * nitconc * layer_depth_ratio
                        self.soil_urea_content[l + 1, k] += self.soil_nitrate_flow_fraction[l] * wmov * nurconc * layer_depth_ratio
                        self.soil_nitrate_content[l, k] += (1 - self.soil_nitrate_flow_fraction[l]) * wmov * nitconc
                        self.soil_urea_content[l, k] += (1 - self.soil_nitrate_flow_fraction[l]) * wmov * nurconc
        # For the lowermost soil layer, loop over columns:
        # It is assumed that the maximum amount of water held at the lowest soil layer (nlx-1) of the slab is equal to self.field_capacity. If water content exceeds max_water_capacity, compute the water drained out (Drainage), update water, nitrate and urea, compute nitrogen lost by drainage, and add it to the cumulative N loss SoilNitrogenLoss.
        Drainage: float = 0  # drainage of water out of the slab, cm3 (return value)
        for k in range(20):
            if self.soil_water_content[nlx - 1, k] > self.max_water_capacity[nlx - 1]:
                Drainage += (self.soil_water_content[nlx - 1, k] - self.max_water_capacity[nlx - 1]) * self.layer_depth[nlx - 1] * self._sim.column_width[k]
                nitconc = self.soil_nitrate_content[nlx - 1, k] / oldvh2oc[k]
                if nitconc < 1.e-30:
                    nitconc = 0
                nurconc = self.soil_urea_content[nlx - 1, k] / oldvh2oc[k]
                if nurconc < 1.e-30:
                    nurconc = 0
                # intermediate variable for computing N loss.
                saven: float = (self.soil_nitrate_content[nlx - 1, k] + self.soil_urea_content[nlx - 1, k]) * self.layer_depth[nlx - 1] * self._sim.column_width[k]
                self.soil_water_content[nlx - 1, k] = self.max_water_capacity[nlx - 1]
                self.soil_nitrate_content[nlx - 1, k] = nitconc * self.max_water_capacity[nlx - 1]
                self.soil_urea_content[nlx - 1, k] = nurconc * self.max_water_capacity[nlx - 1]
        return Drainage

    def drip_flow(self, double Drip, double row_space):
        """Computes the water redistribution in the soil after irrigation by a drip
        system. It also computes the resulting redistribution of nitrate and urea N.

        Arguments
        ---------
        Drip
            amount of irrigation applied by the drip method, mm.
        """
        cdef double dripw[40]  # amount of water applied, or going from one ring of
        # soil cells to the next one, cm3. (array)
        cdef double dripn[40]  # amount of nitrate N applied, or going from one ring of soil
        # soil cells to the next one, mg. (array)
        cdef double dripu[40]  # amount of urea N applied, or going from one ring of soil
        # soil cells to the next one, mg. (array)
        for i in range(40):
            dripw[i] = 0
            dripn[i] = 0
            dripu[i] = 0
        # Incoming flow of water (Drip, in mm) is converted to dripw(0), in cm3 per slab.
        dripw[0] = Drip * row_space * 0.10
        # Wetting the cell in which the emitter is located.
        cdef double h2odef  # the difference between the maximum water capacity (at a water content of uplimit) of this ring of soil cell, and the actual water content, cm3.
        cdef int l0 = self.drip_y  #  layer where the drip emitter is situated
        cdef int k0 = self.drip_x  #  column where the drip emitter is situated
        # SoilCell &soil_cell = soil_cells[l0][k0]
        # It is assumed that wetting cannot exceed max_water_capacity of this cell. Compute h2odef, the amount of water needed to saturate this cell.
        h2odef = (self.max_water_capacity[l0] - self.soil_water_content[l0, k0]) * self.layer_depth[l0] * self._sim.column_width[k0]
        # If maximum water capacity is not exceeded - update cell.water_content of this cell and exit the function.
        if dripw[0] <= h2odef:
            self.soil_water_content[l0, k0] += dripw[0] / (self.layer_depth[l0] * self._sim.column_width[k0])
            return
        # If maximum water capacity is exceeded - calculate the excess of water flowing out of this cell (in cm3 per slab) as dripw[1]. The next ring of cells (kr=1) will receive it as incoming water flow.
        dripw[1] = dripw[0] - h2odef
        # Compute the movement of nitrate N to the next ring
        cdef double cnw = 0  #  concentration of nitrate N in the outflowing water
        if self.soil_nitrate_content[l0, k0] > 1.e-30:
            cnw = self.soil_nitrate_content[l0, k0] / (
                        self.soil_water_content[l0, k0] + dripw[0] / (self.layer_depth[l0] * self._sim.column_width[k0]))
            # cnw is multiplied by dripw[1] to get dripn[1], the amount of nitrate N going out to the next ring of cells. It is assumed, however, that not more than a proportion (NO3FlowFraction) of the nitrate N in this cell can be removed in one iteration.
            if (cnw * self.max_water_capacity[l0]) < (self.soil_nitrate_flow_fraction[l0] * self.soil_nitrate_content[l0, k0]):
                dripn[1] = self.soil_nitrate_flow_fraction[l0] * self.soil_nitrate_content[l0, k0] * self.layer_depth[
                    l0] * self._sim.column_width[k0]
                self.soil_nitrate_content[l0, k0] = (1 - self.soil_nitrate_flow_fraction[l0]) * self.soil_nitrate_content[l0, k0]
            else:
                dripn[1] = dripw[1] * cnw
                self.soil_nitrate_content[l0, k0] = self.max_water_capacity[l0] * cnw

        # The movement of urea N to the next ring is computed similarly.
        cdef double cuw = 0  # concentration of urea N in the outflowing water
        if self.soil_urea_content[l0][k0] > 1.e-30:
            cuw = self.soil_urea_content[l0][k0] / (
                        self.soil_water_content[l0, k0] + dripw[0] / (self.layer_depth[l0] * self._sim.column_width[k0]))
            if (cuw * self.max_water_capacity[l0]) < (self.soil_nitrate_flow_fraction[l0] * self.soil_urea_content[l0][k0]):
                dripu[1] = self.soil_nitrate_flow_fraction[l0] * self.soil_urea_content[l0][k0] * self.layer_depth[l0] * self._sim.column_width[k0]
                self.soil_urea_content[l0][k0] = (1 - self.soil_nitrate_flow_fraction[l0]) * self.soil_urea_content[l0][k0]
            else:
                dripu[1] = dripw[1] * cuw
                self.soil_urea_content[l0][k0] = self.max_water_capacity[l0] * cuw
        cdef double defcit[40][20]  # array of the difference between water capacity and actual water content in each cell of the ring
        # Set cell.water_content of the cell in which the drip is located to max_water_capacity.
        self.soil_water_content[l0, k0] = self.max_water_capacity[l0]
        # Loop of concentric rings of cells, starting from ring 1.
        # Assign zero to the sums sv, st, sn, sn1, su and su1.
        for kr in range(1, 40):
            uplimit: float  # upper limit of soil water content in a soil cell
            sv = 0  # sum of actual water content in a ring of cells, cm3
            st = 0  # sum of total water capacity in a ring of cells, cm3
            sn = 0  # sum of nitrate N content in a ring of cells, mg.
            sn1 = 0  # sum of movable nitrate N content in a ring of cells, mg
            su = 0  # sum of urea N content in a ring of cells, mg
            su1 = 0  # sum of movable urea N content in a ring of cells, mg
            radius = 6 * kr  # radius (cm) of the wetting ring
            dist: float  # distance (cm) of a cell center from drip location
            # Loop over all soil cells
            for l in range(1, 40):
                # Upper limit of water content is the self.pore_space volume in layers below the water table, max_water_capacity in other layers.
                uplimit = self.max_water_capacity[l]
                for k in range(20):
                    # Compute the sums sv, st, sn, sn1, su and su1 within the radius limits of this ring. The array defcit is the sum of difference between uplimit and cell.water_content of each cell.
                    dist = self.cell_distance(l, k, l0, k0)
                    if radius >= dist > (radius - 6):
                        sv += self.soil_water_content[l, k] * self.layer_depth[l] * self._sim.column_width[k]
                        st += uplimit * self.layer_depth[l] * self._sim.column_width[k]
                        sn += self.soil_nitrate_content[l, k] * self.layer_depth[l] * self._sim.column_width[k]
                        sn1 += self.soil_nitrate_content[l, k] * self.layer_depth[l] * self._sim.column_width[k] * \
                               self.soil_nitrate_flow_fraction[l]
                        su += self.soil_urea_content[l, k] * self.layer_depth[l] * self._sim.column_width[k]
                        su1 += self.soil_urea_content[l, k] * self.layer_depth[l] * self._sim.column_width[k] * self.soil_nitrate_flow_fraction[l]
                        defcit[l][k] = uplimit - self.soil_water_content[l, k]
                    else:
                        defcit[l][k] = 0
            # Compute the amount of water needed to saturate all the cells in this ring (h2odef).
            h2odef = st - sv
            # Test if the amount of incoming flow, dripw(kr), is greater than  h2odef.
            if dripw[kr] <= h2odef:
                # In this case, this will be the last wetted ring.
                # Update cell.water_content in this ring, by wetting each cell in proportion to its defcit. Update VolNo3NContent and soil_urea_content of the cells in this ring by the same proportion. this is executed for all the cells in the ring.
                for l in range(1, 40):
                    for k in range(20):
                        dist = self.cell_distance(l, k, l0, k0)
                        if radius >= dist > (radius - 6):
                            self.soil_water_content[l, k] += dripw[kr] * defcit[l][k] / h2odef
                            self.soil_nitrate_content[l, k] += dripn[kr] * defcit[l][k] / h2odef
                            self.soil_urea_content[l, k] += dripu[kr] * defcit[l][k] / h2odef
                return
            # If dripw(kr) is greater than h2odef, calculate cnw and cuw as the concentration of nitrate and urea N in the total water of this ring after receiving the incoming water and nitrogen.
            cnw = (sn + dripn[kr]) / (sv + dripw[kr])
            cuw = (su + dripu[kr]) / (sv + dripw[kr])
            drwout = dripw[kr] - h2odef  # the amount of water going out of a ring, cm3.
            # Compute the nitrate and urea N going out of this ring, and their amount lost from this ring. It is assumed that not more than a certain part of the total nitrate or urea N (previously computed as sn1 an su1) can be lost from a ring in one iteration. drnout and xnloss are adjusted accordingly. druout and xuloss are computed similarly for urea N.
            drnout = drwout * cnw  # the amount of nitrate N going out of a ring, mg
            xnloss = 0  # the amount of nitrate N lost from a ring, mg
            if drnout > dripn[kr]:
                xnloss = drnout - dripn[kr]
                if xnloss > sn1:
                    xnloss = sn1
                    drnout = dripn[kr] + xnloss
            druout = drwout * cuw  # the amount of urea N going out of a ring, mg
            xuloss = 0  # the amount of urea N lost from a ring, mg
            if druout > dripu[kr]:
                xuloss = druout - dripu[kr]
                if xuloss > su1:
                    xuloss = su1
                    druout = dripu[kr] + xuloss
            # For all the cells in the ring, as in the 1st cell, saturate cell.water_content to uplimit, and update soil_nitrate_content and soil_urea_content.
            for l in range(1, 40):
                uplimit = self.max_water_capacity[l]

                for k in range(20):
                    dist = self.cell_distance(l, k, l0, k0)
                    if radius >= dist > (radius - 6):
                        self.soil_water_content[l, k] = uplimit
                        if xnloss <= 0:
                            self.soil_nitrate_content[l, k] = uplimit * cnw
                        else:
                            self.soil_nitrate_content[l, k] *= (1. - xnloss / sn)
                        if xuloss <= 0:
                            self.soil_urea_content[l, k] = uplimit * cuw
                        else:
                            self.soil_urea_content[l, k] *= (1. - xuloss / su)
            # The outflow of water, nitrate and urea from this ring will be the inflow into the next ring.
            if kr < (nl - l0 - 1) and kr < maxl - 1:
                dripw[kr + 1] = drwout
                dripn[kr + 1] = drnout
                dripu[kr + 1] = druout
            else:
                # If this is the last ring, the outflowing water will be added to the drainage, CumWaterDrained, the outflowing nitrogen to SoilNitrogenLoss.
                return
            # Repeat all these procedures for the next ring.

    cdef water_flux(self, double q1[], double psi1[], double dd[], numpy.ndarray qr1, numpy.ndarray qs1, numpy.ndarray pp1, int nn, int iv, int ll, long numiter, int noitr):
        """Computes the movement of water in the soil, caused by potential differences between cells in a soil column or in a soil layer. It is called by function CapillaryFlow(). It calls functions WaterBalance(), psiq(), qpsi() and wcond().

        Arguments
        ---------
        q1
            array of volumetric water content, v/v.
        psi1
            array of matric soil water potential, bars.
        dd1
            array of widths of soil cells in the direction of flow, cm.
        qr1
            array of residual volumetric water content.
        qs1
            array of saturated volumetric water content.
        pp1
            array of pore space v/v.
        nn
            number of cells in the array.
        iv
            indicator if flow direction, iv = 1 for vertical iv = 0 for horizontal.
        ll
            layer number if the flow is horizontal.
        numiter
            counter for the number of iterations.
        """
        delt: float = 1 / noitr  # the time step of this iteration (fraction of day)
        cdef double cond[40]  # values of hydraulic conductivity
        cdef double kx[40]  # non-dimensional conductivity to the lower layer or to the column on the right
        cdef double ky[40]  # non-dimensional conductivity to the upper layer or to the column on the left
        # Loop over all soil cells. if this is a vertical flow, define the profile index j for each soil cell. compute the hydraulic conductivity of each soil cell, using the function wcond(). Zero the arrays kx and ky.
        j = ll  # for horizontal flow (iv = 0)
        for i in range(nn):
            if iv == 1:
                j = i  # for vertical flow
            cond[i] = wcond(q1[i], qr1[i], qs1[i], self.beta[j], self.soil_saturated_hydraulic_conductivity[j], pp1[i])
            kx[i] = 0
            ky[i] = 0

        # Loop from the second soil cell. compute the array dy (distances between the midpoints of adjacent cells).
        # Compute the average conductivity avcond[i] between soil cells i and (i-1). for low values of conductivity in both cells,(( an arithmetic mean is computed)). for higher values the harmonic mean is used, but if in one of the cells the conductivity is too low (less than a minimum value of condmin ), replace it with condmin.

        cdef double dy[40]  # the distance between the midpoint of a layer (or a column) and the midpoint
        # of the layer above it (or the column to the left of it)
        cdef double avcond[40]  # average hydraulic conductivity of two adjacent soil cells
        condmin = 0.000006  # minimum value of conductivity, used for computing averages
        for i in range(1, nn):
            dy[i] = 0.5 * (dd[i - 1] + dd[i])
            if cond[i - 1] <= condmin and cond[i] <= condmin:
                avcond[i] = condmin
            elif cond[i - 1] <= condmin < cond[i]:
                avcond[i] = 2 * condmin * cond[i] / (condmin + cond[i])
            elif cond[i] <= condmin < cond[i - 1]:
                avcond[i] = 2 * condmin * cond[i - 1] / (condmin + cond[i - 1])
            else:
                avcond[i] = 2 * cond[i - 1] * cond[i] / (cond[i - 1] + cond[i])
        # The numerical solution of the flow equation is a combination of the implicit method (weighted by ratio_implicit) and the explicit method (weighted by 1-ratio_implicit).
        # Compute the explicit part of the solution, weighted by (1-ratio_implicit). store water content values, before changing them, in array qx.
        cdef double qx[40]  # previous value of q1.
        cdef double addq[40]  # water added to qx
        cdef double sumaddq = 0  # sum of addq
        for i in range(nn):
            qx[i] = q1[i]
        # Loop from the second to the last but one soil cells.
        for i in range(1, nn - 1):
            # Compute the difference in soil water potential between adjacent cells (deltpsi). This difference is not allowed to be greater than 1000 bars, in order to prevent computational overflow in cells with low water content.
            deltpsi: float = psi1[i - 1] - psi1[i]  # difference of soil water potentials (in bars)
            # between adjacent soil soil cells
            if deltpsi > 1000:
                deltpsi = 1000
            if deltpsi < -1000:
                deltpsi = -1000
            # If this is a vertical flux, add the gravity component of water potential.
            if iv == 1:
                deltpsi += 0.001 * dy[i]
            # Compute dumm1 (the hydraulic conductivity redimensioned to cm), and check that it will not exceed max_conductivity multiplied by the distance between soil cells, in order to prevent overflow errors.
            # redimensioned hydraulic conductivity components between adjacent cells.
            dumm1 = min(1000 * avcond[i] * delt / dy[i], self.max_conductivity * dy[i])
            # Water entering soil cell i is now computed, weighted by (1 - ratio_implicit). It is not allowed to be greater than 25% of the difference between the cells.
            # Compute water movement from soil cell i-1 to i:
            # water added to cell i from cell (i-1)
            dqq1: float = (1 - self.ratio_implicit) * deltpsi * dumm1
            # difference of soil water content (v/v) between adjacent cells.
            deltq: float = qx[i - 1] - qx[i]
            if abs(dqq1) > abs(0.25 * deltq):
                if deltq > 0 > dqq1:
                    dqq1 = 0
                elif deltq < 0 < dqq1:
                    dqq1 = 0
                else:
                    dqq1 = 0.25 * deltq
            # This is now repeated for water movement from i+1 to i.
            deltpsi = psi1[i + 1] - psi1[i]
            deltq = qx[i + 1] - qx[i]
            if deltpsi > 1000:
                deltpsi = 1000
            if deltpsi < -1000:
                deltpsi = -1000
            if iv == 1:
                deltpsi -= 0.001 * dy[i + 1]
            dumm1 = min(1000 * avcond[i + 1] * delt / dy[i + 1], self.max_conductivity * dy[i + 1])
            dqq2: float = (1 - self.ratio_implicit) * deltpsi * dumm1  # water added to cell i from cell (i+1)
            if abs(dqq2) > abs(0.25 * deltq):
                if deltq > 0 > dqq2:
                    dqq2 = 0
                elif deltq < 0 < dqq2:
                    dqq2 = 0
                else:
                    dqq2 = 0.25 * deltq
            addq[i] = (dqq1 + dqq2) / dd[i]
            sumaddq += dqq1 + dqq2
            # Water content of the first and last soil cells is updated to account for flow to or from their adjacent soil cells.
            if i == 1:
                addq[0] = -dqq1 / dd[0]
                sumaddq -= dqq1
            if i == nn - 2:
                addq[nn - 1] = -dqq2 / dd[nn - 1]
                sumaddq -= dqq2
        # Water content q1[i] and soil water potential psi1[i] are updated.
        for i in range(nn):
            q1[i] = qx[i] + addq[i]
            if iv == 1:
                j = i
            psi1[i] = psiq(q1[i], qr1[i], qs1[i], self.alpha[j], self.beta[j])
        # Compute the implicit part of the solution, weighted by ratio_implicit, starting loop from the second cell.
        for i in range(1, nn):
            # Mean conductivity (avcond) between adjacent cells is made "dimensionless" (ky) by multiplying it by the time step (delt)and dividing it by cell length (dd) and by dy. It is also multiplied by 1000 for converting the potential differences from bars to cm.
            ky[i] = min(1000 * avcond[i] * delt / (dy[i] * dd[i]), self.max_conductivity)
            # Very low values of ky are converted to zero, to prevent underflow computer errors, and very high values are converted to maximum limit (max_conductivity), to prevent overflow errors.
            if ky[i] < 0.0000001:
                ky[i] = 0
        # ky[i] is the conductivity between soil cells i and i-1, whereas kx[i] is between i and i+1. Another loop, until the last but one soil cell, computes kx in a similar manner.
        for i in range(nn - 1):
            kx[i] = min(1000 * avcond[i + 1] * delt / (dy[i + 1] * dd[i]), self.max_conductivity)
            if kx[i] < 0.0000001:
                kx[i] = 0
        # Arrays used for the implicit numeric solution:
        cdef double a1[40], b1[40], cau[40], cc1[40], d1[40], dau[40]
        for i in range(nn):
            # Arrays a1, b1, and cc1 are computed for the implicit part of the solution, weighted by ratio_implicit.
            a1[i] = -kx[i] * self.ratio_implicit
            b1[i] = 1 + self.ratio_implicit * (kx[i] + ky[i])
            cc1[i] = -ky[i] * self.ratio_implicit
            if iv == 1:
                j = i
                a1[i] = a1[i] - 0.001 * kx[i] * self.ratio_implicit
                cc1[i] = cc1[i] + 0.001 * ky[i] * self.ratio_implicit
            # The water content of each soil cell is converted to water potential by function psiq and stored in array d1 (in bar units).
            d1[i] = psiq(q1[i], qr1[i], qs1[i], self.alpha[j], self.beta[j])
        # The solution of the simultaneous equations in the implicit method alternates between the two directions along the arrays. The reason for this is because the direction of the solution may cause some cumulative bias. The counter numiter determines the direction of the solution.
        # The solution in this section starts from the last soil cell (nn).
        if numiter % 2 == 0:
            # Intermediate arrays dau and cau are computed.
            cau[nn - 1] = psi1[nn - 1]
            dau[nn - 1] = 0
            for i in range(nn - 2, 0, -1):
                p: float = a1[i] * dau[i + 1] + b1[i]  # temporary
                dau[i] = -cc1[i] / p
                cau[i] = (d1[i] - a1[i] * cau[i + 1]) / p
            if iv == 1:
                j = 0
            psi1[0] = psiq(q1[0], qr1[0], qs1[0], self.alpha[j], self.beta[j])
            # psi1 is now computed for soil cells 1 to nn-2. q1 is computed from psi1 by function qpsi.
            for i in range(1, nn - 1):
                if iv == 1:
                    j = i
                psi1[i] = dau[i] * psi1[i - 1] + cau[i]
                q1[i] = qpsi(psi1[i], qr1[i], qs1[i], self.alpha[j], self.beta[j])
        # The alternative direction of solution is executed here. the solution in this section starts from the first soil cell.
        else:
            # Intermediate arrays dau and cau are computed, and the computations described previously are repeated in the opposite direction.
            cau[0] = psi1[0]
            dau[0] = 0
            for i in range(1, nn - 1):
                p: float = a1[i] * dau[i - 1] + b1[i]  # temporary
                dau[i] = -cc1[i] / p
                cau[i] = (d1[i] - a1[i] * cau[i - 1]) / p
            if iv == 1:
                j = nn - 1
            psi1[nn - 1] = psiq(q1[nn - 1], qr1[nn - 1], qs1[nn - 1], self.alpha[j], self.beta[j])
            for i in range(nn - 2, 0, -1):
                if iv == 1:
                    j = i
                psi1[i] = dau[i] * psi1[i + 1] + cau[i]
                q1[i] = qpsi(psi1[i], qr1[i], qs1[i], self.alpha[j], self.beta[j])
        # The limits of water content are now checked and corrected, and function WaterBalance() is called to correct water amounts.
        for i in range(nn):
            if q1[i] < qr1[i]:
                q1[i] = qr1[i]
            if q1[i] > qs1[i]:
                q1[i] = qs1[i]
            if q1[i] > pp1[i]:
                q1[i] = pp1[i]
        self.water_balance(q1, qx, dd, nn)

    cdef void water_balance(self, double q1[], double qx[], double dd[], int nn):
        """Checks and corrects the water balance in the soil cells within a soil column or a soil layer. It is called by water_flux().

        The implicit part of the solution may cause some deviation in the total amount of water to occur. This module corrects the water balance if the sum of deviations is not zero, so that the total amount of water in the array will not change. The correction is proportional to the difference between the previous and present water amounts in each soil cell.

        Arguments
        ---------
        dd
            one dimensional array of layer or column widths.
        nn
            the number of cells in this layer or column.
        qx[]
            one dimensional array of a layer or a column of the previous values of cell.water_content.
        q1[]
            one dimensional array of a layer or a column of cell.water_content.
        """
        dev: float = 0  # Sum of differences of water amount in soil
        dabs: float = 0  # Sum of absolute value of differences in water content in
        # the array between beginning and end of this time step.
        for i in range(nn):
            dev += dd[i] * (q1[i] - qx[i])
            dabs += abs(q1[i] - qx[i])
        if dabs > 0:
            for i in range(nn):
                q1[i] = q1[i] - abs(q1[i] - qx[i]) * dev / (dabs * dd[i])

    def cell_distance(self, int l, int k, int l0, int k0):
        """This function computes the distance between the centers of cells l,k an l0,k0"""
        # Compute vertical distance between centers of l and l0
        x = self.layer_depth_cumsum[l] - self.layer_depth[l] / 2
        x0 = self.layer_depth_cumsum[l0] - self.layer_depth[l0] / 2
        y = self._sim.column_width_cumsum[k] - self._sim.column_width[k] / 2
        y0 = self._sim.column_width_cumsum[k0] - self._sim.column_width[k0] / 2
        return np.linalg.norm((x - x0, y - y0))

    def apply_fertilizer(self, row_space, plant_population):
        """This function simulates the application of nitrogen fertilizer on each date of application."""
        cdef double ferc = 0.01  # constant used to convert kgs per ha to mg cm-2
        # Loop over all fertilizer applications.
        for i in range(NumNitApps):
            # Check if fertilizer is to be applied on this day.
            if self.date.timetuple().tm_yday == NFertilizer[i].day:
                # If this is a BROADCAST fertilizer application:
                if NFertilizer[i].mthfrt == 0:
                    # Compute the number of layers affected by broadcast fertilizer incorporation (lplow), assuming that the depth of incorporation is 20 cm.
                    lplow = np.searchsorted(self.layer_depth_cumsum, 20, side="right")  # number of soil layers affected by cultivation
                    # Calculate the actual depth of fertilizer incorporation in the soil (fertdp) as the sum of all soil layers affected by incorporation.
                    fertdp = self.layer_depth_cumsum[lplow]  # depth of broadcast fertilizer incorporation, cm
                    # Update the nitrogen contents of all soil soil cells affected by this fertilizer application.
                    for l in range(lplow):
                        for k in range(20):
                            self.soil_ammonium_content[l, k] += NFertilizer[i].amtamm * ferc / fertdp
                            self.soil_nitrate_content[l, k] += NFertilizer[i].amtnit * ferc / fertdp
                            self.soil_urea_content[l, k] += NFertilizer[i].amtura * ferc / fertdp
                # If this is a FOLIAR fertilizer application:
                elif NFertilizer[i].mthfrt == 2:
                    # It is assumed that 70% of the amount of ammonium or urea intercepted by the canopy is added to the leaf N content (state.leaf_nitrogen).
                    self.leaf_nitrogen += 0.70 * self.light_interception * (NFertilizer[i].amtamm + NFertilizer[i].amtura) * 1000 / plant_population
                    # The amount not intercepted by the canopy is added to the soil. If the fertilizer is nitrate, it is assumed that all of it is added to the upper soil layer.
                    # Update nitrogen contents of the upper layer.
                    for k in range(20):
                        self.soil_ammonium_content[0, k] += NFertilizer[i].amtamm * (1 - 0.70 * self.light_interception) * ferc / self.layer_depth[0]
                        self.soil_nitrate_content[0, k] += NFertilizer[i].amtnit * ferc / self.layer_depth[0]
                        self.soil_urea_content[0, k] += NFertilizer[i].amtura * (1 - 0.70 * self.light_interception) * ferc / self.layer_depth[0]
                # If this is a SIDE-DRESSING of N fertilizer:
                elif NFertilizer[i].mthfrt == 1:
                    # Define the soil column (ksdr) and the soil layer (lsdr) in which the side-dressed fertilizer is applied.
                    ksdr = NFertilizer[i].ksdr  # the column in which the side-dressed is applied
                    lsdr = NFertilizer[i].ksdr  # the layer in which the side-dressed is applied
                    n00 = 1  # number of soil soil cells in which side-dressed fertilizer is incorporated.
                    # If the volume of this soil cell is less than 100 cm3, it is assumed that the fertilizer is also incorporated in the soil cells below and to the sides of it.
                    if self.cell_area[lsdr, ksdr] < 100:
                        if ksdr < nk - 1:
                            n00 += 1
                        if ksdr > 0:
                            n00 += 1
                        if lsdr < nl - 1:
                            n00 += 1
                    # amount of ammonium N added to the soil by sidedressing (mg per cell)
                    addamm = NFertilizer[i].amtamm * ferc * row_space / n00
                    # amount of nitrate N added to the soil by sidedressing (mg per cell)
                    addnit = NFertilizer[i].amtnit * ferc * row_space / n00
                    # amount of urea N added to the soil by sidedressing (mg per cell)
                    addnur = NFertilizer[i].amtura * ferc * row_space / n00
                    # Update the nitrogen contents of these soil cells.
                    self.soil_nitrate_content[lsdr, ksdr] += addnit / (self.cell_area[lsdr, ksdr])
                    self.soil_ammonium_content[lsdr, ksdr] += addamm / (self.cell_area[lsdr, ksdr])
                    self.soil_urea_content[lsdr][ksdr] += addnur / (self.cell_area[lsdr, ksdr])
                    if self.cell_area[lsdr, ksdr] < 100:
                        if ksdr < nk - 1:
                            kp1 = ksdr + 1  # column to the right of ksdr.
                            self.soil_nitrate_content[lsdr, kp1] += addnit / (self.cell_area[lsdr, kp1])
                            self.soil_ammonium_content[lsdr, kp1] += addamm / (self.cell_area[lsdr, kp1])
                            self.soil_urea_content[lsdr][kp1] += addnur / (self.cell_area[lsdr, kp1])
                        if ksdr > 0:
                            km1 = ksdr - 1  # column to the left of ksdr.
                            self.soil_nitrate_content[lsdr, km1] += addnit / (self.cell_area[lsdr, km1])
                            self.soil_ammonium_content[lsdr, km1] += addamm / (self.cell_area[lsdr, km1])
                            self.soil_urea_content[lsdr][km1] += addnur / (self.cell_area[lsdr, km1])
                        if lsdr < nl - 1:
                            lp1 = lsdr + 1
                            area = self.cell_area[lp1, ksdr]
                            self.soil_nitrate_content[lp1, ksdr] += addnit / area
                            self.soil_ammonium_content[lp1, ksdr] += addamm / area
                            self.soil_urea_content[lp1][ksdr] += addnur / area
                # If this is FERTIGATION (N fertilizer applied in drip irrigation):
                elif NFertilizer[i].mthfrt == 3:
                    # Convert amounts added to mg cm-3, and update the nitrogen content of the soil cell in which the drip outlet is situated.
                    area = self.cell_area[self.drip_y, self.drip_x]
                    self.soil_ammonium_content[self.drip_y, self.drip_x] += NFertilizer[i].amtamm * ferc * row_space / area
                    self.soil_nitrate_content[self.drip_y, self.drip_x] += NFertilizer[i].amtnit * ferc * row_space / area
                    self.soil_urea_content[self.drip_y, self.drip_x] += NFertilizer[i].amtura * ferc * row_space / area

    def water_uptake(self, row_space, per_plant_area):
        """This function computes the uptake of water by plant roots from the soil (i.e., actual transpiration rate)."""
        # Compute the modified light interception factor (LightInter1) for use in computing transpiration rate.
        # modified light interception factor by canopy
        LightInter1 = min(max(self.light_interception * 1.55 - 0.32,  self.light_interception), 1)

        # The potential transpiration is the product of the daytime Penman equation and LightInter1.
        PotentialTranspiration = self.evapotranspiration * LightInter1
        upf = np.zeros((40, 20), dtype=np.float64)  # uptake factor, computed as a ratio, for each soil cell
        uptk = np.zeros((40, 20), dtype=np.float64)  # actual transpiration from each soil cell, cm3 per day
        sumep = 0  # sum of actual transpiration from all soil soil cells, cm3 per day.

        # Compute the reduction due to soil moisture supply by function PsiOnTranspiration().
        # the actual transpiration converted to cm3 per slab units.
        Transp = 0.10 * row_space * PotentialTranspiration * PsiOnTranspiration(self.average_soil_psi)
        while True:
            for l in range(40):
                # Compute, for each layer, the lower and upper water content limits for the transpiration function. These are set from limiting soil water potentials (-15 to -1 bars).
                vh2lo = qpsi(-15, self.thad[l], self.soil_saturated_water_content[l], self.alpha[l], self.beta[l])  # lower limit of water content for the transpiration function
                vh2hi = qpsi(-1, self.thad[l], self.soil_saturated_water_content[l], self.alpha[l], self.beta[l])  # upper limit of water content for the transpiration function
                for k in range(20):
                    # reduction factor for water uptake, caused by low levels of soil water, as a linear function of cell.water_content, between vh2lo and vh2hi.
                    redfac = min(max((self.soil_water_content[l, k] - vh2lo) / (vh2hi - vh2lo), 0), 1)
                    # The computed 'uptake factor' (upf) for each soil cell is the product of 'root weight capable of uptake' and redfac.
                    upf[l][k] = self.root_weight_capable_uptake[l, k] * redfac

            difupt = 0  # the cumulative difference between computed transpiration and actual transpiration, in cm3, due to limitation of PWP.
            for l in range(40):
                for k in range(20):
                    if upf[l][k] > 0 and self.soil_water_content[l, k] > self.thetar[l]:
                        # The amount of water extracted from each cell is proportional to its 'uptake factor'.
                        upth2o = Transp * upf[l][k] / upf.sum()  # transpiration from a soil cell, cm3 per day
                        # Update cell.water_content, storing its previous value as vh2ocx.
                        vh2ocx = self.soil_water_content[l, k]  # previous value of water_content of this cell
                        self.soil_water_content[l, k] -= upth2o / (self.layer_depth[l] * self._sim.column_width[k])
                        # If the new value of cell.water_content is less than the permanent wilting point, modify the value of upth2o so that water_content will be equal to it.
                        if self.soil_water_content[l, k] < self.thetar[l]:
                            self.soil_water_content[l, k] = self.thetar[l]

                            # Compute the difference due to this correction and add it to difupt.
                            xupt = (vh2ocx - self.thetar[l]) * self.layer_depth[l] * self._sim.column_width[k]  # intermediate computation of upth2o
                            difupt += upth2o - xupt
                            upth2o = xupt
                        upth2o = max(upth2o, 0)

                        # Compute sumep as the sum of the actual amount of water extracted from all soil cells. Recalculate uptk of this soil cell as cumulative upth2o.
                        sumep += upth2o
                        uptk[l][k] += upth2o

            # If difupt is greater than zero, redefine the variable Transp as difuptfor use in next loop.
            if difupt > 0:
                Transp = difupt
            else:
                break

        # recompute SoilPsi for all soil cells with roots by calling function PSIQ,
        for l in range(40):
            for k in range(20):
                self.soil_psi[l, k] = (
                    psiq(self.soil_water_content[l, k], self.thad[l], self.soil_saturated_water_content[l], self.alpha[l], self.beta[l])
                    - PsiOsmotic(self.soil_water_content[l, k], self.soil_saturated_water_content[l], ElCondSatSoilToday)
                )

        # compute ActualTranspiration as actual water transpired, in mm.
        self.actual_transpiration = sumep * 10 / row_space

        # Zeroize the amounts of NH4 and NO3 nitrogen taken up from the soil.
        self.supplied_nitrate_nitrogen = 0
        self.supplied_ammonium_nitrogen = 0

        # Compute the proportional N requirement from each soil cell with roots, and call function NitrogenUptake() to compute nitrogen uptake.
        if sumep > 0 and self.total_required_nitrogen > 0:
            for l in range(40):
                for k in range(20):
                    if uptk[l][k] > 0:
                        # proportional allocation of TotalRequiredN to each cell
                        reqnc = self.total_required_nitrogen * uptk[l][k] / sumep
                        self.nitrogen_uptake((l, k), reqnc, row_space, per_plant_area)

    def nitrogen_uptake(self, index, reqnc, row_space, per_plant_area):
        """Computes the uptake of nitrate and ammonium N from a soil cell. It is called by WaterUptake().

        Arguments
        ---------
        index
            for ndarray indexing
        reqnc
            maximum N uptake (proportional to total N required for plant growth), g N per plant.
        """
        l, k = index
        nitrate_nitrogen_content = self.soil_nitrate_content[index]
        water_content = self.soil_water_content[index]
        # Constant parameters:
        halfn = 0.08  # the N concentration in soil water (mg cm-3) at which
        # uptake is half of the possible rate.
        cparupmax = 0.5  # constant parameter for computing upmax.
        p1 = 100
        p2 = 5  # constant parameters for computing AmmonNDissolved.

        # coefficient used to convert g per plant to mg cm-3 units.
        coeff: float = 10 * row_space / (per_plant_area * self.layer_depth[l] * self._sim.column_width[k])
        # A Michaelis-Menten procedure is used to compute the rate of nitrate uptake from each cell. The maximum possible amount of uptake is reqnc (g N per plant), and the half of this rate occurs when the nitrate concentration in the soil solution is halfn (mg N per cm3 of soil water).
        # Compute the uptake of nitrate from this soil cell, upno3c in g N per plant units.
        # Define the maximum possible uptake, upmax, as a fraction of VolNo3NContent.
        if nitrate_nitrogen_content > 0:
            # uptake rate of nitrate, g N per plant per day
            upno3c = reqnc * nitrate_nitrogen_content / (halfn * water_content + nitrate_nitrogen_content)
            # maximum possible uptake rate, mg N per soil cell per day
            upmax = cparupmax * nitrate_nitrogen_content
            # Make sure that uptake will not exceed upmax and update VolNo3NContent and upno3c.
            if (coeff * upno3c) < upmax:
                self.soil_nitrate_content[index] -= coeff * upno3c
            else:
                self.soil_nitrate_content[index] -= upmax
                upno3c = upmax / coeff
            # upno3c is added to the total uptake by the plant (supplied_nitrate_nitrogen).
            self.supplied_nitrate_nitrogen += upno3c
        # Ammonium in the soil is in a dynamic equilibrium between the adsorbed and the soluble fractions. The parameters p1 and p2 are used to compute the dissolved concentration, AmmonNDissolved, of ammoniumnitrogen. bb, cc, ee are intermediate values for computing.
        if self.soil_ammonium_content[l, k] > 0:
            bb = p1 + p2 * water_content - self.soil_ammonium_content[l, k]
            cc = p2 * water_content * self.soil_ammonium_content[l, k]
            ee = max(bb ** 2 + 4 * cc, 0)
            AmmonNDissolved = (sqrt(ee) - bb) / 2  # ammonium N dissolved in soil water, mg cm-3 of soil
            # Uptake of ammonium N is now computed from AmmonNDissolved , using the Michaelis-Menten method, as for nitrate. upnh4c is added to the total uptake supplied_ammonium_nitrogen.
            if AmmonNDissolved > 0:
                # uptake rate of ammonium, g N per plant per day
                upnh4c = reqnc * AmmonNDissolved / (halfn * water_content + AmmonNDissolved)
                # maximum possible uptake rate, mg N per soil cell per day
                upmax = cparupmax * self.soil_ammonium_content[l, k]
                if (coeff * upnh4c) < upmax:
                    self.soil_ammonium_content[l, k] -= coeff * upnh4c
                else:
                    self.soil_ammonium_content[l, k] -= upmax
                    upnh4c = upmax / coeff
                self.supplied_ammonium_nitrogen += upnh4c

    def average_psi(self, row_space):
        """This function computes and returns the average soil water potential of the root zone of the soil slab. This average is weighted by the amount of active roots (roots capable of uptake) in each soil cell. Soil zones without roots are not included."""
        # Constants used:
        vrcumin = 0.1e-9
        vrcumax = 0.025

        psinum = np.zeros(9, dtype=np.float64)  # sum of weighting coefficients for computing avgwat.
        sumwat = np.zeros(9, dtype=np.float64)  # sum of weighted soil water content for computing avgwat.
        sumdl = np.zeros(9, dtype=np.float64)  # sum of thickness of all soil layers containing roots.
        # Compute sum of dl as sumdl for each soil horizon.
        for l in range(40):
            j = self.soil_horizon_number[l]
            sumdl[j] += self.layer_depth[l]
            for k in range(20):
                # Check that RootWtCapblUptake in any cell is more than a minimum value vrcumin.
                if self.root_weight_capable_uptake[l, k] >= vrcumin:
                    # Compute sumwat as the weighted sum of the water content, and psinum as the sum of these weights. Weighting is by root weight capable of uptake, or if it exceeds a maximum value (vrcumax) this maximum value is used for weighting.
                    sumwat[j] += self.soil_water_content[l, k] * self.layer_depth[l] * self._sim.column_width[k] * min(self.root_weight_capable_uptake[l, k], vrcumax)
                    psinum[j] += self.layer_depth[l] * self._sim.column_width[k] * min(self.root_weight_capable_uptake[l, k], vrcumax)
        sumpsi = 0  # weighted sum of avgpsi
        sumnum = 0  # sum of weighting coefficients for computing average_soil_psi.
        for j in range(9):
            if psinum[j] > 0 and sumdl[j] > 0:
                # Compute avgwat and the parameters to compute the soil water potential in each soil horizon
                avgwat = sumwat[j] / psinum[j]  # weighted average soil water content (V/V) in root zone
                # Soil water potential computed for a soil profile layer:
                avgpsi = psiq(avgwat, self._sim.soil_hydrology["air_dry"][j], self.soil_hydrology["theta"][j], self.soil_hydrology["alpha"][j], self.soil_hydrology["beta"][j]) - PsiOsmotic(avgwat, self.soil_hydrology["theta"][j], ElCondSatSoilToday)
                # Use this to compute the average for the whole root zone.
                sumpsi += avgpsi * psinum[j]
                sumnum += psinum[j]
        return sumpsi / sumnum if sumnum > 0 else 0  # average soil water potential for the whole root zone
    #end soil


cdef class Simulation:
    cdef uint32_t _emerge_day
    cdef uint32_t _start_day
    cdef uint32_t _stop_day
    cdef uint32_t _plant_day
    cdef uint32_t _topping_day
    cdef uint32_t _defoliate_day
    cdef uint32_t _first_bloom_day
    cdef uint32_t _first_square_day
    cdef public numpy.ndarray alpha  # parameter of the Van Genuchten equation.
    cdef public numpy.ndarray beta  # parameter of the Van Genuchten equation.
    cdef public numpy.ndarray soil_saturated_hydraulic_conductivity
    cdef public numpy.ndarray heat_capacity_soil_solid  # heat capacity of the solid phase of the soil.
    cdef public numpy.ndarray pore_space  # pore space of soil, volume fraction.
    cdef public numpy.ndarray cell_area
    cdef public numpy.ndarray column_width
    cdef public numpy.ndarray column_width_cumsum
    cdef public numpy.ndarray layer_depth
    cdef public numpy.ndarray layer_depth_cumsum
    cdef public numpy.ndarray soil_hydrology
    cdef public numpy.ndarray soil_bulk_density
    cdef public numpy.ndarray soil_clay_volume_fraction
    cdef public numpy.ndarray soil_sand_volume_fraction
    cdef public numpy.ndarray soil_saturated_water_content  # saturated volumetric water content of each soil layer, cm3 cm-3.
    cdef public numpy.ndarray soil_horizon_number  # the soil horizon number associated with each soil layer in the slab.
    cdef public numpy.ndarray max_water_capacity  # volumetric water content of a soil layer at maximum capacity, before drainage, cm3 cm-3.
    cdef public numpy.ndarray thad  # residual volumetric water content of soil layers (at air-dry condition), cm3 cm-3.
    cdef public numpy.ndarray field_capacity  # volumetric water content of soil at field capacity for each soil layer, cm3 cm-3.
    cdef public numpy.ndarray oma  # organic matter at the beginning of the season, percent of soil weight,
    cdef public numpy.ndarray soil_nitrate_flow_fraction  # fraction of nitrate that can move to the next layer.
    cdef public object meteor
    cdef public unsigned int emerge_switch
    cdef public unsigned int version
    cdef public unsigned int year
    cdef public uint32_t plant_row_column  # column number to the left of plant row location.
    cdef public numpy.ndarray rnno3  # residual nitrogen as nitrate in soil at beginning of season, kg per ha.
    cdef public numpy.ndarray rnnh4  # residual nitrogen as ammonium in soil at beginning of season, kg per ha.
    cdef public numpy.ndarray thetar  # volumetric water content of soil layers at permanent wilting point (-15 bars), cm3 cm-3.
    cdef public double cultivar_parameters[51]
    cdef public double density_factor  # empirical plant density factor.
    cdef public double elevation  # meter
    cdef public double latitude
    cdef public double longitude
    cdef public double max_leaf_area_index
    cdef public double ptsred  # The effect of moisture stress on the photosynthetic rate
    cdef public double plant_population  # plant population, plants per hectar.
    cdef public double plants_per_meter  # average number of plants pre meter of row.
    cdef public double per_plant_area  # average soil surface area per plant, dm2
    cdef public double row_space  # average row spacing, cm.
    cdef public double site_parameters[17]
    cdef public double skip_row_width  # the smaller distance between skip rows, cm
    cdef public double ratio_implicit  # the ratio for the implicit numerical solution of the water transport equation (used in FLUXI and in SFLUX.
    cdef public double soil_psi_field_capacity  # soil matric water potential at field capacity, bars (suggested value -0.33 to -0.1).
    cdef public double max_conductivity  # the maximum value for non-dimensional hydraulic conductivity
    cdef public State _current_state
    cdef double defkgh  # amount of defoliant applied, kg per ha
    cdef double tdfkgh  # total cumulative amount of defoliant
    cdef bool_t idsw  # switch indicating if predicted defoliation date was defined.
    # switch affecting the method of computing soil temperature.
    # 0 = one dimensional (no horizontal flux) - used to predict emergence when emergence date is not known;
    # 1 = one dimensional - used before emergence when emergence date is given;
    # 2 = two dimensional - used after emergence.
    irradiation_soil_surface = np.ones(20)  # the relative radiation received by a soil column, as affected by shading by plant canopy.
    irrigation = {}

    def __init__(self, version=0x0400, **kwargs):
        self.version = version
        self.max_leaf_area_index = 0.001
        for attr in (
            "start_date",
            "stop_date",
            "emerge_date",
            "plant_date",
            "topping_date",
            "latitude",
            "longitude",
            "elevation",
            "site_parameters",
            "cultivar_parameters",
            "row_space",
            "skip_row_width",
            "plants_per_meter",
        ):
            if attr in kwargs:
                setattr(self, attr, kwargs.get(attr))
        meteor = kwargs.get("meteor", {})
        for d, met in meteor.items():
            if "tdew" not in met:
                met["tdew"] = tdewest(met["tmax"], self.site_parameters[5], self.site_parameters[6])
        self.meteor = meteor

    @property
    def start_date(self):
        return date.fromordinal(self._start_day)

    @start_date.setter
    def start_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._start_day = d.toordinal()

    @property
    def stop_date(self):
        return date.fromordinal(self._stop_day)

    @stop_date.setter
    def stop_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._stop_day = d.toordinal()

    @property
    def emerge_date(self):
        return date.fromordinal(self._emerge_day) if self._emerge_day else None

    @emerge_date.setter
    def emerge_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._emerge_day = d.toordinal()

    @property
    def plant_date(self):
        return date.fromordinal(self._plant_day) if self._plant_day else None

    @plant_date.setter
    def plant_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._plant_day = d.toordinal()

    @property
    def topping_date(self):
        if self.version >= 0x500 and self._topping_day > 0:
            return date.fromordinal(self._topping_day)
        return None

    @topping_date.setter
    def topping_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._topping_day = d.toordinal()

    @property
    def first_square_date(self):
        return date.fromordinal(self._first_square_day) if self._first_square_day else None

    @first_square_date.setter
    def first_square_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._first_square_day = d.toordinal()

    @property
    def first_bloom_date(self):
        return date.fromordinal(self._first_bloom_day) if self._first_bloom_day else None

    @first_bloom_date.setter
    def first_bloom_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._first_bloom_day = d.toordinal()

    @property
    def defoliate_date(self):
        return date.fromordinal(self._defoliate_day) if self._defoliate_day else None

    @defoliate_date.setter
    def defoliate_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._defoliate_day = d.toordinal()

    def _init_state(self):
        cdef State state0 = self._current_state
        state0.date = self.start_date
        state0.plant_height = 4.0
        state0.stem_weight = 0.2
        state0.petiole_weight = 0
        state0.green_bolls_burr_weight = 0
        state0.open_bolls_burr_weight = 0
        state0.reserve_carbohydrate = 0.06
        state0.water_stress = 1
        state0.water_stress_stem = 1
        state0.carbon_stress = 1
        state0.extra_carbon = 0
        state0.leaf_area_index = 0.001
        state0.leaf_weight = 0.20
        state0.leaf_nitrogen = 0.0112
        state0.number_of_green_bolls = 0
        state0.fiber_length = 0
        state0.fiber_strength = 0
        state0.nitrogen_stress = 1
        state0.nitrogen_stress_vegetative = 1
        state0.nitrogen_stress_fruiting = 1
        state0.nitrogen_stress_root = 1
        state0.total_required_nitrogen = 0
        state0.petiole_nitrogen_concentration = 0
        state0.seed_nitrogen_concentration = 0
        state0.burr_nitrogen_concentration = 0
        state0.burr_nitrogen = 0
        state0.seed_nitrogen = 0
        state0.root_nitrogen_concentration = .026
        state0.root_nitrogen = 0.0052
        state0.square_nitrogen_concentration = 0
        state0.square_nitrogen = 0
        state0.stem_nitrogen_concentration = 0.036
        state0.stem_nitrogen = 0.0072
        state0.fruit_growth_ratio = 1
        state0.ginning_percent = 0.35
        state0.number_of_pre_fruiting_nodes = 1
        state0.total_actual_leaf_growth = 0
        state0.total_actual_petiole_growth = 0
        state0.carbon_allocated_for_root_growth = 0
        state0.supplied_ammonium_nitrogen = 0
        state0.supplied_nitrate_nitrogen = 0
        state0.petiole_nitrate_nitrogen_concentration = 0
        state0.delay_of_new_fruiting_branch = [0, 0, 0]
        state0.fruiting_nodes_age = np.zeros((3, 30, 5), dtype=np.double)
        state0.fruiting_nodes_boll_weight = np.zeros((3, 30, 5), dtype=np.double)
        state0.fruiting_nodes_fraction = np.zeros((3, 30, 5), dtype=np.double)
        state0.fruiting_nodes_stage = np.zeros((3, 30, 5), dtype=np.int_)
        state0.fruiting_nodes_ginning_percent = np.ones((3, 30, 5), dtype=np.double) * 0.35
        for i in range(9):
            state0.pre_fruiting_nodes_age[i] = 0
            state0.pre_fruiting_leaf_area[i] = 0
            state0.leaf_weight_pre_fruiting[i] = 0

    def _initialize_root_data(self):
        """ This function initializes the root submodel parameters and variables."""
        state0 = self._current_state
        # The parameters of the root model are defined for each root class:
        # grind(i), cuind(i), thtrn(i), trn(i), thdth(i), dth(i).
        cdef double rlint = 10  # Vertical interval, in cm, along the taproot, for initiating lateral roots.
        cdef int ll = 1  # Counter for layers with lateral roots.
        cdef double sumdl = 0  # Distance from soil surface to the middle of a soil layer.
        for l in range(nl):
            # Using the value of rlint (interval between lateral roots), the layers from which lateral roots may be initiated are now computed.
            # LateralRootFlag[l] is assigned a value of 1 for these layers.
            LateralRootFlag[l] = 0
            sumdl = 0.5 * self.layer_depth[l] + (self.layer_depth_cumsum[l - 1] if l > 0 else 0)
            if sumdl >= ll * rlint:
                LateralRootFlag[l] = 1
                ll += 1

        state0.init_root_data(self.plant_row_column, 0.01 * self.row_space / self.per_plant_area)
        # Start loop for all soil layers containing roots.
        self.last_layer_with_root_depth = self.layer_depth_cumsum[6]  # compute total depth to the last layer with roots (self.last_layer_with_root_depth).
        # Initial value of taproot length, taproot_length, is computed to the middle of the last layer with roots. The last soil layer with taproot, state.taproot_layer_number, is defined.
        state0.taproot_length = (self.last_layer_with_root_depth - 0.5 * self.layer_depth[6])
        state0.taproot_layer_number = 6


    def _copy_state(self, State from_, State to_):
        to_._ = from_._

    def _soil_temperature_init(self):
        """This function is called from SoilTemperature() at the start of the simulation. It sets initial values to soil and canopy temperatures."""
        # Compute initial values of soil temperature: It is assumed that at the start of simulation
        # the temperature of the first soil layer (upper boundary) is equal to the average air temperature
        # of the previous five days (if climate data not available - start from first climate data).
        # NOTE: For a good simulation of soil temperature, it is recommended to start simulation at
        # least 10 days before planting date. This means that climate data should be available for
        # this period. This is especially important if emergence date has to be simulated.
        state0 = self._current_state
        idd = 0  # number of days minus 4 from start of simulation.
        tsi1 = 0  # Upper boundary (surface layer) initial soil temperature, C.
        for i in range(5):
            d = self.start_date + timedelta(days=i)
            tsi1 += self.meteor[d]["tmax"] + self.meteor[d]["tmin"]
        tsi1 /= 10
        # The temperature of the last soil layer (lower boundary) is computed as a sinusoidal function of day of year, with site-specific parameters.
        state0.deep_soil_temperature = self.site_parameters[9] + self.site_parameters[10] * sin(2 * pi * (self.start_date.timetuple().tm_yday - self.site_parameters[11]) / 365) + 273.161
        # hourly_soil_temperature is assigned to all columns, converted to degrees K.
        tsi1 += 273.161
        # The temperatures of the other soil layers are linearly interpolated.
        # tsi = computed initial soil temperature, C, for each layer
        state0.hourly_soil_temperature[0] = ((np.arange(40) * state0.deep_soil_temperature + ((40 - 1) - np.arange(40)) * tsi1) / (40 - 1))[:, None].repeat(20, axis=1)

    def _stress(self, u):
        state = self._current_state
        global AverageLwp
        # The following constant parameters are used:
        cdef double[9] vstrs = [-3.0, 3.229, 1.907, 0.321, -0.10, 1.230, 0.340, 0.30, 0.05]
        # Call state.leaf_water_potential() to compute leaf water potentials.
        state.leaf_water_potential(self.row_space)
        # The running averages, for the last three days, are computed:
        # average_min_leaf_water_potential is the average of state.min_leaf_water_potential, and AverageLwp of state.min_leaf_water_potential + state.max_leaf_water_potential.
        state.average_min_leaf_water_potential += (state.min_leaf_water_potential - LwpMinX[2]) / 3
        AverageLwp += (state.min_leaf_water_potential + state.max_leaf_water_potential - LwpX[2]) / 3
        for i in (2, 1):
            LwpMinX[i] = LwpMinX[i - 1]
            LwpX[i] = LwpX[i - 1]
        LwpMinX[0] = state.min_leaf_water_potential
        LwpX[0] = state.min_leaf_water_potential + state.max_leaf_water_potential
        if state.kday < 5:
            self.ptsred = 1
            state.water_stress_stem = 1
            return
        # The computation of ptsred, the effect of moisture stress on the photosynthetic rate, is based on the following work:
        # Ephrath, J.E., Marani, A., Bravdo, B.A., 1990. Effects of moisture stress on stomatal resistance and photosynthetic rate in cotton (Gossypium hirsutum) 1. Controlled levels of stress. Field Crops Res.23:117-131.
        # It is a function of average_min_leaf_water_potential (average min_leaf_water_potential for the last three days).
        state.average_min_leaf_water_potential = max(state.average_min_leaf_water_potential, vstrs[0])
        self.ptsred = min(vstrs[1] + state.average_min_leaf_water_potential * (vstrs[2] + vstrs[3] * state.average_min_leaf_water_potential), 1)
        # The general moisture stress factor (WaterStress) is computed as an empirical function of AverageLwp. psilim, the value of AverageLwp at the maximum value of the function, is used for truncating it.
        # The minimum value of WaterStress is 0.05, and the maximum is 1.
        cdef double psilim  # limiting value of AverageLwp.
        cdef double WaterStress
        psilim = -0.5 * vstrs[5] / vstrs[6]
        WaterStress = 1 if AverageLwp > psilim else min(max(vstrs[4] - AverageLwp * (vstrs[5] + vstrs[6] * AverageLwp), 0.05), 1)
        # Water stress affecting plant height and stem growth(WaterStressStem) is assumed to be more severe than WaterStress, especially at low WaterStress values.
        state.water_stress_stem = max(WaterStress * (1 + vstrs[7] * (2 - WaterStress)) - vstrs[7], vstrs[8])
        state.water_stress = WaterStress

    def _potential_fruit_growth(self, u):
        """
        This function simulates the potential growth of fruiting sites of cotton plants. It is called from PlantGrowth(). It calls TemperatureOnFruitGrowthRate()

        The following global variables are set here:
            PotGroAllBolls, PotGroAllBurrs, PotGroAllSquares.

        References:
        Marani, A. 1979. Growth rate of cotton bolls and their components. Field Crops Res. 2:169-175.
        Marani, A., Phene, C.J. and Cardon, G.E. 1992. CALGOS, a version of GOSSYM adapted for irrigated cotton.  III. leaf and boll growth routines. Beltwide Cotton Grow, Res. Conf. 1992:1361-1363.
        """
        state = self._current_state
        global PotGroAllSquares, PotGroAllBolls, PotGroAllBurrs
        # The constant parameters used:
        cdef double[5] vpotfrt = [0.72, 0.30, 3.875, 0.125, 0.17]
        # Compute tfrt for the effect of temperature on boll and burr growth rates. Function TemperatureOnFruitGrowthRate() is used (with parameters derived from GOSSYM), for day time and night time temperatures, weighted by day and night lengths.
        cdef double tfrt  # the effect of temperature on rate of boll, burr or square growth.
        tfrt = (state.day_length * TemperatureOnFruitGrowthRate(state.daytime_temperature) + (24 - state.day_length) * TemperatureOnFruitGrowthRate(state.nighttime_temperature)) / 24
        # Assign zero to sums of potential growth of squares, bolls and burrs.
        PotGroAllSquares = 0
        PotGroAllBolls = 0
        PotGroAllBurrs = 0
        # Assign values for the boll growth equation parameters. These are cultivar - specific.
        cdef double agemax = self.cultivar_parameters[9]  # maximum boll growth period (physiological days).
        cdef double rbmax = self.cultivar_parameters[10]  # maximum rate of boll (seed and lint) growth, g per boll per physiological day.
        cdef double wbmax = self.cultivar_parameters[11]  # maximum possible boll (seed and lint) weight, g per boll.
        # Loop for all vegetative stems.
        for i, stage in np.ndenumerate(state.fruiting_nodes_stage):
            # Calculate potential square growth for node (k,l,m).
            # Sum potential growth rates of squares as PotGroAllSquares.
            if stage == Stage.Square:
                # ratesqr is the rate of square growth, g per square per day.
                # The routine for this is derived from GOSSYM, and so are the parameters used.
                ratesqr = tfrt * vpotfrt[3] * exp(-vpotfrt[2] + vpotfrt[3] * state.fruiting_nodes_age[i])
                state.square_potential_growth[i] = ratesqr * state.fruiting_nodes_fraction[i]
                PotGroAllSquares += state.square_potential_growth[i]
            # Growth of seedcotton is simulated separately from the growth of burrs. The logistic function is used to simulate growth of seedcotton. The constants of this function for cultivar 'Acala-SJ2', are based on the data of Marani (1979); they are derived from calibration for other cultivars
            # agemax is the age of the boll (in physiological days after bloom) at the time when the boll growth rate is maximal.
            # rbmax is the potential maximum rate of boll growth (g seeds plus lint dry weight per physiological day) at this age.
            # wbmax is the maximum potential weight of seed plus lint (g dry weight per boll).
            # The auxiliary variable pex is computed as
            #    pex = exp(-4 * rbmax * (t - agemax) / wbmax)
            # where t is the physiological age of the boll after bloom (= agebol).
            # Boll weight (seed plus lint) at age T, according to the logistic function is:
            #    wbol = wbmax / (1 + pex)
            # and the potential boll growth rate at this age will be the derivative of this function:
            #    ratebol = 4 * rbmax * pex / (1. + pex)**2
            elif stage in [Stage.YoungGreenBoll, Stage.GreenBoll]:
                # pex is an intermediate variable to compute boll growth.
                pex = exp(-4 * rbmax * (state.fruiting_nodes_boll_age[i] - agemax) / wbmax)
                # ratebol is the rate of boll (seed and lint) growth, g per boll per day.
                ratebol = 4 * tfrt * rbmax * pex / (1 + pex) ** 2
                # Potential growth rate of the burrs is assumed to be constant (vpotfrt[4] g dry weight per day) until the boll reaches its final volume. This occurs at the age of 22 physiological days in 'Acala-SJ2'. Both ratebol and ratebur are modified by temperature (tfrt) and ratebur is also affected by water stress (wfdb).
                # Compute wfdb for the effect of water stress on burr growth rate. wfdb is the effect of water stress on rate of burr growth.
                wfdb = min(max(vpotfrt[0] + vpotfrt[1] * state.water_stress, 0), 1)
                ratebur = None  # rate of burr growth, g per boll per day.
                if state.fruiting_nodes_boll_age[i] >= 22:
                    ratebur = 0
                else:
                    ratebur = vpotfrt[4] * tfrt * wfdb
                # Potential boll (seeds and lint) growth rate (ratebol) and potential burr growth rate (ratebur) are multiplied by FruitFraction to compute PotGroBolls and PotGroBurrs for node (k,l,m).
                state.fruiting_nodes_boll_potential_growth[i] = ratebol * state.fruiting_nodes_fraction[i]
                state.burr_potential_growth[i] = ratebur * state.fruiting_nodes_fraction[i]
                # Sum potential growth rates of bolls and burrs as PotGroAllBolls and PotGroAllBurrs, respectively.
                PotGroAllBolls += state.fruiting_nodes_boll_potential_growth[i]
                PotGroAllBurrs += state.burr_potential_growth[i]

            # If these are not green bolls, their potential growth is 0. End loop.
            else:
                state.fruiting_nodes_boll_potential_growth[i] = 0
                state.burr_potential_growth[i] = 0

    def _potential_leaf_growth(self, u):
        """
        This function simulates the potential growth of leaves of cotton plants. It is called from self._growth(). It calls function temperature_on_leaf_growth_rate().

        The following monomolecular growth function is used :
            leaf area = smax * (1 - exp(-c * pow(t,p)))
        where    smax = maximum leaf area.
            t = time (leaf age).
            c, p = constant parameters.
        Note: p is constant for all leaves, whereas smax and c depend on leaf position.
        The rate per day (the derivative of this function) is :
            r = smax * c * p * exp(-c * pow(t,p)) * pow(t, (p-1))
        """
        state = self._current_state
        # The following constant parameters. are used in this function:
        p = 1.6  # parameter of the leaf growth rate equation.
        vpotlf = [3.0, 0.95, 1.2, 13.5, -0.62143, 0.109365, 0.00137566, 0.025, 0.00005, 30., 0.02, 0.001, 2.50, 0.18]
        # Calculate water stress reduction factor for leaf growth rate (wstrlf). This has been empirically calibrated in COTTON2K.
        wstrlf = state.water_stress * (1 + vpotlf[0] * (2 - state.water_stress)) - vpotlf[0]
        if wstrlf < 0.05:
            wstrlf = 0.05
        # Calculate wtfstrs, the effect of leaf water stress on state.leaf_weight_area_ratio (the ratio of leaf dry weight to leaf area). This has also been empirically calibrated in COTTON2K.
        wtfstrs = vpotlf[1] + vpotlf[2] * (1 - wstrlf)
        # Compute the ratio of leaf dry weight increment to leaf area increment (g per dm2), as a function of average daily temperature and water stress. Parameters for the effect of temperature are adapted from GOSSYM.
        tdday = state.average_temperature  # limited value of today's average temperature.
        if tdday < vpotlf[3]:
            tdday = vpotlf[3]
        state.leaf_weight_area_ratio = wtfstrs / (vpotlf[4] + tdday * (vpotlf[5] - tdday * vpotlf[6]))
        # Assign zero to total potential growth of leaf and petiole.
        state.leaf_potential_growth = 0
        state.petiole_potential_growth = 0
        c = 0  # parameter of the leaf growth rate equation.
        smax = 0  # maximum possible leaf area, a parameter of the leaf growth rate equation.
        rate = 0  # growth rate of area of a leaf.
        # Compute the potential growth rate of prefruiting leaves. smax and c are functions of prefruiting node number.
        for j in range(state.number_of_pre_fruiting_nodes):
            if state.pre_fruiting_leaf_area[j] <= 0:
                PotGroLeafAreaPreFru[j] = 0
                PotGroLeafWeightPreFru[j] = 0
                PotGroPetioleWeightPreFru[j] = 0
            else:
                jp1 = j + 1
                smax = max(self.cultivar_parameters[4], jp1 * (self.cultivar_parameters[2] - self.cultivar_parameters[3] * jp1))
                c = vpotlf[7] + vpotlf[8] * jp1 * (jp1 - vpotlf[9])
                rate = smax * c * p * exp(-c * pow(state.pre_fruiting_nodes_age[j], p)) * pow(state.pre_fruiting_nodes_age[j], (p - 1))
                # Growth rate is modified by water stress and a function of average temperature.
                # Compute potential growth of leaf area, leaf weight and petiole weight for leaf on node j. Add leaf weight potential growth to leaf_potential_growth.
                # Add potential growth of petiole weight to petiole_potential_growth.
                if rate >= 1e-12:
                    PotGroLeafAreaPreFru[j] = rate * wstrlf * temperature_on_leaf_growth_rate(state.average_temperature)
                    PotGroLeafWeightPreFru[j] = PotGroLeafAreaPreFru[j] * state.leaf_weight_area_ratio
                    PotGroPetioleWeightPreFru[j] = PotGroLeafAreaPreFru[j] * state.leaf_weight_area_ratio * vpotlf[13]
                    state.leaf_potential_growth += PotGroLeafWeightPreFru[j]
                    state.petiole_potential_growth += PotGroPetioleWeightPreFru[j]
        # denfac is the effect of plant density on leaf growth rate.
        cdef double denfac = 1 - vpotlf[12] * (1 - self.density_factor)
        for (k, l), area in np.ndenumerate(state.main_stem_leaf_area):
            # smax and c are  functions of fruiting branch number.
            # smax is modified by plant density, using the density factor denfac.
            # Compute potential main stem leaf growth, assuming that the main stem leaf is initiated at the same time as leaf (k,l,0).
            if area <= 0:
                state.main_stem_leaf_area_potential_growth[k, l] = 0
                state.main_stem_leaf_potential_growth[k, l] = 0
                state.main_stem_leaf_petiole_potential_growth[k, l] = 0
            else:
                lp1 = l + 1
                smax = denfac * (self.cultivar_parameters[5] + self.cultivar_parameters[6] * lp1 * (self.cultivar_parameters[7] - lp1))
                smax = max(self.cultivar_parameters[4], smax)
                c = vpotlf[10] + lp1 * vpotlf[11]
                if state.node_leaf_age[k, l, 0] > 70:
                    rate = 0
                else:
                    rate = smax * c * p * exp(-c * pow(state.node_leaf_age[k, l, 0], p)) * pow(state.node_leaf_age[k, l, 0], (p - 1))
                # Add leaf and petiole weight potential growth to SPDWL and SPDWP.
                if rate >= 1e-12:
                    state.main_stem_leaf_area_potential_growth[k, l] = rate * wstrlf * temperature_on_leaf_growth_rate(state.average_temperature)
                    state.main_stem_leaf_potential_growth[k, l] = state.main_stem_leaf_area_potential_growth[k, l] * state.leaf_weight_area_ratio
                    state.main_stem_leaf_petiole_potential_growth[k, l] = state.main_stem_leaf_area_potential_growth[k, l] * state.leaf_weight_area_ratio * vpotlf[13]
                    state.leaf_potential_growth += state.main_stem_leaf_potential_growth[k, l]
                    state.petiole_potential_growth += state.main_stem_leaf_petiole_potential_growth[k, l]
            # Assign smax value of this main stem leaf to smaxx, c to cc.
            # Loop over the nodes of this fruiting branch.
            smaxx = smax  # value of smax for the corresponding main stem leaf.
            cc = c  # value of c for the corresponding main stem leaf.
            for m in range(5):
                if state.node_leaf_area[k, l, m] <= 0:
                    state.node_leaf_area_potential_growth[k, l, m] = 0
                    state.node_petiole_potential_growth[k, l, m] = 0
                # Compute potential growth of leaf area and leaf weight for leaf on fruiting branch node (k,l,m).
                # Add leaf and petiole weight potential growth to spdwl and spdwp.
                else:
                    mp1 = m + 1
                    # smax and c are reduced as a function of node number on this fruiting branch.
                    smax = smaxx * (1 - self.cultivar_parameters[8] * mp1)
                    c = cc * (1 - self.cultivar_parameters[8] * mp1)
                    # Compute potential growth for the leaves on fruiting branches.
                    if state.node_leaf_age[k, l, m] > 70:
                        rate = 0
                    else:
                        rate = smax * c * p * exp(-c * pow(state.node_leaf_age[k, l, m], p)) * pow(state.node_leaf_age[k, l, m], (p - 1))
                    if rate >= 1e-12:
                        # Growth rate is modified by water stress. Potential growth is computed as a function of average temperature.
                        state.node_leaf_area_potential_growth[k, l, m] = rate * wstrlf * temperature_on_leaf_growth_rate(state.average_temperature)
                        state.node_petiole_potential_growth[k, l, m] = state.node_leaf_area_potential_growth[k, l, m] * state.leaf_weight_area_ratio * vpotlf[13]
                        state.leaf_potential_growth += state.node_leaf_area_potential_growth[k, l, m] * state.leaf_weight_area_ratio
                        state.petiole_potential_growth += state.node_petiole_potential_growth[k, l, m]

    def _defoliate(self, u):
        """This function simulates the effects of defoliating chemicals applied on the cotton. It is called from SimulateThisDay()."""
        global PercentDefoliation
        cdef State state = self._current_state
        # constant parameters:
        cdef double p1 = -50.0
        cdef double p2 = 0.525
        cdef double p3 = 7.06
        cdef double p4 = 0.85
        cdef double p5 = 2.48
        cdef double p6 = 0.0374
        cdef double p7 = 0.0020

        # If this is first day set initial values of tdfkgh, defkgh to 0.
        if state.date <= self.emerge_date:
            self.tdfkgh = 0
            self.defkgh = 0
            self.idsw = False
        # Start a loop for five possible defoliant applications.
        for i in range(5):
            # If there are open bolls and defoliation prediction has been set, execute the following.
            if state.number_of_open_bolls > 0 and DefoliantAppRate[i] <= -99.9:
                # percentage of open bolls in total boll number
                OpenRatio = <int>(100 * state.number_of_open_bolls / (state.number_of_open_bolls + state.number_of_green_bolls))
                if i == 0 and not self.idsw:
                    # If this is first defoliation - check the percentage of boll opening.
                    # If it is after the defined date, or the percent boll opening is greater than the defined threshold - set defoliation date as this day and set a second prediction.
                    if (state.date.timetuple().tm_yday >= DefoliationDate[0] > 0) or OpenRatio > DefoliationMethod[i]:
                        self.idsw = True
                        DefoliationDate[0] = state.date.timetuple().tm_yday
                        DefoliantAppRate[1] = -99.9
                        if self.defoliate_date is None or state.date < self.defoliate_date:
                            self.defoliate_date = state.date
                        DefoliationMethod[0] = 0
                # If 10 days have passed since the last defoliation, and the leaf area index is still greater than 0.2, set another defoliation.
                if i >= 1:
                    if state.date == doy2date(self.year, DefoliationDate[i - 1] + 10) and state.leaf_area_index >= 0.2:
                        DefoliationDate[i] = state.date.timetuple().tm_yday
                        if i < 4:
                            DefoliantAppRate[i + 1] = -99.9
                        DefoliationMethod[i] = 0
            if state.date.timetuple().tm_yday == DefoliationDate[i]:
                # If it is a predicted defoliation, assign tdfkgh as 2.5 .
                # Else, compute the amount intercepted by the plants in kg per ha (defkgh), and add it to tdfkgh.
                if DefoliantAppRate[i] < -99:
                    self.tdfkgh = 2.5
                else:
                    if DefoliationMethod[i] == 0:
                        self.defkgh += DefoliantAppRate[i] * 0.95 * 1.12085 * 0.75
                    else:
                        self.defkgh += DefoliantAppRate[i] * state.light_interception * 1.12085 * 0.75
                    self.tdfkgh += self.defkgh
            # If this is after the first day of defoliant application, compute the percent of leaves to be defoliated (PercentDefoliation), as a function of average daily temperature, leaf water potential, days after first defoliation application, and tdfkgh. The regression equation is modified from the equation suggested in GOSSYM.
            if DefoliationDate[i] > 0 and state.date > self.defoliate_date:
                dum = -state.min_leaf_water_potential * 10  # value of min_leaf_water_potential in bars.
                PercentDefoliation = p1 + p2 * state.average_temperature + p3 * self.tdfkgh + p4 * (state.date - self.defoliate_date).days + p5 * dum - p6 * dum * dum + p7 * state.average_temperature * self.tdfkgh * (state.date - self.defoliate_date).days * dum
                PercentDefoliation = min(max(PercentDefoliation, 0), 40)

    def _initialize_globals(self):
        # Define the numbers of rows and columns in the soil slab (nl, nk).
        # Define the depth, in cm, of consecutive nl layers.
        # NOTE: maxl and maxk are defined as constants in file "global.h".
        global nl, nk
        nl = maxl
        nk = maxk

    def _read_agricultural_input(self, inputs):
        global NumNitApps
        NumNitApps = 0
        idef = 0
        cdef NitrogenFertilizer nf
        for i in inputs:
            if "type" not in i:
                continue
            if i["type"] == "fertilization":
                nf.day = date2doy(i["date"])
                nf.amtamm = i.get("ammonium", 0)
                nf.amtnit = i.get("nitrate", 0)
                nf.amtura = i.get("urea", 0)
                nf.mthfrt = i.get("method", 0)
                isdhrz = i.get("drip_horizontal_place",
                            0)  # horizontal placement of DRIP, cm from left edge of soil slab.
                isddph = i.get("drip_depth",
                            0)  # vertical placement of DRIP, cm from soil surface.
                if nf.mthfrt == 1 or nf.mthfrt == 3:
                    nf.lsdr, nf.ksdr = self.slab_location(isdhrz, isddph)
                else:
                    nf.ksdr = 0
                    nf.lsdr = 0
                NFertilizer[NumNitApps] = nf
                NumNitApps += 1
            elif i["type"] == "defoliation prediction":
                DefoliationDate[idef] = date2doy(i["date"])
                DefoliantAppRate[idef] = -99.9
                if idef == 0:
                    self.defoliate_date = doy2date(self.start_date.year, DefoliationDate[0])
                DefoliationMethod[idef] = i.get("method", 0)
                idef += 1

    def slab_location(self, x, y):
        """Computes the layer (lsdr) or column (ksdr) where the emitter of drip irrigation, or the fertilizer side - dressing is located. It is called from ReadAgriculturalInput().

        Arguments
        ---------
        x, y
            horizontal and vertical distance

        Returns
        -------
        tuple[float, float]
            cell index
        """
        k = np.searchsorted(self.column_width_cumsum, x)
        l = np.searchsorted(self.layer_depth_cumsum, y)
        return l, k
