from libcpp cimport bool as bool_t
from .fruiting_site cimport FruitingSite
from .soil cimport cSoil

cdef extern from "State.hpp":

    ctypedef struct cMainStemLeaf "MainStemLeaf":
        double leaf_area
        double leaf_weight
        double petiole_weight
        double potential_growth_for_leaf_area
        double potential_growth_for_leaf_weight
        double potential_growth_for_petiole_weight

    ctypedef struct cFruitingBranch "FruitingBranch":
        unsigned int number_of_fruiting_nodes
        double delay_for_new_node
        cMainStemLeaf main_stem_leaf
        FruitingSite nodes[5]

    ctypedef struct cVegetativeBranch "VegetativeBranch":
        unsigned int number_of_fruiting_branches
        cFruitingBranch fruiting_branches[30]

    ctypedef struct cState "State":
        double day_inc
        double cumulative_nitrogen_loss
        double carbon_stress
        double day_length
        double petiole_weight
        double square_weight
        double green_bolls_weight
        double green_bolls_burr_weight
        double open_bolls_weight
        double open_bolls_burr_weight
        double reserve_carbohydrate
        double runoff
        double solar_noon
        double evapotranspiration
        double actual_transpiration
        double potential_evaporation
        double actual_soil_evaporation
        unsigned int number_of_vegetative_branches
        double number_of_squares
        double number_of_green_bolls
        double number_of_open_bolls
        double nitrogen_stress
        double nitrogen_stress_vegetative
        double nitrogen_stress_fruiting
        double nitrogen_stress_root
        double total_required_nitrogen
        double leaf_area_index
        double leaf_area
        double leaf_weight
        double leaf_weight_pre_fruiting[9]
        double leaf_weight_area_ratio
        double leaf_nitrogen_concentration
        double leaf_nitrogen
        double petiole_nitrogen_concentration
        double seed_nitrogen_concentration
        double seed_nitrogen
        double root_nitrogen_concentration
        double root_nitrogen
        double square_nitrogen_concentration
        double burr_nitrogen_concentration
        double burr_nitrogen
        double square_nitrogen
        double stem_nitrogen_concentration
        double stem_nitrogen
        double fruit_growth_ratio
        double ginning_percent
        double deep_soil_temperature
        double total_actual_leaf_growth
        double total_actual_petiole_growth
        double actual_square_growth
        double actual_stem_growth
        double actual_boll_growth
        double actual_burr_growth
        double supplied_nitrate_nitrogen
        double supplied_ammonium_nitrogen
        double petiole_nitrogen
        double petiole_nitrate_nitrogen_concentration
        bool_t pollination_switch
        double age_of_pre_fruiting_nodes[9]
        int number_of_pre_fruiting_nodes
        double leaf_area_pre_fruiting[9]
        double delay_for_new_branch[3]
        cVegetativeBranch vegetative_branches[3]
        cSoil soil


cdef class StateBase:
    cdef cState *_
    cdef public unsigned int seed_layer_number  # layer number where the seeds are located.
    cdef public unsigned int taproot_layer_number  # last soil layer with taproot.
    cdef public unsigned int year
    cdef public unsigned int version
    cdef public unsigned int kday
    cdef unsigned int _ordinal
    cdef public double average_min_leaf_water_potential  #running average of min_leaf_water_potential for the last 3 days.
    cdef public double average_temperature  # average daily temperature, C, for 24 hours.
    cdef public double carbon_allocated_for_root_growth  # available carbon allocated for root growth, g per plant.
    cdef public double daytime_temperature  # average day-time temperature, C.
    cdef public double delay_of_emergence  # effect of negative values of xt on germination rate.
    cdef public double extra_carbon  # Extra carbon, not used for plant potential growth requirements, assumed to accumulate in taproot.
    cdef public double fiber_length
    cdef public double fiber_strength
    cdef public double hypocotyl_length  # length of hypocotyl, cm.
    cdef public double leaf_potential_growth  # sum of potential growth rates of all leaves, g plant-1 day-1.
    cdef public double light_interception  # ratio of light interception by plant canopy.
    cdef public double lint_yield  # yield of lint, kgs per hectare.
    cdef public double max_leaf_water_potential  # maximum (dawn) leaf water potential, MPa.
    cdef public double min_leaf_water_potential  # minimum (noon) leaf water potential, MPa.
    cdef public double net_photosynthesis  # net photosynthetic rate, g per plant per day.
    cdef public double net_radiation  # daily total net radiation, W m-2.
    cdef public double nighttime_temperature  # average night-time temperature, C.
    cdef public double pavail  # residual available carbon for root growth from previous day.
    cdef public double petiole_potential_growth  # sum of potential growth rates of all petioles, g plant-1 day-1.
    cdef public double plant_height
    cdef public double root_potential_growth  # potential growth rate of roots, g plant-1 day-1
    cdef public double seed_moisture  # moisture content of germinating seeds, percent.
    cdef public double stem_potential_growth  # potential growth rate of stems, g plant-1 day-1.
    cdef public double stem_weight  # total stem weight, g per plant.
    cdef public double taproot_length  # the length of the taproot, in cm.
    cdef public double water_stress  # general water stress index (0 to 1).
    cdef public double water_stress_stem  # water stress index for stem growth (0 to 1).
    cdef public double last_layer_with_root_depth  # the depth to the end of the last layer with roots (cm).
