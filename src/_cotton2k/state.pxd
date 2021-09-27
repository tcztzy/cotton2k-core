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
        double day_length
        double runoff
        double solar_noon
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
