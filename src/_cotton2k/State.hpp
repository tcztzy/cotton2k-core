#ifndef STATE_TYPE
#define STATE_TYPE
#include "stdbool.h"
#include "Soil.h"
#include "FruitingSite.h"

typedef struct MainStemLeafStruct
{
    double leaf_area;
    double leaf_weight;                         // mainstem leaf weight at each node, g.
    double petiole_weight;                      // weight of mainstem leaf petiole at each node, g.
    double potential_growth_for_leaf_area;      // potential growth in area of an individual main stem node leaf, dm2 day-1.
    double potential_growth_for_leaf_weight;    // potential growth in weight of an individual main stem node leaf, g day-1.
    double potential_growth_for_petiole_weight; // potential growth in weight of an individual main stem node petiole, g day-1.
} MainStemLeaf;
typedef struct FruitingBranchStruct
{
    unsigned int number_of_fruiting_nodes; // number of nodes on each fruiting branch.
    double delay_for_new_node;             // cumulative effect of stresses on delaying the formation of a new node on a fruiting branch.
    MainStemLeaf main_stem_leaf;
    FruitingSite nodes[5];
} FruitingBranch;
typedef struct VegetativeBranchStruct
{
    unsigned int number_of_fruiting_branches; // number of fruiting branches at each vegetative branch.
    FruitingBranch fruiting_branches[30];
} VegetativeBranch;
typedef struct State
{
    double day_length;               // day length, in hours
    double green_bolls_burr_weight; // total weight of burrs in green bolls, g plant-1.
    double open_bolls_weight;       // total weight of seedcotton in open bolls, g per plant.
    double open_bolls_burr_weight;
    double reserve_carbohydrate;    // reserve carbohydrates in leaves, g per plant.
    double runoff;
    double solar_noon;
    double evapotranspiration;                  // daily sum of hourly reference evapotranspiration, mm per day.
    double actual_transpiration;                // actual transpiration from plants, mm day-1.
    double potential_evaporation;               //
    double actual_soil_evaporation;             // actual evaporation from soil surface, mm day-1.
    unsigned int number_of_vegetative_branches; // number of vegetative branches (including the main branch), per plant.
    double number_of_squares;                   // number of squares per plant.
    double number_of_green_bolls;               // average number of retained green bolls, per plant.
    double number_of_open_bolls;                // number of open bolls, per plant.
    double nitrogen_stress;                     // the average nitrogen stress coefficient for vegetative and reproductive organs
    double nitrogen_stress_vegetative;          // nitrogen stress limiting vegetative development.
    double nitrogen_stress_fruiting;            // nitrogen stress limiting fruit development.
    double nitrogen_stress_root;                // nitrogen stress limiting root development.
    double total_required_nitrogen;             // total nitrogen required for plant growth, g per plant.
    double leaf_area_index;                     // Leaf area index
    double leaf_area;
    double leaf_weight;
    double leaf_weight_pre_fruiting[9];         // weight of prefruiting node leaves, g.
    double leaf_weight_area_ratio;              // temperature dependent factor for converting leaf area to leaf weight during the day, g dm-1
    double leaf_nitrogen_concentration;         // average nitrogen concentration in leaves.
    double leaf_nitrogen;                       // total leaf nitrogen, g per plant.
    double petiole_nitrogen_concentration;      // average nitrogen concentration in petioles.
    double seed_nitrogen_concentration;         // average nitrogen concentration in seeds.
    double seed_nitrogen;                       // total seed nitrogen, g per plant.
    double root_nitrogen_concentration;         // average nitrogen concentration in roots.
    double root_nitrogen;                       // total root nitrogen, g per plant.
    double square_nitrogen_concentration;       // average concentration of nitrogen in the squares.
    double burr_nitrogen_concentration;         // average nitrogen concentration in burrs.
    double burr_nitrogen;                       // nitrogen in burrs, g per plant.
    double square_nitrogen;                     // total nitrogen in the squares, g per plant
    double stem_nitrogen_concentration;         // ratio of stem nitrogen to dry matter.
    double stem_nitrogen;                       // total stem nitrogen, g per plant
    double fruit_growth_ratio;                  // ratio between actual and potential square and boll growth.
    double ginning_percent;                     // weighted average ginning percentage of all open bolls.
    double deep_soil_temperature;               // boundary soil temperature of deepest layer (K)
    double total_actual_leaf_growth;            // actual growth rate of all the leaves, g plant-1 day-1.
    double total_actual_petiole_growth;         // actual growth rate of all the petioles, g plant-1 day-1.
    double actual_square_growth;                // total actual growth of squares, g plant-1 day-1.
    double actual_stem_growth;                  // actual growth rate of stems, g plant-1 day-1.
    double actual_boll_growth;                  // total actual growth of seedcotton in bolls, g plant-1 day-1.
    double actual_burr_growth;                  // total actual growth of burrs in bolls, g plant-1 day-1.
    double supplied_nitrate_nitrogen;           // uptake of nitrate by the plant from the soil, mg N per slab per day.
    double supplied_ammonium_nitrogen;          // uptake of ammonia N by the plant from the soil, mg N per slab per day.
    double petiole_nitrogen;                    // total petiole nitrogen, g per plant.
    double petiole_nitrate_nitrogen_concentration; // average nitrate nitrogen concentration in petioles.
    bool pollination_switch;                    // pollination switch: false = no pollination, true = yes.
    double age_of_pre_fruiting_nodes[9];        // age of each prefruiting node, physiological days.
    int number_of_pre_fruiting_nodes;           // number of prefruiting nodes, per plant.
    double leaf_area_pre_fruiting[9];           // area of prefruiting node leaves, dm2.
    double delay_for_new_branch[3];
    VegetativeBranch vegetative_branches[3];
    Soil soil;
} State;
#endif
