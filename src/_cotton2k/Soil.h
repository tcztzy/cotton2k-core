#ifndef SOIL_TYPE
#define SOIL_TYPE

typedef struct SoilCellStruct
{
    double nitrate_nitrogen_content; // volumetric nitrate nitrogen content of a soil cell, mg N cm-3.
    double fresh_organic_matter;     // fresh organic matter in the soil, mg / cm3.
    double water_content;            // volumetric water content of a soil cell, cm3 cm-3.
} SoilCell;

typedef struct SoilLayerStruct
{
    unsigned int number_of_left_columns_with_root;  // first column with roots in a soil layer.
    unsigned int number_of_right_columns_with_root; // last column with roots in a soil layer.
} SoilLayer;

typedef struct SoilStruct
{
    unsigned int number_of_layers_with_root;
    SoilLayer layers[40];
    SoilCell cells[40][20];
} Soil;
#endif
