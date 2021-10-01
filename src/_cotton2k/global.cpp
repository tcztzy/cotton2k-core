#include "global.h"

struct NitrogenFertilizer NFertilizer[150]; // nitrogen fertilizer application information for each day.
// int day = date of application (DOY)
// int mthfrt = method of application ( 0 = broadcast; 1 = sidedress; 2 = foliar; 3 = drip fertigation);
// int ksdr = horizontal placement of side-dressed fertilizer, cm.
// int lsdr = vertical placement of side-dressed fertilizer, cm.
// double amtamm = ammonium N applied, kg N per ha;
// double amtnit = nitrate N applied, kg N per ha;
// double amtura = urea N applied, kg N per ha;
//
// Integer variables:
//

int LocationColumnDrip,  // number of column in which the drip emitter is located
    LocationLayerDrip,   // number of layer in which the drip emitter is located.
    MinDaysBetweenIrrig, // minimum number of days between consecutive irrigations (used for computing predicted irrigation).
    nk,                  // number of vertical columns of soil cells in the slab.
    nl,                  // number of horizontal layers of soil cells in the slab.
    NumIrrigations,      // number of irrigations.
    NumNitApps;          // number of applications of nitrogen fertilizer.

int SoilHorizonNum[maxl];  // the soil horizon number associated with each soil layer in the slab.

double
    airdr[9],                         // volumetric water content of soil at "air-dry" for each soil horizon, cm3 cm-3.
    alpha[9],                         // parameter of the Van Genuchten equation.
    AverageSoilPsi,                   // average soil matric water potential, bars, computed as the weighted average of the root zone.
    vanGenuchtenBeta[9],              // parameter of the Van Genuchten equation.
    BulkDensity[9],                   // bulk density of soil in a horizon, g cm-3.
    ClayVolumeFraction[maxl],         // fraction by volume of clay in the soil.
    conmax,                           // the maximum value for non-dimensional hydraulic conductivity
    dclay,                            // aggregation factor for clay in water.
    dsand,                            // aggregation factor for sand in water.
    ElCondSatSoilToday,               // electrical conductivity of saturated extract (mmho/cm) on this day.
    FieldCapacity[maxl],              // volumetric water content of soil at field capacity for each soil layer, cm3 cm-3.
    HeatCapacitySoilSolid[maxl],      // heat capacity of the solid phase of the soil.
    HeatCondDrySoil[maxl],            // the heat conductivity of dry soil.
    HumusNitrogen[maxl][maxk],        // N in stable humic fraction material in a soil cells, mg/cm3.
    HumusOrganicMatter[maxl][maxk],   // humus fraction of soil organic matter, mg/cm3.
    IrrigationDepth,                  // depth of predicted irrigation, cm.
    MarginalWaterContent[maxl],       // marginal soil water content (as a function of soil texture) for computing soil heat conductivity.
    MaxWaterCapacity[maxl],           // volumetric water content of a soil layer at maximum capacity, before drainage, cm3 cm-3.
    NO3FlowFraction[maxl],            // fraction of nitrate that can move to the next layer.
    PoreSpace[maxl],                  // pore space of soil, volume fraction.
    PotGroAllBolls,                   // sum of potential growth rates of seedcotton in all bolls, g plant-1 day-1.
    PotGroAllBurrs,                   // sum of potential growth rates of burrs in all bolls, g plant-1 day-1.
    PotGroAllSquares,                 // sum of potential growth rates of all squares, g plant-1 day-1.
    RatioImplicit,                    // the ratio for the implicit numerical solution of the water transport equation (used in FLUXI and in SFLUX.
    SandVolumeFraction[maxl],         // fraction by volume of sand plus silt in the soil.
    SaturatedHydCond[9],              // saturated hydraulic conductivity, cm per day.
    SitePar[21],                      // array of site specific constant parameters.
    SoilPsi[maxl][maxk],              // matric water potential of a soil cell, bars.
    SoilTemp[maxl][maxk],             // hourly soil temperature oK.
    thad[maxl],                       // residual volumetric water content of soil layers (at air-dry condition), cm3 cm-3.
    thetar[maxl],                     // volumetric water content of soil layers at permanent wilting point (-15 bars), cm3 cm-3.
    thetas[9],                        // volumetric saturated water content of soil horizon, cm3 cm-3.
    thts[maxl],                       // saturated volumetric water content of each soil layer, cm3 cm-3.
    VolNh4NContent[maxl][maxk],       // volumetric ammonium nitrogen content of a soil cell, mg N cm-3.
    VolUreaNContent[maxl][maxk];      // volumetric urea nitrogen content of a soil cell, mg N cm-3.
