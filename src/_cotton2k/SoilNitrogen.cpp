// File SoilNitrogen.cpp
//
//   functions in this file:
// SoilNitrogen()
// SoilWaterEffect();
// MineralizeNitrogen()
// Nitrification()
// Denitrification()
//
#include <cmath>
#include "global.h"
#include "Simulation.hpp"

using namespace std;

extern "C"
{
    double SoilWaterEffect(double, double, double, double, double);
}

//////////////////////////////////
void Nitrification(SoilCell &soil_cell, int l, int k, double DepthOfLayer, double soil_temperature)
//     This function computes the transformation of soil ammonia nitrogen to nitrate.
//  It is called by SoilNitrogen(). It calls the function SoilWaterEffect()
//
//     The following global variables are set here:   VolNh4NContent, VolNo3NContent
//     The following arguments are used:
//  DepthOfLayer - depth to the bottom of this layer, cm.
//  k, l - soil column and layer numbers.
//
{
//     The following constant parameters are used:
    const double cpardepth = 0.45;
    const double cparnit1 = 24.635;
    const double cparnit2 = 8227;
    const double cparsanc = 204; // this constant parameter is modified from kg/ha units in CERES
    // to mg/cm3 units of VolNh4NContent (assuming 15 cm layers)
    double sanc; // effect of NH4 N in the soil on nitrification rate (0 to 1).
    if (VolNh4NContent[l][k] < 0.1)
        sanc = 1 - exp(-cparsanc * VolNh4NContent[l][k]);
    else
        sanc = 1;
//     The rate of nitrification, con1, is a function of soil temperature. It is slightly
//  modified from GOSSYM. it is transformed from immediate rate to a daily time step ratenit.
//     The rate is modified by soil depth, assuming that for an increment
//  of 30 cm depth, the rate is decreased by 55% (multiply by a power of
//  cpardepth). It is also multiplied by the environmental limiting
//  factors (sanc, SoilWaterEffect) to get the actual rate of nitrification.
//     The maximum rate is assumed not higher than 10%.
    double con1;    // rate of nitrification as a function of temperature.
    con1 = exp(cparnit1 - cparnit2 / soil_temperature);
    double ratenit; // actual rate of nitrification (day-1).
    ratenit = 1 - exp(-con1);
    double tff; // effect of soil depth on nitrification rate.
    tff = (DepthOfLayer - 30) / 30;
    if (tff < 0)
        tff = 0;
//     Add the effects of NH4 in soil, soil water content, and depth of soil layer.
    ratenit = ratenit * sanc * SoilWaterEffect(soil_cell.water_content, FieldCapacity[l], thetar[l], thts[l], 1) * pow(cpardepth, tff);
    if (ratenit < 0)
        ratenit = 0;
    if (ratenit > 0.10)
        ratenit = 0.10;
// Compute the actual amount of N nitrified, and update VolNh4NContent and VolNo3NContent.
    double dnit; // actual nitrification (mg n cm-3 day-1).
    dnit = ratenit * VolNh4NContent[l][k];
    VolNh4NContent[l][k] -= dnit;
    soil_cell.nitrate_nitrogen_content += dnit;
}

/////////////////////////
void Denitrification(SoilCell &soil_cell, int l, int k, double row_space, double soil_temperature)
//     This function computes the denitrification of nitrate N in the soil.
//     It is called by function SoilNitrogen().
//     The procedure is based on the CERES routine, as documented by Godwin and Jones (1991).
//
//     The following global variables are referenced here:
//       dl, FieldCapacity, HumusOrganicMatter, thts
//    The following global variables are set here:     SoilNitrogenLoss, VolNo3NContent
{
//    The following constant parameters are used:
    const double cpar01 = 24.5;
    const double cpar02 = 3.1;
    const double cpardenit = 0.00006;
    const double cparft = 0.046;
    const double cparhum = 0.58;
    const double vno3min = 0.00025;
//
    double soilc; // soil carbon content, mg/cm3.
//     soilc is calculated as 0.58 (cparhum) of the stable humic fraction
//  (following CERES), and cw is estimated following Rolston et al. (1980).
    soilc = cparhum * HumusOrganicMatter[l][k];
    double cw;    // water soluble carbon content of soil, ppm.
    cw = cpar01 + cpar02 * soilc;
//     The effects of soil moisture (fw) and soil temperature (ft) are computed as 0 to 1 factors.
    double fw; // effect of soil moisture on denitrification rate.
    fw = (soil_cell.water_content - FieldCapacity[l]) / (thts[l] - FieldCapacity[l]);
    if (fw < 0)
        fw = 0;
    double ft; // effect of soil temperature on denitrification rate.
    ft = 0.1 * exp(cparft * (soil_temperature - 273.161));
    if (ft > 1)
        ft = 1;
//     The actual rate of denitrification is calculated. The equation is modified from CERES to
//  units of mg/cm3/day.
    double dnrate; // actual rate of denitrification, mg N per cm3 of soil per day.
    dnrate = cpardenit * cw * soil_cell.nitrate_nitrogen_content * fw * ft;
//     Make sure that a minimal amount of nitrate will remain after denitrification.
    if (dnrate > (soil_cell.nitrate_nitrogen_content - vno3min))
        dnrate = soil_cell.nitrate_nitrogen_content - vno3min;
    if (dnrate < 0)
        dnrate = 0;
//     Update VolNo3NContent, and add the amount of nitrogen lost to SoilNitrogenLoss.
    soil_cell.nitrate_nitrogen_content -= dnrate;
}
