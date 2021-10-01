// File SoilNitrogen.cpp
//
//   functions in this file:
// SoilNitrogen()
// SoilWaterEffect();
// MineralizeNitrogen()
// Nitrification()
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
