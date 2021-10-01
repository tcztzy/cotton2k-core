//   global.h
#pragma once

#include "Irrigation.h"
//
//  definition of global variables
//  ==============================
//    For dictionary of global variables see file "global.cpp"
////    Constants    ////
const int maxl = 40;
const int maxk = 20;
const double pi = 3.14159;
////    Structures    ////
typedef struct NitrogenFertilizer
{
        int day, mthfrt, ksdr, lsdr;
        double amtamm, amtnit, amtura;
} NitrogenFertilizer;
extern NitrogenFertilizer NFertilizer[150];
////    Integers    ////
extern int LocationColumnDrip, LocationLayerDrip, MinDaysBetweenIrrig,
    nk, nl, NumIrrigations, NumNitApps;
extern int SoilHorizonNum[maxl];
////    Double    ////
extern double AverageSoilPsi, conmax, dclay, dsand, ElCondSatSoilToday, IrrigationDepth,
    PotGroAllBolls, PotGroAllBurrs, PotGroAllSquares, RatioImplicit;

extern double airdr[9], alpha[9], vanGenuchtenBeta[9], BulkDensity[9],
    ClayVolumeFraction[maxl], FieldCapacity[maxl],
    HeatCapacitySoilSolid[maxl], HeatCondDrySoil[maxl], HumusNitrogen[maxl][maxk],
    HumusOrganicMatter[maxl][maxk],
    MarginalWaterContent[maxl], MaxWaterCapacity[maxl],
    NO3FlowFraction[maxl], PoreSpace[maxl],
    SandVolumeFraction[maxl], SaturatedHydCond[9],
    SitePar[21], SoilPsi[maxl][maxk],
    SoilTemp[maxl][maxk], thad[maxl], thetar[maxl], thetas[9], thts[maxl],
    VolNh4NContent[maxl][maxk], VolUreaNContent[maxl][maxk];

extern "C"
{
    double dl(unsigned int);
    double wk(unsigned int, double);
}
