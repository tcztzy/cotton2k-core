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
extern double AverageSoilPsi, conmax, ElCondSatSoilToday, IrrigationDepth,
    PotGroAllBolls, PotGroAllBurrs, PotGroAllSquares, RatioImplicit;

extern double airdr[9], alpha[9], vanGenuchtenBeta[9], BulkDensity[9],
    HumusNitrogen[maxl][maxk],
    HumusOrganicMatter[maxl][maxk], MaxWaterCapacity[maxl],
    NO3FlowFraction[maxl],
    SaturatedHydCond[9],
    SitePar[21], SoilPsi[maxl][maxk],
    SoilTemp[maxl][maxk], thad[maxl], thetar[maxl], thetas[9], thts[maxl],
    VolNh4NContent[maxl][maxk], VolUreaNContent[maxl][maxk];

extern "C"
{
    double dl(unsigned int);
    double wk(unsigned int, double);
}
