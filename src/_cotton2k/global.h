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
typedef struct scratch
{
    double amitri, ep;
} scratch;
extern scratch Scratch21[400];
typedef struct NitrogenFertilizer
{
        int day, mthfrt, ksdr, lsdr;
        double amtamm, amtnit, amtura;
} NitrogenFertilizer;
extern NitrogenFertilizer NFertilizer[150];
////    Integers    ////
extern int DayStartPredIrrig, DayStopPredIrrig, LastIrrigation,
    LocationColumnDrip, LocationLayerDrip,
    MainStemNodes, MinDaysBetweenIrrig,
    nk, nl, noitr, NumIrrigations, NumNitApps;
extern int DefoliationDate[5], DefoliationMethod[5], SoilHorizonNum[maxl];
////    Double    ////
extern double AverageLwp, AverageSoilPsi,
    conmax, CumNitrogenUptake, CumWaterDrained, dclay,
    dsand, ElCondSatSoilToday,
    IrrigationDepth, MineralizedOrganicN,
    PercentDefoliation, PotGroAllBolls, PotGroAllBurrs, PotGroAllSquares,
    RatioImplicit, SoilNitrogenLoss, SumNO3N90;

extern double airdr[9], alpha[9], vanGenuchtenBeta[9], BulkDensity[9],
    ClayVolumeFraction[maxl],
    DefoliantAppRate[5], ElCondSatSoil[20],
    FieldCapacity[maxl], FoliageTemp[maxk],
    FreshOrganicNitrogen[maxl][maxk],
    HeatCapacitySoilSolid[maxl], HeatCondDrySoil[maxl], HumusNitrogen[maxl][maxk],
    HumusOrganicMatter[maxl][maxk], LwpMinX[3], LwpX[3],
    MarginalWaterContent[maxl], MaxWaterCapacity[maxl],
    NO3FlowFraction[maxl], PetioleWeightPreFru[9], PoreSpace[maxl],
    PotGroLeafAreaPreFru[9], PotGroLeafWeightPreFru[9], PotGroPetioleWeightPreFru[9],
    RootImpede[maxl][maxk], SandVolumeFraction[maxl], SaturatedHydCond[9],
    SitePar[21], SoilPsi[maxl][maxk],
    SoilTemp[maxl][maxk], thad[maxl], thetar[maxl], thetas[9], thts[maxl],
    VolNh4NContent[maxl][maxk], VolUreaNContent[maxl][maxk];

void InitializeGlobal();

extern "C"
{
    double dl(unsigned int);
    double wk(unsigned int, double);
}
