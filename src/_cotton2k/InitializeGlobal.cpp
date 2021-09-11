// File InitializeGlobal.cpp
//
#include "global.h"

///////////////////////////////////////////////////////////////////////////
void InitializeGlobal()
//     This function initializes many "global" variables at the start of a
//  simulation. It is called from ReadInput(). Note that initialization
//  is needed at the start of each simulation (NOT at start of the run).
{
    AverageLwp = 0;

    CumNitrogenUptake = 0;
    CumWaterDrained = 0;

    LastIrrigation = 0;

    MineralizedOrganicN = 0;

    PercentDefoliation = 0;

    SoilNitrogenLoss = 0;
    SumNO3N90 = 0;
//
    for (int i = 0; i < 3; i++) {
        LwpMinX[i] = 0;
        LwpX[i] = 0;
    }
//
    for (int i = 0; i < 5; i++) {
        DefoliationDate[i] = 0;
        DefoliationMethod[i] = 0;
        DefoliantAppRate[i] = 0;
    }
//
    for (int i = 0; i < 9; i++) {
        PotGroLeafAreaPreFru[i] = 0;
        PotGroLeafWeightPreFru[i] = 0;
        PotGroPetioleWeightPreFru[i] = 0;
        PetioleWeightPreFru[i] = 0;
    }
//
    for (int i = 0; i < 20; i++) {
        ElCondSatSoil[i] = 0;
    }
//
    for (int k = 0; k < maxk; k++) {
        FoliageTemp[k] = 295;
    }
//
    for (int l = 0; l < maxl; l++) {
        for (int k = 0; k < maxk; k++) {
            RootImpede[l][k] = 0;
        }
    }
}
//////////////////////////////////////////////////////////
