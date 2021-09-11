from .climate cimport ClimateStruct
from .irrigation cimport Irrigation
from .state cimport cState

cdef extern from "Simulation.hpp":
    ctypedef struct cSimulation "Simulation":
        double row_space
        double plant_population
        double cultivar_parameters[61]
        ClimateStruct climate[400]
        Irrigation irrigation[150]
        cState states[200]

cdef extern from "global.h":
    ctypedef struct NitrogenFertilizer:
        int day
        int mthfrt
        int ksdr
        int lsdr
        double amtamm
        double amtnit
        double amtura
    void InitializeGlobal()
    const int maxl
    const int maxk
    int nl
    int nk
    double SitePar[21]
    double RatioImplicit
    double conmax
    double airdr[9]
    double thetas[9]
    double alpha[9]
    double vanGenuchtenBeta[9]
    double SaturatedHydCond[9]
    double BulkDensity[9]
    double DefoliantAppRate[5]
    double SandVolumeFraction[40]
    double ClayVolumeFraction[40]
    double thad[40]
    double FieldCapacity[40]
    double FoliageTemp[20]
    double SoilTemp[40][20]
    double LwpMinX[3]
    double LwpX[3]
    double AverageLwp
    double PotGroAllSquares
    double PotGroAllBolls
    double PotGroAllBurrs
    double PotGroLeafAreaPreFru[9]
    double PotGroLeafWeightPreFru[9]
    double PotGroPetioleWeightPreFru[9]
    double PetioleWeightPreFru[9]
    int DefoliationDate[5]
    int DefoliationMethod[5]
    double PercentDefoliation
    NitrogenFertilizer NFertilizer[150]
    int NumNitApps
    int NumIrrigations
    double PoreSpace[40]
    double SoilPsi[40][20]
    double RootImpede[40][20]
    int SoilHorizonNum[40]
    double AverageSoilPsi
    double thts[40]
    int LocationColumnDrip
    int LocationLayerDrip
    double CumNitrogenUptake
    int noitr
    double VolNh4NContent[40][20]
    double VolUreaNContent[40][20]
    double ElCondSatSoilToday
    double thetar[40]