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
    double thad[40]
    double SoilTemp[40][20]
    double PotGroAllSquares
    double PotGroAllBolls
    double PotGroAllBurrs
    NitrogenFertilizer NFertilizer[150]
    int NumNitApps
    int NumIrrigations
    double SoilPsi[40][20]
    int SoilHorizonNum[40]
    double AverageSoilPsi
    double thts[40]
    int LocationColumnDrip
    int LocationLayerDrip
    double VolNh4NContent[40][20]
    double VolUreaNContent[40][20]
    double ElCondSatSoilToday
    double thetar[40]
    double HumusOrganicMatter[40][20]
    double NO3FlowFraction[40]
    double MaxWaterCapacity[40]
    double HumusNitrogen[40][20]
