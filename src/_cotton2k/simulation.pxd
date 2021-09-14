from libc.stdint cimport uint32_t

from .cxx cimport (
BulkDensity,
DefoliantAppRate,
DefoliationDate,
DefoliationMethod,
FieldCapacity,
PercentDefoliation,
InitializeGlobal,
NFertilizer,
NitrogenFertilizer,
NumIrrigations,
NumNitApps,
RatioImplicit,
SaturatedHydCond,
SitePar,
SoilTemp,
airdr,
alpha,
conmax,
maxk,
maxl,
nk,
nl,
thad,
thetas,
vanGenuchtenBeta,
cSimulation,
)
from .fruiting_site cimport Stage, FruitingSite
from .state cimport cState
from .soil cimport cSoilCell, cSoil

cdef extern:
    double daytmp(cSimulation &, uint32_t, double, double, double, double)
    double tdewhour(cSimulation &, uint32_t, double, double, double, double, double, double, double, double)
    double SimulateRunoff(cSimulation &, uint32_t, double, double, uint32_t)

cdef extern from "SoilNitrogen.h":
    void UreaHydrolysis(cSoilCell &, int, int, double)
    void MineralizeNitrogen(cSoilCell &, int, int, const int &, const int &, double, double)
    void Nitrification(cSoilCell &, int, int, double, double)
    void Denitrification(cSoilCell &, int, int, double, double)

cdef extern from "SoilProcedures.h":
    void GravityFlow(cSoilCell[40][20], double, double)
    void CapillaryFlow(cSimulation &, unsigned int)
    void DripFlow(cSoilCell[40][20], double, double)
    void NitrogenUptake(cState &, cSoilCell &, int, int, double, double, double)

cdef extern from "SoilTemperature.h":
    void SoilHeatFlux(cState &, double, int, int, int, int, double)
    double ThermalCondSoil(double, double, int)
