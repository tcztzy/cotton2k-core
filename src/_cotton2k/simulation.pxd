from libc.stdint cimport uint32_t

from .cxx cimport (
BulkDensity,
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
    double tdewhour(cSimulation &, uint32_t, double, double, double, double, double, double, double, double)
    double SimulateRunoff(cSimulation &, uint32_t, double, double, uint32_t)
