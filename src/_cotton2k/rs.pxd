from libc.stdint cimport uint32_t, int32_t
from .fruiting_site cimport Stage

cdef extern:
    double dl(unsigned int)
    double wk(unsigned int, double)
    double tdewest(double, double, double)
    int SlabLoc(int, double)
    double daywnd(double, double, double, double, double, double)
    double TemperatureOnFruitGrowthRate(double)
    double SoilNitrateOnRootGrowth(double)
    double SoilMechanicResistance(double)
    double wcond(double, double, double, double, double, double)
    double PsiOsmotic(double, double, double)
    double psiq(double, double, double, double, double)
    double qpsi(double, double, double, double, double)
    double PsiOnTranspiration(double)