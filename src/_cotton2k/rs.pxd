from libc.stdint cimport uint32_t, int32_t
from .fruiting_site cimport Stage

cdef extern:
    double wk(unsigned int, double)
    int SlabLoc(int, double)
    double SoilMechanicResistance(double)
    double wcond(double, double, double, double, double, double)
    double PsiOsmotic(double, double, double)
    double psiq(double, double, double, double, double)
    double qpsi(double, double, double, double, double)
    double form(double, double, double)
    double SoilWaterEffect(double, double, double, double, double)
    double SoilTemperatureEffect(double)
