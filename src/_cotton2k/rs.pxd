from libc.stdint cimport uint32_t, int32_t
from .fruiting_site cimport Stage

cdef extern:
    double wk(unsigned int, double)
    double dl(unsigned int)
    int SlabLoc(int, double)
    double SoilMechanicResistance(double)
    double PsiOsmotic(double, double, double)
    double form(double, double, double)
    double SoilTemperatureEffect(double)
