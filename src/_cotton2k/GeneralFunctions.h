//  General auxiliary functions
#include <cinttypes>

using namespace std;

extern "C"
{
    double PsiOsmotic(double q, double qsat, double ec);
    double qpsi(double psi, double qr, double qsat, double alpha, double beta);
    double psiq(double q, double qr, double qsat, double alpha, double beta);
    double wcond(double q, double qr, double qsat, double beta, double SaturatedHydCond, double PoreSpace);
    double form(double c0, double d0, double g0);
}
