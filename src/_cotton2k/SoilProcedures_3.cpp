//  SoilProcedures_3.cpp
//
//   functions in this file:
// WaterFlux()
// WaterBalance()
// NitrogenFlow()
//
#include <cmath>
#include "global.h"
#include "GeneralFunctions.h"
#include "Simulation.hpp"

void WaterBalance(double[], double[], double [], int);

////////////////////////////////////////////////////////////////////////////////
void WaterFlux(double q1[], double psi1[], double dd[], double qr1[],
               double qs1[], double pp1[], int nn, int iv, int ll, long numiter, int noitr)
//     This function computes the movement of water in the soil, caused by potential differences
//  between cells in a soil column or in a soil layer. It is called by function
//  CapillaryFlow(). It calls functions WaterBalance(), psiq(), qpsi() and wcond().
//
//     The following arguments are used:
//       q1 = array of volumetric water content, v/v.
//       psi1 = array of matric soil water potential, bars.
//       dd1 = array of widths of soil cells in the direction of flow, cm.
//       qr1 = array of residual volumetric water content.
//       qs1 = array of saturated volumetric water content.
//       pp1 = array of pore space v/v.
//       nn = number of cells in the array.
//       iv = indicator if flow direction, iv = 1 for vertical iv = 0 for horizontal.
//       ll = layer number if the flow is horizontal.
//       numiter = counter for the number of iterations.
//
//     Global variables referenced:
//       alpha, vanGenuchtenBeta, RatioImplicit, SaturatedHydCond, SoilHorizonNum.
//
{
    double delt = 1 / (double) noitr; // the time step of this iteration (fraction of day)
    double cond[40]; // values of hydraulic conductivity
    double kx[40]; // non-dimensional conductivity to the lower layer or to the column on the right
    double ky[40]; // non-dimensional conductivity to the upper layer or to the column on the left
//     Loop over all soil cells. if this is a vertical flow, define the profile index j
//  for each soil cell. compute the hydraulic conductivity of each soil cell, using the
//  function wcond(). Zero the arrays kx and ky.
    int j = SoilHorizonNum[ll]; // for horizontal flow (iv = 0)
    for (int i = 0; i < nn; i++) {
        if (iv == 1)
            j = SoilHorizonNum[i]; // for vertical flow
        cond[i] = wcond(q1[i], qr1[i], qs1[i], vanGenuchtenBeta[j], SaturatedHydCond[j], pp1[i]);
        kx[i] = 0;
        ky[i] = 0;
    }
//     Loop from the second soil cell. compute the array dy (distances
//  between the midpoints of adjacent cells).
//     Compute the average conductivity avcond[i] between soil cells
//  i and (i-1). for low values of conductivity in both cells,(( an
//  arithmetic mean is computed)). for higher values the harmonic mean is
//  used, but if in one of the cells the conductivity is too low (less
//  than a minimum value of condmin ), replace it with condmin.
//
    double dy[40]; // the distance between the midpoint of a layer (or a column) and the midpoint
    // of the layer above it (or the column to the left of it)
    double avcond[40]; // average hydraulic conductivity of two adjacent soil cells
    double condmin = 0.000006;  // minimum value of conductivity, used for computing averages
    for (int i = 1; i < nn; i++) {
        dy[i] = 0.5 * (dd[i - 1] + dd[i]);
        if (cond[i - 1] <= condmin && cond[i] <= condmin)
            avcond[i] = condmin;
        else if (cond[i - 1] <= condmin && cond[i] > condmin)
            avcond[i] = 2 * condmin * cond[i] / (condmin + cond[i]);
        else if (cond[i] <= condmin && cond[i - 1] > condmin)
            avcond[i] = 2 * condmin * cond[i - 1] / (condmin + cond[i - 1]);
        else
            avcond[i] = 2 * cond[i - 1] * cond[i] / (cond[i - 1] + cond[i]);
    }
//     The numerical solution of the flow equation is a combination of the implicit method
//  (weighted by RatioImplicit) and the explicit method (weighted by 1-RatioImplicit).
//     Compute the explicit part of the solution, weighted by (1-RatioImplicit).
//  store water content values, before changing them, in array qx.
    double qx[40]; // previous value of q1.
    double addq[40]; // water added to qx
    double sumaddq = 0; // sum of addq
    for (int i = 0; i < nn; i++)
        qx[i] = q1[i];
//     Loop from the second to the last but one soil cells.
    for (int i = 1; i < nn - 1; i++) {
//     Compute the difference in soil water potential between adjacent cells (deltpsi).
//  This difference is not allowed to be greater than 1000 bars, in order to prevent computational
//  overflow in cells with low water content.
        double deltpsi = psi1[i - 1] - psi1[i]; // difference of soil water potentials (in bars)
        // between adjacent soil soil cells
        if (deltpsi > 1000)
            deltpsi = 1000;
        if (deltpsi < -1000)
            deltpsi = -1000;
//     If this is a vertical flux, add the gravity component of water potential.
        if (iv == 1)
            deltpsi += 0.001 * dy[i];
//     Compute dumm1 (the hydraulic conductivity redimensioned to cm), and check that it will
//  not exceed conmax multiplied by the distance between soil cells, in order to prevent
//  overflow errors.
        double dumm1; // redimensioned hydraulic conductivity components between adjacent cells.
        dumm1 = 1000 * avcond[i] * delt / dy[i];
        if (dumm1 > conmax * dy[i])
            dumm1 = conmax * dy[i];
//     Water entering soil cell i is now computed, weighted by (1 - RatioImplicit).
//  It is not allowed to be greater than 25% of the difference between the cells.
//     Compute water movement from soil cell i-1 to i:
        double dqq1; // water added to cell i from cell (i-1)
        dqq1 = (1 - RatioImplicit) * deltpsi * dumm1;
        double deltq; // difference of soil water content (v/v) between adjacent cells.
        deltq = qx[i - 1] - qx[i];
        if (fabs(dqq1) > fabs(0.25 * deltq)) {
            if (deltq > 0 && dqq1 < 0)
                dqq1 = 0;
            else if (deltq < 0 && dqq1 > 0)
                dqq1 = 0;
            else
                dqq1 = 0.25 * deltq;
        }
//     This is now repeated for water movement from i+1 to i.
        deltpsi = psi1[i + 1] - psi1[i];
        deltq = qx[i + 1] - qx[i];
        if (deltpsi > 1000)
            deltpsi = 1000;
        if (deltpsi < -1000)
            deltpsi = -1000;
        if (iv == 1)
            deltpsi -= 0.001 * dy[i + 1];
        dumm1 = 1000 * avcond[i + 1] * delt / dy[i + 1];
        if (dumm1 > (conmax * dy[i + 1]))
            dumm1 = conmax * dy[i + 1];
        double dqq2 = (1 - RatioImplicit) * deltpsi * dumm1; // water added to cell i from cell (i+1)
        if (fabs(dqq2) > fabs(0.25 * deltq)) {
            if (deltq > 0 && dqq2 < 0)
                dqq2 = 0;
            else if (deltq < 0 && dqq2 > 0)
                dqq2 = 0;
            else
                dqq2 = 0.25 * deltq;
        }
        addq[i] = (dqq1 + dqq2) / dd[i];
        sumaddq += dqq1 + dqq2;
//     Water content of the first and last soil cells is
//  updated to account for flow to or from their adjacent soil cells.
        if (i == 1) {
            addq[0] = -dqq1 / dd[0];
            sumaddq -= dqq1;
        }
        if (i == nn - 2) {
            addq[nn - 1] = -dqq2 / dd[nn - 1];
            sumaddq -= dqq2;
        }
    }
//     Water content q1[i] and soil water potential psi1[i] are updated.
    for (int i = 0; i < nn; i++) {
        q1[i] = qx[i] + addq[i];
        if (iv == 1)
            j = SoilHorizonNum[i];
        psi1[i] = psiq(q1[i], qr1[i], qs1[i], alpha[j], vanGenuchtenBeta[j]);
    }
//     Compute the implicit part of the solution, weighted by RatioImplicit, starting
//  loop from the second cell.
    for (int i = 1; i < nn; i++) {
//     Mean conductivity (avcond) between adjacent cells is made "dimensionless" (ky) by
//  multiplying it by the time step (delt)and dividing it by cell length (dd) and by dy.
//  It is also multiplied by 1000 for converting the potential differences from bars to cm.
        ky[i] = 1000 * avcond[i] * delt / (dy[i] * dd[i]);
//     Very low values of ky are converted to zero, to prevent underflow computer errors, and
//  very high values are converted to maximum limit (conmax), to prevent overflow errors.
        if (ky[i] < 0.0000001)
            ky[i] = 0;
        if (ky[i] > conmax)
            ky[i] = conmax;
    }
//     ky[i] is the conductivity between soil cells i and i-1, whereas kx[i] is between i and i+1.
//  Another loop, until the last but one soil cell, computes kx in a similar manner.
    for (int i = 0; i < nn - 1; i++) {
        kx[i] = 1000 * avcond[i + 1] * delt / (dy[i + 1] * dd[i]);
        if (kx[i] < 0.0000001) kx[i] = 0;
        if (kx[i] > conmax) kx[i] = conmax;
    }
//     Arrays used for the implicit numeric solution:
    double a1[40], b1[40], cau[40], cc1[40], d1[40], dau[40];
    for (int i = 0; i < nn; i++) {
//     Arrays a1, b1, and cc1 are computed for the implicit part of
//  the solution, weighted by RatioImplicit.
        a1[i] = -kx[i] * RatioImplicit;
        b1[i] = 1 + RatioImplicit * (kx[i] + ky[i]);
        cc1[i] = -ky[i] * RatioImplicit;
        if (iv == 1) {
            j = SoilHorizonNum[i];
            a1[i] = a1[i] - 0.001 * kx[i] * RatioImplicit;
            cc1[i] = cc1[i] + 0.001 * ky[i] * RatioImplicit;
        }
//     The water content of each soil cell is converted to water
//  potential by function psiq and stored in array d1 (in bar units).
        d1[i] = psiq(q1[i], qr1[i], qs1[i], alpha[j], vanGenuchtenBeta[j]);
    }
//     The solution of the simultaneous equations in the implicit method alternates between
//  the two directions along the arrays. The reason for this is because the direction of the
//  solution may cause some cumulative bias. The counter numiter determines the direction
//  of the solution.
//     The solution in this section starts from the last soil cell (nn).
    if ((numiter % 2) == 0) {
//     Intermediate arrays dau and cau are computed.
        cau[nn - 1] = psi1[nn - 1];
        dau[nn - 1] = 0;
        for (int i = nn - 2; i > 0; i--) {
            double p = a1[i] * dau[i + 1] + b1[i]; // temporary
            dau[i] = -cc1[i] / p;
            cau[i] = (d1[i] - a1[i] * cau[i + 1]) / p;
        }
        if (iv == 1)
            j = SoilHorizonNum[0];
        psi1[0] = psiq(q1[0], qr1[0], qs1[0], alpha[j], vanGenuchtenBeta[j]);
//     psi1 is now computed for soil cells 1 to nn-2. q1 is
//  computed from psi1 by function qpsi.
        for (int i = 1; i < nn - 1; i++) {
            if (iv == 1)
                j = SoilHorizonNum[i];
            psi1[i] = dau[i] * psi1[i - 1] + cau[i];
            q1[i] = qpsi(psi1[i], qr1[i], qs1[i], alpha[j], vanGenuchtenBeta[j]);
        }
    }
//     The alternative direction of solution is executed here. the
//  solution in this section starts from the first soil cell.
    else {
//     Intermediate arrays dau and cau are computed, and the computations
//  described previously are repeated in the opposite direction.
        cau[0] = psi1[0];
        dau[0] = 0;
        for (int i = 1; i < nn - 1; i++) {
            double p = a1[i] * dau[i - 1] + b1[i]; // temporary
            dau[i] = -cc1[i] / p;
            cau[i] = (d1[i] - a1[i] * cau[i - 1]) / p;
        }
        if (iv == 1)
            j = SoilHorizonNum[nn - 1];
        psi1[nn - 1] = psiq(q1[nn - 1], qr1[nn - 1], qs1[nn - 1], alpha[j], vanGenuchtenBeta[j]);
        for (int i = nn - 2; i > 0; i--) {
            if (iv == 1)
                j = SoilHorizonNum[i];
            psi1[i] = dau[i] * psi1[i + 1] + cau[i];
            q1[i] = qpsi(psi1[i], qr1[i], qs1[i], alpha[j], vanGenuchtenBeta[j]);
        }
    }
//     The limits of water content are now checked and corrected, and
//  function WaterBalance() is called to correct water amounts.
    for (int i = 0; i < nn; i++) {
        if (q1[i] < qr1[i])
            q1[i] = qr1[i];
        if (q1[i] > qs1[i])
            q1[i] = qs1[i];
        if (q1[i] > pp1[i])
            q1[i] = pp1[i];
    }
    WaterBalance(q1, qx, dd, nn);
}

////////////////////////
void WaterBalance(double q1[], double qx[], double dd[], int nn)
//     This function checks and corrects the water balance in the soil cells
//  within a soil column or a soil layer. It is called by WaterFlux().
//     The implicit part of the solution may cause some deviation in the total amount of water
//  to occur. This module corrects the water balance if the sum of deviations is not zero, so
//  that the total amount of water in the array will not change. The correction is proportional
//  to the difference between the previous and present water amounts in each soil cell.
//
//     The following arguments are used here:
//        dd[] - one dimensional array of layer or column widths.
//        nn - the number of cells in this layer or column.
//        qx[] - one dimensional array of a layer or a column of the
//               previous values of cell.water_content.
//        q1[] - one dimensional array of a layer or a column of cell.water_content.
//
{
    double dev = 0;  // Sum of differences of water amount in soil
    double dabs = 0; // Sum of absolute value of differences in water content in
    // the array between beginning and end of this time step.
    for (int i = 0; i < nn; i++) {
        dev += dd[i] * (q1[i] - qx[i]);
        dabs += fabs(q1[i] - qx[i]);
    }
    if (dabs > 0)
        for (int i = 0; i < nn; i++)
            q1[i] = q1[i] - fabs(q1[i] - qx[i]) * dev / (dabs * dd[i]);

}

////////////////////////////////////////////////////////////////////////////////
void NitrogenFlow(int nn, double q01[], double q1[], double dd[], double nit[], double nur[])
//     This function computes the movement of nitrate and urea between the soil cells,
//  within a soil column or within a soil layer, as a result of water flux.
//     It is called by function CapillaryFlow().
//     It is assumed that there is only a passive movement of nitrate and urea
//  (i.e., with the movement of water).
//     The following arguments are used here:
//       dd[] - one dimensional array of layer or column widths.
//       nit[] - one dimensional array of a layer or a column of VolNo3NContent.
//       nn - the number of cells in this layer or column.
//       nur[] - one dimensional array of a layer or a column of VolUreaNContent.
//       q01[] - one dimensional array of a layer or a column of the previous values of cell.water_content.
//       q1[] - one dimensional array of a layer or a column of cell.water_content.
//
{
//     Zeroise very small values to prevent underflow.
    for (int i = 0; i < nn; i++) {
        if (nur[i] < 1e-20)
            nur[i] = 0;
        if (nit[i] < 1e-20)
            nit[i] = 0;
    }
//     Declare and zeroise arrays.
    double qdn[40] = {40 * 0}; // amount of nitrate N moving to the previous cell.
    double qup[40] = {40 * 0}; // amount of nitrate N moving to the following cell.
    double udn[40] = {40 * 0}; // amount of urea N moving to the previous cell.
    double uup[40] = {40 * 0}; // amount of urea N moving to the following cell.
    for (int i = 0; i < nn; i++) {
//     The amout of water in each soil cell before (aq0) and after (aq1) water movement is
//  computed from the previous values of water content (q01), the present values (q1),
//  and layer thickness. The associated transfer of soluble nitrate N (qup and qdn) and urea N
//  (uup and udn) is now computed. qup and uup are upward movement (from cell i+1 to i),
//  qdn and udn are downward movement (from cell i-1 to i).
        double aq0 = q01[i] * dd[i]; // previous amount of water in cell i
        double aq1 = q1[i] * dd[i];  // amount of water in cell i now
        if (i == 0) {
            qup[i] = 0;
            uup[i] = 0;
        } else {
            qup[i] = -qdn[i - 1];
            uup[i] = -udn[i - 1];
        }
//
        if (i == nn - 1) {
            qdn[i] = 0;
            udn[i] = 0;
        } else {
            qdn[i] = (aq1 - aq0) * nit[i + 1] / q01[i + 1];
            if (qdn[i] < (-0.2 * nit[i] * dd[i]))
                qdn[i] = -0.2 * nit[i] * dd[i];
            if (qdn[i] > (0.2 * nit[i + 1] * dd[i + 1]))
                qdn[i] = 0.2 * nit[i + 1] * dd[i + 1];
            udn[i] = (aq1 - aq0) * nur[i + 1] / q01[i + 1];
            if (udn[i] < (-0.2 * nur[i] * dd[i]))
                udn[i] = -0.2 * nur[i] * dd[i];
            if (udn[i] > (0.2 * nur[i + 1] * dd[i + 1]))
                udn[i] = 0.2 * nur[i + 1] * dd[i + 1];
        }
    }
//     Loop over all cells to update nit and nur arrays.
    for (int i = 0; i < nn; i++) {
        nit[i] += (qdn[i] + qup[i]) / dd[i];
        nur[i] += (udn[i] + uup[i]) / dd[i];
    }
}
