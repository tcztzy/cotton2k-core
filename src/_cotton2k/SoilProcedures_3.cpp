//  SoilProcedures_3.cpp
//
//   functions in this file:
// WaterBalance()
// NitrogenFlow()
//
#include <cmath>
#include "global.h"
#include "GeneralFunctions.h"
#include "Simulation.hpp"

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
