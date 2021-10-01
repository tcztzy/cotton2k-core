// File SoilProcedures_2.cpp
//
//   functions in this file:
// DripFlow()
// CellDistance()
//
#include <cmath>
#include "global.h"
#include "GeneralFunctions.h"
#include "Simulation.hpp"

double CellDistance(int, int, int, int, double);

/////////////////////////////////////////////////////////////////////////////////////
void DripFlow(SoilCell soil_cells[40][20], double Drip, double row_space)
//     This function computes the water redistribution in the soil after irrigation
//  by a drip system. It also computes the resulting redistribution of nitrate and urea N.
//  It is called by SoilProcedures() noitr times per day. It calls function CellDistrib().
//     The following argument is used:
//  Drip - amount of irrigation applied by the drip method, mm.
//
//     The following global variables are referenced:
//       dl, LocationColumnDrip, LocationLayerDrip, MaxWaterCapacity,
//       nk, nl, NO3FlowFraction, PoreSpace, wk
//
//     The following global variables are set:
//       CumWaterDrained, SoilNitrogenLoss, VolNo3NContent, VolUreaNContent
//
{
    double dripw[40]; // amount of water applied, or going from one ring of
    // soil cells to the next one, cm3. (array)
    double dripn[40]; // amount of nitrate N applied, or going from one ring of soil
    // soil cells to the next one, mg. (array)
    double dripu[40]; // amount of urea N applied, or going from one ring of soil
    // soil cells to the next one, mg. (array)
    for (int i = 0; i < 40; i++) {
        dripw[i] = 0;
        dripn[i] = 0;
        dripu[i] = 0;
    }
//     Incoming flow of water (Drip, in mm) is converted to dripw(0), in cm3 per slab.
    dripw[0] = Drip * row_space * .10;
//     Wetting the cell in which the emitter is located.
    double h2odef; // the difference between the maximum water capacity (at a water content
    // of uplimit) of this ring of soil cell, and the actual water content, cm3.
    int l0 = LocationLayerDrip;  //  layer where the drip emitter is situated
    int k0 = LocationColumnDrip; //  column where the drip emitter is situated
    SoilCell &soil_cell = soil_cells[l0][k0];
//     It is assumed that wetting cannot exceed MaxWaterCapacity of this cell. Compute
//  h2odef, the amount of water needed to saturate this cell.
    h2odef = (MaxWaterCapacity[l0] - soil_cells[l0][k0].water_content) * dl(l0) * wk(k0, row_space);
//      If maximum water capacity is not exceeded - update cell.water_content of
//  this cell and exit the function.
    if (dripw[0] <= h2odef) {
        soil_cells[l0][k0].water_content += dripw[0] / (dl(l0) * wk(k0, row_space));
        return;
    }
//      If maximum water capacity is exceeded - calculate the excess of water flowing out of
//  this cell (in cm3 per slab) as dripw[1]. The next ring of cells (kr=1) will receive it
//  as incoming water flow.
    dripw[1] = dripw[0] - h2odef;
//      Compute the movement of nitrate N to the next ring
    double cnw = 0;  //  concentration of nitrate N in the outflowing water
    if (soil_cell.nitrate_nitrogen_content > 1.e-30) {
        cnw = soil_cell.nitrate_nitrogen_content / (soil_cells[l0][k0].water_content + dripw[0] / (dl(l0) * wk(k0, row_space)));
//     cnw is multiplied by dripw[1] to get dripn[1], the amount of nitrate N going out
//  to the next ring of cells. It is assumed, however, that not more than a proportion
//  (NO3FlowFraction) of the nitrate N in this cell can be removed in one iteration.
        if ((cnw * MaxWaterCapacity[l0]) < (NO3FlowFraction[l0] * soil_cell.nitrate_nitrogen_content)) {
            dripn[1] = NO3FlowFraction[l0] * soil_cell.nitrate_nitrogen_content * dl(l0) * wk(k0, row_space);
            soil_cell.nitrate_nitrogen_content = (1 - NO3FlowFraction[l0]) * soil_cell.nitrate_nitrogen_content;
        } else {
            dripn[1] = dripw[1] * cnw;
            soil_cell.nitrate_nitrogen_content = MaxWaterCapacity[l0] * cnw;
        }
    }
//     The movement of urea N to the next ring is computed similarly.
    double cuw = 0;  //  concentration of urea N in the outflowing water
    if (VolUreaNContent[l0][k0] > 1.e-30) {
        cuw = VolUreaNContent[l0][k0] / (soil_cells[l0][k0].water_content + dripw[0] / (dl(l0) * wk(k0, row_space)));
        if ((cuw * MaxWaterCapacity[l0]) < (NO3FlowFraction[l0] * VolUreaNContent[l0][k0])) {
            dripu[1] = NO3FlowFraction[l0] * VolUreaNContent[l0][k0] * dl(l0) * wk(k0, row_space);
            VolUreaNContent[l0][k0] = (1 - NO3FlowFraction[l0]) * VolUreaNContent[l0][k0];
        } else {
            dripu[1] = dripw[1] * cuw;
            VolUreaNContent[l0][k0] = MaxWaterCapacity[l0] * cuw;
        }
    }
    double defcit[40][20]; // array of the difference between water capacity and
    // actual water content in each cell of the ring
//     Set cell.water_content of the cell in which the drip is located to MaxWaterCapacity.
    soil_cells[l0][k0].water_content = MaxWaterCapacity[l0];
//     Loop of concentric rings of cells, starting from ring 1.
//     Assign zero to the sums sv, st, sn, sn1, su and su1.
    for (int kr = 1; kr < maxl; kr++) {
        double uplimit; //  upper limit of soil water content in a soil cell
        double sv = 0; // sum of actual water content in a ring of cells, cm3
        double st = 0; // sum of total water capacity in a ring of cells, cm3
        double sn = 0; // sum of nitrate N content in a ring of cells, mg.
        double sn1 = 0;// sum of movable nitrate N content in a ring of cells, mg
        double su = 0; // sum of urea N content in a ring of cells, mg
        double su1 = 0;// sum of movable urea N content in a ring of cells, mg
        double radius = 6 * kr;// radius (cm) of the wetting ring
        double dist; // distance (cm) of a cell center from drip location
//     Loop over all soil cells
        for (int l = 1; l < nl; l++) {
//     Upper limit of water content is the porespace volume in layers below the water table,
//  MaxWaterCapacity in other layers.
            uplimit = MaxWaterCapacity[l];
            for (int k = 0; k < nk; k++) {
//     Compute the sums sv, st, sn, sn1, su and su1 within the radius limits of this ring. The
//  array defcit is the sum of difference between uplimit and cell.water_content of each cell.
                dist = CellDistance(l, k, l0, k0, row_space);
                if (dist <= radius && dist > (radius - 6)) {
                    sv += soil_cells[l][k].water_content * dl(l) * wk(k, row_space);
                    st += uplimit * dl(l) * wk(k, row_space);
                    sn += soil_cells[l][k].nitrate_nitrogen_content * dl(l) * wk(k, row_space);
                    sn1 += soil_cells[l][k].nitrate_nitrogen_content * dl(l) * wk(k, row_space) * NO3FlowFraction[l];
                    su += VolUreaNContent[l][k] * dl(l) * wk(k, row_space);
                    su1 += VolUreaNContent[l][k] * dl(l) * wk(k, row_space) * NO3FlowFraction[l];
                    defcit[l][k] = uplimit - soil_cells[l][k].water_content;
                } else
                    defcit[l][k] = 0;
            } // end loop k
        } // end loop l
//     Compute the amount of water needed to saturate all the cells in this ring (h2odef).
        h2odef = st - sv;
//     Test if the amount of incoming flow, dripw(kr), is greater than  h2odef.
        if (dripw[kr] <= h2odef) {
//     In this case, this will be the last wetted ring.
//     Update cell.water_content in this ring, by wetting each cell in proportion
//  to its defcit. Update VolNo3NContent and VolUreaNContent of the cells in this ring
//  by the same proportion. this is executed for all the cells in the ring.
            for (int l = 1; l < nl; l++) {
                for (int k = 0; k < nk; k++) {
                    dist = CellDistance(l, k, l0, k0, row_space);
                    if (dist <= radius && dist > (radius - 6)) {
                        soil_cells[l][k].water_content += dripw[kr] * defcit[l][k] / h2odef;
                        soil_cells[l][k].nitrate_nitrogen_content += dripn[kr] * defcit[l][k] / h2odef;
                        VolUreaNContent[l][k] += dripu[kr] * defcit[l][k] / h2odef;
                    }
                } // end loop k
            } // end loop l
            return;
        } // end if dripw
//     If dripw(kr) is greater than h2odef, calculate cnw and cuw as the concentration of nitrate
//  and urea N in the total water of this ring after receiving the incoming water and nitrogen.
        cnw = (sn + dripn[kr]) / (sv + dripw[kr]);
        cuw = (su + dripu[kr]) / (sv + dripw[kr]);
        double drwout = dripw[kr] - h2odef;  //  the amount of water going out of a ring, cm3.
//     Compute the nitrate and urea N going out of this ring, and their amount lost from this
//  ring. It is assumed that not more than a certain part of the total nitrate or urea N
//  (previously computed as sn1 an su1) can be lost from a ring in one iteration. drnout and
//  xnloss are adjusted accordingly. druout and xuloss are computed similarly for urea N.
        double drnout = drwout * cnw;  //  the amount of nitrate N going out of a ring, mg
        double xnloss = 0; // the amount of nitrate N lost from a ring, mg
        if (drnout > dripn[kr]) {
            xnloss = drnout - dripn[kr];
            if (xnloss > sn1) {
                xnloss = sn1;
                drnout = dripn[kr] + xnloss;
            }
        }
        double druout = drwout * cuw;  //  the amount of urea N going out of a ring, mg
        double xuloss = 0;             // the amount of urea N lost from a ring, mg
        if (druout > dripu[kr]) {
            xuloss = druout - dripu[kr];
            if (xuloss > su1) {
                xuloss = su1;
                druout = dripu[kr] + xuloss;
            }
        }
//     For all the cells in the ring, as in the 1st cell, saturate cell.water_content to uplimit,
//  and update VolNo3NContent and VolUreaNContent.
        for (int l = 1; l < nl; l++) {
            uplimit = MaxWaterCapacity[l];
//
            for (int k = 0; k < nk; k++) {
                dist = CellDistance(l, k, l0, k0, row_space);
                if (dist <= radius && dist > (radius - 6)) {
                    soil_cells[l][k].water_content = uplimit;
                    if (xnloss <= 0)
                        soil_cells[l][k].nitrate_nitrogen_content = uplimit * cnw;
                    else
                        soil_cells[l][k].nitrate_nitrogen_content = soil_cells[l][k].nitrate_nitrogen_content * (1. - xnloss / sn);
                    if (xuloss <= 0)
                        VolUreaNContent[l][k] = uplimit * cuw;
                    else
                        VolUreaNContent[l][k] = VolUreaNContent[l][k] * (1. - xuloss / su);
                } // end if dist
            } // end loop k
        } // end loop l
//     The outflow of water, nitrate and urea from this ring will be the inflow into the next ring.
        if (kr < (nl - l0 - 1) && kr < maxl - 1) {
            dripw[kr + 1] = drwout;
            dripn[kr + 1] = drnout;
            dripu[kr + 1] = druout;
        } else
//     If this is the last ring, the outflowing water will be added to the drainage,
//  CumWaterDrained, the outflowing nitrogen to SoilNitrogenLoss.
        {
            return;
        } // end if kr...
//     Repeat all these procedures for the next ring.
    } // end loop kr
}

/////////////////////////
double CellDistance(int l, int k, int l0, int k0, double row_space)
//     This function computes the distance between the centers of cells l,k an l0,k0
//  It is called from DripFlow().
{
//     Compute vertical distance between centers of l and l0
    double xl = 0;  // vertical distance (cm) between cells
    if (l > l0) {
        for (int il = l0; il <= l; il++)
            xl += dl(il);
        xl -= (dl(l) + dl(l0)) * 0.5;
    } else if (l < l0) {
        for (int il = l0; il >= l; il--)
            xl += dl(il);
        xl -= (dl(l) + dl(l0)) * 0.5;
    }
//     Compute horizontal distance between centers of k and k0
    double xk = 0;  // horizontal distance (cm) between cells
    if (k > k0) {
        for (int ik = k0; ik <= k; ik++)
            xk += wk(ik, row_space);
        xk -= (wk(k, row_space) + wk(k0, row_space)) * 0.5;
    } else if (k < k0) {
        for (int ik = k0; ik >= k; ik--)
            xk += wk(ik, row_space);
        xk -= (wk(k, row_space) + wk(k0, row_space)) * 0.5;
    }
//     Compute diagonal distance between centers of cells
    return sqrt(xl * xl + xk * xk);
}
