#pragma once
#include <cinttypes>
#include "Simulation.hpp"

void GravityFlow(SoilCell[40][20], double, double);
// SoilProcedure_2
void CapillaryFlow(Simulation &, unsigned int, int);
void DripFlow(SoilCell[40][20], double, double);

void NitrogenUptake(State &, SoilCell &, int, int, double, double, double);
