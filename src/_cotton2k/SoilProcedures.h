#pragma once
#include <cinttypes>
#include "Simulation.hpp"
// SoilProcedure_2
void DripFlow(SoilCell[40][20], double, double);

void NitrogenFlow(int, double[], double[], double[], double[], double[]);
void WaterFlux(double[], double[], double[], double[], double[], double[], int, int, int, long, int);
