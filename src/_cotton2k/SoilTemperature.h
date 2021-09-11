#pragma once
#include <cinttypes>
#include "Simulation.hpp"

void SoilHeatFlux(State &, double, int, int, int, int, double);

double ThermalCondSoil(double, double, int);
