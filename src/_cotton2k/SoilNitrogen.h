#pragma once
#include "Simulation.hpp"

void UreaHydrolysis(int, int, double, double, double);
void MineralizeNitrogen(int, int, int, int, double, double, double, double, double);
void Nitrification(SoilCell &, int, int, double, double);
void Denitrification(SoilCell &, int, int, double, double);
