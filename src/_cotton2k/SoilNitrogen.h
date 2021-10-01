#pragma once
#include "Simulation.hpp"

void UreaHydrolysis(int, int, double, double, double);
void MineralizeNitrogen(SoilCell &, int, int, const int &, const int &, double, double);
void Nitrification(SoilCell &, int, int, double, double);
void Denitrification(SoilCell &, int, int, double, double);
