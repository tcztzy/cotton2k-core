#ifndef SIMULATION_TYPE
#define SIMULATION_TYPE
#include "State.hpp"
#include "Climate.h"
#include "Irrigation.h"
typedef struct Simulation
{
    double row_space;                       // average row spacing, cm.
    double plant_population;                // plant population, plants per hectar.
    double cultivar_parameters[61];
    ClimateStruct climate[400];             // structure containing the following daily weather data:
                                            // int nDay =    day of year.
                                            // double Rad =  daily global radiation, in langleys.
                                            // double Tmax = maximum daily temperature, C.
                                            // double Tmin = minimum daily temperature, C.
                                            // double Tdew = dew point temperature, C.
                                            // double Rain = daily rainfall, mm.
                                            // double Wind = daily wind run, km.
    Irrigation irrigation[150];
    State states[200];
} Simulation;
#endif
