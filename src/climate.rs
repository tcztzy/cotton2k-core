use super::*;

/// Function tdewhour() computes the hourly values of dew point temperature from average dew-point and the daily estimated range. This range is computed as a regression on maximum and minimum temperatures.
#[no_mangle]
extern "C" fn tdewhour(
    sim: &Simulation,
    u: u32,
    time: f64,
    temperature: f64,
    sunrise: f64,
    solar_noon: f64,
    site8: f64,
    site12: f64,
    site13: f64,
    site14: f64,
) -> f64 {
    let im1 = if u > 1 { u - 1 } else { 0 }; // day of year yesterday
    let yesterday = sim.climate[im1 as usize];
    let today = sim.climate[u as usize];
    let ip1 = u + 1; // day of year tomorrow
    let tomorrow = sim.climate[ip1 as usize];
    let tdmin; // minimum of dew point temperature.
    let mut tdrange; // range of dew point temperature.
    let hmax = solar_noon + site8; // time of maximum air temperature
    if time <= sunrise {
        // from midnight to sunrise
        tdrange = site12 + site13 * yesterday.Tmax + site14 * today.Tmin;
        if tdrange < 0f64 {
            tdrange = 0f64;
        }
        tdmin = yesterday.Tdew - tdrange / 2f64;
        tdmin + tdrange * (temperature - today.Tmin) / (yesterday.Tmax - today.Tmin)
    } else if time <= hmax {
        // from sunrise to hmax
        tdrange = site12 + site13 * today.Tmax + site14 * today.Tmin;
        if tdrange < 0f64 {
            tdrange = 0f64;
        }
        tdmin = today.Tdew - tdrange / 2f64;
        tdmin + tdrange * (temperature - today.Tmin) / (today.Tmax - today.Tmin)
    } else {
        //  from hmax to midnight
        tdrange = site12 + site13 * today.Tmax + site14 * tomorrow.Tmin;
        if tdrange < 0f64 {
            tdrange = 0f64;
        }
        tdmin = tomorrow.Tdew - tdrange / 2f64;
        tdmin + tdrange * (temperature - tomorrow.Tmin) / (today.Tmax - tomorrow.Tmin)
    }
}

enum SoilRunoff {
    Low,
    Moderate,
    High,
}

#[no_mangle]
extern "C" fn SimulateRunoff(
    sim: &Simulation,
    u: u32,
    SandVolumeFraction: f64,
    ClayVolumeFraction: f64,
    NumIrrigations: u32,
) -> f64
//     This function is called from DayClim() and is executed on each day with raifall more
//  than 2 mm. It computes the runoff and the retained portion of the rainfall. Note: This
//  function is based on the code of GOSSYM. No changes have been made from the original GOSSYM
//  code (except translation to C++). It has not been validated by actual field measurement.
//     It calculates the portion of rainfall that is lost to runoff, and reduces rainfall to the
//  amount which is actually infiltrated into the soil. It uses the soil conservation service
//  method of estimating runoff.
//     References:
//  - Brady, Nyle C. 1984. The nature and properties of soils, 9th ed. Macmillan Publishing Co.
//  - Schwab, Frevert, Edminster, and Barnes. 1981. Soil and water conservation engineering,
//  3rd ed. John Wiley & Sons, Inc.
//
//     The following global variables are referenced here:
//  ClayVolumeFraction, Irrig (structure), NumIrrigations, SandVolumeFraction.
//     The argument used here:  rain = today,s rainfall.
//     The return value:  the amount of water (mm) lost by runoff.
{
    let iGroup: SoilRunoff;
    let d01: f64; // Adjustment of curve number for soil groups A,B,C.

    // Infiltration rate is estimated from the percent sand and percent clay in the Ap layer.
    // If clay content is greater than 35%, the soil is assumed to have a higher runoff potential,
    // if clay content is less than 15% and sand is greater than 70%, a lower runoff potential is
    // assumed. Other soils (loams) assumed moderate runoff potential. No 'impermeable' (group D)
    // soils are assumed.  References: Schwab, Brady.

    if SandVolumeFraction > 0.70 && ClayVolumeFraction < 0.15 {
        // Soil group A = 1, low runoff potential
        iGroup = SoilRunoff::Low;
        d01 = 1.0;
    } else if ClayVolumeFraction > 0.35 {
        // Soil group C = 3, high runoff potential
        iGroup = SoilRunoff::High;
        d01 = 1.14;
    } else {
        // Soil group B = 2, moderate runoff potential
        iGroup = SoilRunoff::Moderate;
        d01 = 1.09;
    }
    // Loop to accumulate 5-day antecedent rainfall (mm) which will affect the soil's ability
    // to accept new rainfall. This also includes all irrigations.
    let mut i01 = u as i32 - 5;
    if i01 < 0 {
        i01 = 0;
    }
    let mut PreviousWetting = 0f64; // five day total (before this day) of rain and irrigation, mm
    for Dayn in (i01 as usize)..u as usize {
        let mut amtirr = 0f64; // mm water applied on this day by irrigation
        for i in 0..NumIrrigations as usize {
            if (Dayn as i32) == sim.irrigation[i].day {
                amtirr = sim.irrigation[i].amount;
            }
        }
        PreviousWetting += amtirr + sim.climate[Dayn].Rain;
    }
    //
    let d02; // Adjusting curve number for antecedent rainfall conditions.
    if PreviousWetting < 3f64 {
        //  low moisture, low runoff potential.
        d02 = match iGroup {
            SoilRunoff::Low => 0.71,
            SoilRunoff::Moderate => 0.78,
            SoilRunoff::High => 0.83,
        };
    } else if PreviousWetting > 53f64 {
        //  wet conditions, high runoff potential.
        d02 = match iGroup {
            SoilRunoff::Low => 1.24,
            SoilRunoff::Moderate => 1.15,
            SoilRunoff::High => 1.10,
        };
    } else {
        //  moderate conditions
        d02 = 1.00;
    }
    //  Assuming straight rows, and good cropping practice:
    let mut crvnum = 78.0; // Runoff curve number, unadjusted for moisture and soil type.
    crvnum *= d01 * d02; // adjusted curve number
    let d03 = 25400f64 / crvnum - 254f64; // maximum potential difference between rainfall and runoff.
                                          //
    let rain = sim.climate[u as usize].Rain;
    if rain <= 0.2 * d03 {
        0f64
    } else {
        (rain - 0.2 * d03).powi(2) / (rain + 0.8 * d03)
    }
}
