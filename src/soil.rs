/// This function computes soil water osmotic potential (in bars, positive value).
#[no_mangle]
extern "C" fn PsiOsmotic(q: f64, qsat: f64, ec: f64) -> f64
// The following arguments are used:
//   q - soil water content, cm3 cm-3.
//   qsat - saturated water content, cm3 cm-3.
//   ec - electrical conductivity of saturated extract (mmho/cm)
{
    if ec > 0f64 {
        let result = 0.36 * ec * qsat / q;
        if result > 6f64 {
            6f64
        } else {
            result
        }
    } else {
        0f64
    }
}

/// This function calculates soil mechanical resistance of cell l,k. It is computed
/// on the basis of parameters read from the input and calculated in RootImpedance().
///
/// It is called from PotentialRootGrowth().
///
///  The function has been adapted, without change, from the code of GOSSYM. Soil mechanical
/// resistance is computed as an empirical function of bulk density and water content.
/// It should be noted, however, that this empirical function is based on data for one type
/// of soil only, and its applicability for other soil types is questionable. The effect of soil
/// moisture is only indirectly reflected in this function. A new module (root_psi)
/// has therefore been added in COTTON2K to simulate an additional direct effect of soil
/// moisture on root growth.
///
/// The minimum value of rtimpd of this and neighboring soil cells is used to compute
/// rtpct. The code is based on a segment of RUTGRO in GOSSYM, and the values of the p1 to p3
/// parameters are based on GOSSYM usage:
#[no_mangle]
extern "C" fn SoilMechanicResistance(rtimpdmin: f64) -> f64 {
    let p1 = 1.046;
    let p2 = 0.034554;
    let p3 = 0.5;

    // effect of soil mechanical resistance on root growth (the return value).
    let rtpct = p1 - p2 * rtimpdmin;
    if rtpct > 1f64 {
        1f64
    } else if rtpct < p3 {
        p3
    } else {
        rtpct
    }
}

/// This function computes the aggregation factor for 2 mixed soil materials.
#[no_mangle]
extern "C" fn form(c0: f64, d0: f64, g0: f64) -> f64
// Arguments referenced:
//   c0 - heat conductivity of first material
//   d0 - heat conductivity of second material
//   g0 - shape factor for these materials
{
    (2f64 / (1f64 + (c0 / d0 - 1f64) * g0) + 1f64 / (1f64 + (c0 / d0 - 1f64) * (1f64 - 2f64 * g0)))
        / 3f64
}

/// the vertical width of a soil layer (cm).
#[no_mangle]
extern "C" fn dl(l: u32) -> f64 {
    if l >= 40 {
        panic!("Out of range");
    };
    match l {
        0 => 2.,
        1 => 2.,
        2 => 2.,
        3 => 4.,
        38 => 10.,
        39 => 10.,
        _ => 5.,
    }
}

pub fn depth(l: u32) -> f64 {
    if l >= 40 {
        panic!("Out of range");
    };
    match l {
        0 => 2.,
        1 => 4.,
        2 => 6.,
        3 => 10.,
        38 => 190.,
        39 => 200.,
        _ => 5. * l as f64 - 5.,
    }
}

/// horizontal width of a soil column (cm).
#[no_mangle]
extern "C" fn wk(_k: u32, row_space: f64) -> f64 {
    row_space / 20.
}

pub fn width(k: u32, row_space: f64) -> f64 {
    k as f64 * row_space / 20.
}
