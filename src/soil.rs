/// This function computes soil water content (cm3 cm-3) for
/// a given value of matrix potential, using the Van-Genuchten equation.
#[no_mangle]
extern "C" fn qpsi(psi: f64, qr: f64, qsat: f64, alpha: f64, beta: f64) -> f64
// The following arguments are used:
//   alpha, beta  - parameters of the van-genuchten equation.
//   psi - soil water matrix potential (bars).
//   qr - residual water content, cm3 cm-3.
//   qsat - saturated water content, cm3 cm-3.
{
    // For very high values of PSI, saturated water content is assumed.
    // For very low values of PSI, air-dry water content is assumed.
    if psi >= -0.00001 {
        qsat
    } else if psi <= -500000f64 {
        qr
    } else {
        // The soil water matric potential is transformed from bars (psi)
        // to cm in positive value (psix).
        let psix = 1000. * (psi + 0.00001).abs();
        // The following equation is used (in FORTRAN notation):
        //   QPSI = QR + (QSAT-QR) / (1 + (ALPHA*PSIX)**BETA)**(1-1/BETA)
        let gama = 1. - 1. / beta;
        let term = 1. + (alpha * psix).powf(beta); //  intermediate variable
        let swfun = qr + (qsat - qr) / term.powf(gama); //  computed water content
        if swfun < (qr + 0.0001) {
            qr + 0.0001
        } else {
            swfun
        }
    }
}

/// This function computes soil water matric potential (in bars) for a given value of soil water content, using the Van-Genuchten equation.
#[no_mangle]
extern "C" fn psiq(q: f64, qr: f64, qsat: f64, alpha: f64, beta: f64) -> f64
// The following arguments are used:
//   alpha, beta  - parameters of the van-genuchten equation.
//   q - soil water content, cm3 cm-3.
//   qr - residual water content, cm3 cm-3.
//   qsat - saturated water content, cm3 cm-3.
{
    // For very low values of water content (near the residual water
    // content) psiq is -500000 bars, and for saturated or higher water
    // content psiq is -0.00001 bars.
    if (q - qr) < 0.00001 {
        return -500000.;
    } else if q >= qsat {
        return -0.00001;
    }
    // The following equation is used (FORTRAN notation):
    // PSIX = (((QSAT-QR) / (Q-QR))**(1/GAMA) - 1) **(1/BETA) / ALPHA
    let gama = 1. - 1. / beta;
    let gaminv = 1. / gama;
    let term = ((qsat - qr) / (q - qr)).powf(gaminv); //  intermediate variable
    let mut psix = (term - 1.).powf(1. / beta) / alpha;
    if psix < 0.01 {
        psix = 0.01;
    }
    // psix (in cm) is converted to bars (negative value).
    psix = (0.01 - psix) * 0.001;
    if psix < -500000. {
        psix = -500000.;
    }
    if psix > -0.00001 {
        psix = -0.00001;
    }
    psix
}

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
