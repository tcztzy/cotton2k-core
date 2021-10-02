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
