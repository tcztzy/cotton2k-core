use super::root::*;

fn approx_equal(a: f64, b: f64, decimal_places: u8) -> bool {
    let factor = 10.0f64.powi(decimal_places as i32);
    let a = (a * factor).round();
    let b = (b * factor).round();
    a == b
}
