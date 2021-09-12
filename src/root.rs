/// This function calculates the reduction of potential root growth rate in cells with low nitrate content. It is called from PotentialRootGrowth().
///
/// It has been adapted from GOSSYM. It is assumed that root growth is reduced when nitrate N content falls below a certain level.
///
/// NOTE: This function actually does nothing. It is disabled by the choice of the constant parameters. It may be redefined when more experimental data become available.
#[no_mangle]
#[allow(unused_variables)]
extern "C" fn SoilNitrateOnRootGrowth(vno3clk: f64) -> f64
// The following argument is used:
//   vno3clk - VolNo3NContent value for this cell
{
    1f64
}
