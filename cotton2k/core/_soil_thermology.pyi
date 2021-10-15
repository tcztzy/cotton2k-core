import numpy.typing as npt
import numpy as np

def soil_thermal_conductivity_np(
    cka: float,
    ckw: float,
    dsand: float,
    bsand: float,
    dclay: float,
    bclay: float,
    soil_sand_volume_fraction: npt.NDArray[np.double],
    soil_clay_volume_fraction: npt.NDArray[np.double],
    pore_space: npt.NDArray[np.double],
    field_capacity: npt.NDArray[np.double],
    marginal_water_content: npt.NDArray[np.double],
    heat_conductivity_dry_soil: npt.NDArray[np.double],
    q0: npt.NDArray[np.double],
    t0: npt.NDArray[np.double],
    l0: npt.NDArray[np.uint64],
) -> npt.NDArray[np.double]: ...
