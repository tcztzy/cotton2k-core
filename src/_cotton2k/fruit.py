import numpy as np


def TemperatureOnFruitGrowthRate(t: float) -> float:
    """Computes the effect of air temperature on growth rate of bolls in cotton plants.

    Arguments
    ---------
    t
        air temperature (°C)

    Examples
    --------
    >>> TemperatureOnFruitGrowthRate(12)
    0.0
    >>> TemperatureOnFruitGrowthRate(15)
    0.33575
    >>> TemperatureOnFruitGrowthRate(20)
    0.751
    >>> TemperatureOnFruitGrowthRate(25)
    0.97775
    >>> TemperatureOnFruitGrowthRate(26)
    1.00048
    >>> TemperatureOnFruitGrowthRate(28.514588859416445)
    1.0243183023872677
    >>> TemperatureOnFruitGrowthRate(30)
    1.016
    >>> TemperatureOnFruitGrowthRate(35)
    0.86575
    >>> TemperatureOnFruitGrowthRate(40)
    0.527
    >>> TemperatureOnFruitGrowthRate(45)
    0.0
    """
    return max(np.polynomial.Polynomial((-2.041, 0.215, -0.00377))(t), 0.0)
