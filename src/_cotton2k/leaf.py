import numpy as np


def temperature_on_leaf_growth_rate(t):
    """The temperature function for leaf growth rate.

    It is based on the original code of GOSSYM, and the parameters are the same.

    Arguments
    ---------
    t
        temperature in C.

    Returns
    -------
    float
        factor between 0 and 1.

    Examples
    --------
    >>> temperature_on_leaf_growth_rate(12)
    0
    >>> temperature_on_leaf_growth_rate(16)
    0.265521614326615
    >>> temperature_on_leaf_growth_rate(20)
    0.5445166319979001
    >>> temperature_on_leaf_growth_rate(24)
    0.7618973851987698
    >>> temperature_on_leaf_growth_rate(27)
    0.9420718203047439
    >>> temperature_on_leaf_growth_rate(30)
    0.9998762589644965
    >>> temperature_on_leaf_growth_rate(36)
    0.7350456585969558
    >>> temperature_on_leaf_growth_rate(42)
    0
    """
    p1 = np.polynomial.Polynomial((-1.14277, 0.0910026, -0.00152344))
    p2 = np.polynomial.Polynomial((-0.317136, 0.0300712, -0.000416356))

    ra = p1(t) if t > 24 else p2(t)
    return max(0, ra / p1(p1.coef[1] / p1.coef[2] / -2))


def leaf_resistance_for_transpiration(age: float) -> float:
    """This function computes and returns the resistance of leaves of cotton plants to
    transpiration.

    It is assumed to be a function of leaf age.

    :param age: leaf age in physiological days.
    """
    # The following constant parameters are used:
    afac: float = 160.0  # factor used for computing leaf resistance.
    agehi: float = 94.0  # higher limit for leaf age.
    agelo: float = 48.0  # lower limit for leaf age.
    rlmin: float = 0.5  # minimum leaf resistance.

    if age <= agelo:
        return rlmin
    if age >= agehi:
        return rlmin + (agehi - agelo) * (agehi - agelo) / afac
    ax: float = 2.0 * agehi - agelo  # intermediate variable
    return rlmin + (age - agelo) * (ax - age) / afac
