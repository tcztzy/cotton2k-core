def canopy_balance(  # pylint: disable=too-many-arguments,too-many-locals
    etp1: float,
    rlzero: float,
    rsv: float,
    c2: float,
    sf: float,
    so: float,
    thet: float,
    tv: float,
) -> float:
    """Solves the energy balance equations at the foliage / air interface, and computes
    the resulting temperature of the foliage.

    Units for all energy fluxes are: ly / sec.

    Parameters
    ----------
    c2
        multiplier for sensible heat transfer at plant surface.
    etp1
        transpiration (mm / sec).
    rlzero
        incoming long wave radiation (ly / sec).
    rsv
        global radiation absorbed by the vegetation
    sf
        fraction of shaded soil area
    so
        temperature of soil surface (k).
    thet
        air temperature (k).
    tv
        temperature of plant canopy.

    Returns
    -------
    float
        temperature of plant canopy.

    Examples
    --------
    >>> canopy_balance(0.1, 1, 10, 1, 0.8, 12, 15, 18)
    20.938243153794296
    """
    # Constants:
    ef = 0.95  # emissivity of the foliage surface
    eg = 0.95  # emissivity of the soil surface
    stefa1 = 1.38e-12  # stefan-boltsman constant.

    # long wave radiation reaching the canopy
    rlv1 = sf * ef * rlzero + sf * ef * eg * stefa1 * so ** 4  # from sky and soil
    # rlv4 is the multiplier of tv**4 for emitted long wave radiation from vegetation,
    # corrected for the amount reflected back from soil surface and absorbed by foliage
    # This is two-sided (note that when eg = ef = 1, the coefficient corr will be 2)
    corr = 1 + eg / (ef + eg - ef * eg)  # coefficient
    rlv4 = stefa1 * sf * ef * corr
    tvex = 0.0  # previous value of tv
    ccadx = 0.0  # previous value of ccadjust
    # Start iterations for tv:
    for mot in range(50):
        # Emitted long wave radiation from vegetation (cclwe)
        cclwe = rlv4 * tv ** 4
        dcclwe = 4 * rlv4 * tv ** 3  # derivative of cclwe
        # Latent heat flux (hvlat) is computed from the transpiration rate.
        hvlat = (75.5255 - 0.05752 * tv) * etp1
        dhvlat = -0.05752 * etp1  # derivative of hvlat
        # Sensible heat transfer from vegetation
        # average air temperature above soil surface (k) in canopy
        tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)
        senfheat = c2 * (tv - tafk)  # sensible heat transfer from foliage
        dsenfheat = c2 * (1 - 0.6 * sf)  # derivative of senfheat
        # Compute the energy balance at the plant surface (cc), and if it is small
        # enough end the computation.
        cc = (
            cclwe  # (a) long wave emission from vegetation
            + hvlat  # (b) latent heat transfer
            + senfheat  # (c) sensible heat transfer from vegetation to air
            - rlv1  # long wave radiation reaching the vegetation
            - rsv  # global radiation on vegetation
        )
        if abs(cc) < 1e-5:
            return tv  # end iterations for tv
        # If cc is not small enough, compute its derivative by tv (ccp).
        # ccp is the derivative of energy balance at the plant surface (by tv)
        ccp = dcclwe + dhvlat + dsenfheat  # (a)  # (b)  # (c)
        # Correct the canopy temperature by  the ratio of cc to ccp.
        ccadjust = cc / ccp  # adjustment of tv before next iteration
        # If adjustment is small enough, no more iterations are needed.
        if abs(ccadjust) < 2e-3:
            return tv
        # If ccadjust is not the same sign as ccadx, reduce fluctuations
        if mot >= 2 and abs(ccadjust - ccadx) > abs(ccadjust + ccadx):
            ccadjust = (ccadjust + ccadx) / 2
            tv = (tv + tvex) / 2
        ccadjust = min(max(ccadjust, -10), 10)
        tv -= ccadjust
        tvex = tv
        ccadx = ccadjust
    # If reached 50 iterations there must be an error somewhere!
    raise RuntimeError
