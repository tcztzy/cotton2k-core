from math import cos, pi


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


class Thermology:  # pylint: disable=E0203,E1101,R0903,R0912,R0914,R0915,W0201
    def soil_thermology(self):
        """The main part of the soil temperature sub-model.

        References
        ----------

        Benjamin, J.G., Ghaffarzadeh, M.R. and Cruse, R.M. 1990. Coupled water and heat
        transport in ridged soils. Soil Sci. Soc. Am. J. 54:963-969.

        Chen, J. 1984. Uncoupled multi-layer model for the transfer of sensible and
        latent heat flux densities from vegetation. Boundary-Layer Meteorology 28:213-
        225.

        Chen, J. 1985. A graphical extrapolation method to determine canopy resistance
        from measured temperature and humidity profiles above a crop canopy. Agric.
        For. Meteorol. 37:75-88.

        Clothier, B.E., Clawson, K.L., Pinter, P.J.Jr., Moran, M.S., Reginato, R.J. and
        Jackson, R.D. 1986. Estimation of soil heat flux from net radiation during the
        growth of alfalfa. Agric. For. Meteorol. 37:319-329.

        Costello, T.A. and Braud, H.J. Jr. 1989. Thermal diffusivity of soil by non-
        linear regression analysis of soil temperature data. Trans. ASAE 32:1281-1286.

        De Vries, D.A. 1963. Thermal properties of soils. In: W.R. Van Wijk (ed)
        Physics of plant environment, North Holland, Amsterdam, pp 210-235.

        Deardorff, J.W. 1978. Efficient prediction of ground surface temperature and
        moisture with inclusion of a layer of vegetation. J. Geophys. Res. 83 (C4):
        1889-1903.

        Dong, A., Prashar, C.K. and Grattan, S.R. 1988. Estimation of daily and hourly
        net radiation. CIMIS Final Report June 1988, pp. 58-79.

        Ephrath, J.E., Goudriaan, J. and Marani, A. 1996. Modelling diurnal patterns of
        air temperature, radiation, wind speed and relative humidity by equations from
        daily characteristics. Agricultural Systems 51:377-393.

        Hadas, A. 1974. Problem involved in measuring the soil thermal conductivity and
        diffusivity in a moist soil. Agric. Meteorol. 13:105-113.

        Hadas, A. 1977. Evaluation of theoretically predicted thermal conductivities of
        soils under field and laboratory conditions. Soil Sci. Soc. Am. J. 41:460-466.

        Hanks, R.J., Austin, D.D. and Ondrechen, W.T. 1971. Soil temperature estimation
        by a numerical method. Soil Sci. Soc. Am. Proc. 35:665-667.

        Hares, M.A. and Novak, M.D. 1992. Simulation of surface energy balance and soil
        temperature under strip tillage: I. Model description. Soil Sci. Soc. Am. J.
        56:22-29.

        Hares, M.A. and Novak, M.D. 1992. Simulation of surface energy balance and soil
        temperature under strip tillage: II. Field test. Soil Sci. Soc. Am. J. 56:29-36.

        Horton, E. and Wierenga, P.J. 1983. Estimating the soil heat flux from
        observations of soil temperature near the surface. Soil Sci. Soc. Am. J. 47:
        14-20.

        Horton, E., Wierenga, P.J. and Nielsen, D.R. 1983. Evaluation of methods for
        determining apparent thermal diffusivity of soil near the surface. Soil Sci.
        Soc. Am. J. 47:25-32.

        Horton, R. 1989. Canopy shading effects on soil heat and water flow. Soil Sci.
        Soc. Am. J. 53:669-679.

        Horton, R., and Chung, S-O, 1991. Soil Heat Flow. Ch. 17 in: Hanks, J., and
        Ritchie, J.T., (Eds.) Modeling Plant and Soil Systems. Am. Soc. Agron.,
        Madison, WI, pp 397-438.

        Iqbal, M. 1983. An Introduction to Solar Radiation. Academic Press.

        Kimball, B.A., Jackson, R.D., Reginato, R.J., Nakayama, F.S. and Idso, S.B.
        1976. Comparison of field-measured and calculated soil heat fluxes. Soil Sci.
        Soc. Am. J. 40:18-28.

        Lettau, B. 1971. Determination of the thermal diffusivity in the upper layers
        of a natural ground cover. Soil Sci. 112:173-177.

        Monin, A.S. 1973. Boundary layers in planetary atmospheres. In: P. Morrel
        (ed.), Dynamic meteorology, D. Reidel Publishing Company, Boston, pp. 419-458.

        Spitters, C.J.T., Toussaint, H.A.J.M. and Goudriaan, J. 1986. Separating the
        diffuse and direct component of global radiation and its implications for
        modeling canopy photosynthesis. Part I. Components of incoming radiation.
        Agric. For. Meteorol. 38:217-229.

        Wierenga, P.J. and de Wit, C.T. 1970. Simulation of heat flow in soils. Soil
        Sci. Soc. Am. Proc. 34:845-848.

        Wierenga, P.J., Hagan, R.M. and Nielsen, D.R. 1970. Soil temperature profiles
        during infiltration and redistribution of cool and warm irrigation water. Water
        Resour. Res. 6:230-238.

        Wierenga, P.J., Nielsen, D.R. and Hagan, R.M. 1969. Thermal properties of soil
        based upon field and laboratory measurements. Soil Sci. Soc. Am. Proc. 33:354-
        360.
        """
        state = self._current_state
        u = (state.date - self.start_date).days
        # Compute dts, the daily change in deep soil temperature (C), as a site-
        # dependent function of Daynum.
        dts = (
            2
            * pi
            * self.site_parameters[10]
            / 365
            * cos(
                2
                * pi
                * (state.date.timetuple().tm_yday - self.site_parameters[11])
                / 365
            )
        )
        # Define iter1 and dlt for hourly time step.
        iter1 = 24  # number of iterations per day.
        dlt = 3600  # time (seconds) of one iteration.
        kk = 1  # number of soil columns for executing computations.
        # If there is no canopy cover, no horizontal heat flux is assumed, kk = 1.
        # Otherwise it is equal to the number of columns in the slab.
        shadeav = 0  # average shaded area in all shaded soil columns.
        # emerge_switch defines the type of soil temperature computation.
        if self.emerge_switch > 1:
            shadetot = 0  # sum of shaded area in all shaded soil columns.
            nshadedcol = 0  # number of at least partially shaded soil columns.
            kk = 20
            for k in range(20):
                if self.irradiation_soil_surface[k] <= 0.99:
                    shadetot += 1 - self.irradiation_soil_surface[k]
                    nshadedcol += 1

            if nshadedcol > 0:
                shadeav = shadetot / nshadedcol
        # es and ActualSoilEvaporation are computed as the average for the whole soil
        # slab, weighted by column widths.
        es = 0  # potential evaporation rate, mm day-1
        state.actual_soil_evaporation = 0
        # Start hourly loop of iterations.
        for ihr in range(iter1):
            # Update the temperature of the last soil layer (lower boundary conditions).
            state.deep_soil_temperature += dts * dlt / 86400
            etp0 = 0  # actual transpiration (mm s-1) for this hour
            if state.evapotranspiration > 0.000001:
                etp0 = (
                    state.actual_transpiration
                    * state.hours[ihr].ref_et
                    / state.evapotranspiration
                    / dlt
                )
            # Compute vertical transport for each column
            for k in range(kk):
                #  Set hourly_soil_temperature for the lowest soil layer.
                state.hourly_soil_temperature[
                    ihr, 40 - 1, k
                ] = state.deep_soil_temperature
                # Compute transpiration from each column, weighted by its relative
                # shading.
                etp1 = 0  # actual hourly transpiration (mm s-1) for a column.
                if shadeav > 0.000001:
                    etp1 = etp0 * (1 - self.irradiation_soil_surface[k]) / shadeav
                ess = 0  # evaporation rate from surface of a soil column (mm / sec).
                # The potential evaporation rate (escol1k) from a column is the sum of
                # the radiation component of the Penman equation(es1hour), multiplied
                # by the relative radiation reaching this column, and the wind and
                # vapor deficit component of the Penman equation (es2hour).
                # potential evaporation fron soil surface of a column, mm per hour.
                escol1k = (
                    state.hours[ihr].et1 * self.irradiation_soil_surface[k]
                    + state.hours[ihr].et2
                )
                es += escol1k * self.column_width[k]
                # Compute actual evaporation from soil surface. update water content of
                # the soil cell, and add to daily sum of actual evaporation.
                evapmax = (
                    0.9
                    * (state.soil_water_content[0, k] - self.thad[0])
                    * 10
                    * self.layer_depth[0]
                )  # maximum possible evaporatio from a soil cell near the surface.
                escol1k = min(evapmax, escol1k)
                state.soil_water_content[0, k] -= 0.1 * escol1k / self.layer_depth[0]
                state.actual_soil_evaporation += escol1k * self.column_width[k]
                ess = escol1k / dlt
                # Call self._energy_balance to compute soil surface and canopy
                # temperature.
                self._energy_balance(u, ihr, k, ess, etp1)
            # Compute soil temperature flux in the vertical direction.
            # Assign iv = 1, layer = 0, nn = nl.
            iv = 1  # indicates vertical (=1) or horizontal (=0) flux.
            nn = 40  # number of array members for heat flux.
            layer = 0  # soil layer number
            # Loop over kk columns, and call SoilHeatFlux().
            for k in range(kk):
                state.soil_heat_flux(dlt, iv, nn, layer, k, self.row_space, ihr)
            # If no horizontal heat flux is assumed, make all array members of
            # hourly_soil_temperature equal to the value computed for the first column.
            # Also, do the same for array memebers of soil_water_content.
            if self.emerge_switch <= 1:
                state.hourly_soil_temperature[
                    ihr, :, :
                ] = state.hourly_soil_temperature[ihr, :, 0][:, None].repeat(20, axis=1)
                state.soil_water_content[0, :] = state.soil_water_content[0, 0]
            # Compute horizontal transport for each layer

            # Compute soil temperature flux in the horizontal direction, when
            # emerge_switch = 2.
            # Assign iv = 0 and nn = nk. Start loop for soil layers, and call
            # soil_heat_flux.
            if self.emerge_switch > 1:
                iv = 0
                nn = 20
                for l in range(40):
                    layer = l
                    state.soil_heat_flux(dlt, iv, nn, layer, l, self.row_space, ihr)
            # Compute average temperature of foliage, in degrees C. The average is
            # weighted by the canopy shading of each column, only columns which are
            # shaded 5% or more by canopy are used.
            tfc = 0  # average foliage temperature, weighted by shading in each column
            shading = 0  # sum of shaded area in all shaded columns, to compute TFC
            for k in range(20):
                if self.irradiation_soil_surface[k] <= 0.95:
                    tfc += (state.foliage_temperature[k] - 273.161) * (
                        1 - self.irradiation_soil_surface[k]
                    )
                    shading += 1 - self.irradiation_soil_surface[k]
            if shading >= 0.01:
                tfc /= shading
            # If emergence date is to be simulated, call predict_emergence().
            if self.emerge_switch == 0 and state.date >= self.plant_date:
                emerge_date = state.predict_emergence(
                    self.plant_date, ihr, self.plant_row_column
                )
                if emerge_date is not None:
                    self.emerge_date = emerge_date
                    self.emerge_switch = 2
            if ihr < 23:
                state.hourly_soil_temperature[ihr + 1] = state.hourly_soil_temperature[
                    ihr
                ]
        # At the end of the day compute actual daily evaporation and its cumulative sum
        if kk == 1:
            es /= self.column_width[1]
            state.actual_soil_evaporation /= self.column_width[1]
        else:
            es /= self.row_space
            state.actual_soil_evaporation /= self.row_space
        # compute daily averages.
        state.soil_temperature = state.hourly_soil_temperature.mean(axis=0)
