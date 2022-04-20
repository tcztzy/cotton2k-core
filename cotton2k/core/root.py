import numpy as np

from .soil import SoilTemOnRootGrowth


def depth_of_layer(l: int) -> float:
    """
    Examples
    --------
    >>> depth_of_layer(0)
    2.0
    >>> depth_of_layer(1)
    2.0
    >>> depth_of_layer(2)
    2.0
    >>> depth_of_layer(3)
    4.0
    >>> depth_of_layer(4)
    5.0
    >>> depth_of_layer(37)
    5.0
    >>> depth_of_layer(38)
    10.0
    >>> depth_of_layer(39)
    10.0
    >>> depth_of_layer(40)
    Traceback (most recent call last):
    ...
    IndexError: Out of range
    """
    if l >= 40 or l < 0:
        raise IndexError("Out of range")
    return (
        {
            0: 2.0,
            1: 2.0,
            2: 2.0,
            3: 4.0,
            38: 10.0,
            39: 10.0,
        }
    ).get(l, 5.0)


class RootGrowth:  # pylint: disable=no-member,attribute-defined-outside-init,too-few-public-methods,too-many-arguments
    def compute_actual_root_growth(
        self,
        sumpdr,
        row_space,
        column_width,
        per_plant_area,
        NumRootAgeGroups,
        emerge_date,
        plant_row_column,
    ):
        # The following constant parameters are used:
        # The index for the relative partitioning of root mass produced by new growth
        # to class i.
        # to soil cell volume (g/cm3); when this threshold is reached, a part of root
        # growth in this cell may be extended to adjoining cells.
        # Assign zero to pavail if this is the day of emergence.
        if self.date <= emerge_date:
            self.pavail = 0
        # Assign zero to the arrays of actual root growth rate.
        self.actual_root_growth[:] = 0
        # The amount of carbon allocated for root growth is calculated from
        # carbon_allocated_for_root_growth, converted to g dry matter per slab, and
        # added to previously allocated carbon that has not been used for growth.
        # if there is no potential root growth, this will be stored in pavail.
        # Otherwise, zero is assigned to pavail.
        self.pavail += (
            self.carbon_allocated_for_root_growth * 0.01 * row_space / per_plant_area
        )
        if sumpdr <= 0:
            return
        # The ratio of available C to potential root growth (actgf) is calculated.
        # pavail (if not zero) is used here, and zeroed after being used.
        actgf = (  # actual growth factor (ratio of available C to potential growth).
            self.pavail / sumpdr
        )
        self.pavail = 0

        # actual growth rate from roots existing in this soil cell.
        adwr1 = self._root_potential_growth.copy()
        adwr1[self.root_age > 0] *= actgf
        self.distribute_extra_carbon_for_root(
            row_space, column_width, per_plant_area, plant_row_column
        )
        self.distribute_root_weight_in_cell(
            NumRootAgeGroups, adwr1, column_width, plant_row_column
        )
        self.distribute_root_weight_by_age()
        self.tap_root_growth(NumRootAgeGroups, plant_row_column)
        self.lateral_root_growth(NumRootAgeGroups, plant_row_column, row_space)
        # Initialize daily_root_loss (weight of sloughed roots) for this day.
        daily_root_loss = 0  # total weight of sloughed roots, g per plant per day.
        for l in range(40):
            for k in range(20):
                # Check RootAge to determine if this soil cell contains roots, and then
                # compute root aging and root death by calling root_aging() and
                # root_death() for each soil cell with roots.
                if self.root_age[l][k] > 0:
                    self.root_aging(l, k)
                    daily_root_loss += self.root_death(l, k)
        # Convert daily_root_loss to g per plant units
        daily_root_loss *= 100.0 * per_plant_area / row_space
        # Adjust root_nitrogen (root N content) for loss by death of roots.
        self.root_nitrogen -= daily_root_loss * self.root_nitrogen_concentration

    def distribute_root_weight_in_cell(
        self, NumRootAgeGroups, adwr1, column_width, plant_row_column
    ):
        # Check each cell if the ratio of root weight capable of growth to cell volume
        # (rtconc) exceeds the threshold rtminc, and call redist_root_new_growth() for
        # this cell.
        # Otherwise, all new growth is contained in the same cell, and the actual
        # growth in this cell, actual_root_growth(l,k) will be equal to adwr1(l,k).
        cgind = [1, 1, 0.10]
        rtminc = 0.0000001  # the threshold ratio of root mass capable of growth
        for l in range(40):
            for k in range(20):
                if self.root_age[l][k] > 0:
                    rtconc = 0  # ratio of root weight capable of growth to cell volume
                    for i in range(NumRootAgeGroups):
                        rtconc += self.root_weights[l][k][i] * cgind[i]
                    rtconc = rtconc / (depth_of_layer(l) * column_width)
                    if rtconc > rtminc:
                        self.redist_root_new_growth(
                            l, k, adwr1[l][k], column_width, plant_row_column
                        )
                    else:
                        self.actual_root_growth[l][k] += adwr1[l][k]

    def distribute_root_weight_by_age(
        self, root_growth_index=np.array((1.0, 0.0, 0.0), dtype=np.float64)
    ):
        # The new actual growth actual_root_growth(l,k) in each cell is partitioned
        # among the root classes in it in proportion to the parameters
        # root_growth_index(i), and the previous values of root_weights(k,l,i), and
        # added to root_weights(k,l,i).
        for l in range(40):
            for k in range(20):
                if self.root_age[l][k] > 0:
                    # sum of growth index multiplied by root weight, for all classes in
                    # a cell.
                    if (
                        sumgr := (root_growth_index * self.root_weights[l][k]).sum()
                    ) > 0:
                        self.root_weights[l][k] *= (
                            1
                            + self.actual_root_growth[l][k] * root_growth_index / sumgr
                        )
                    else:
                        self.root_weights[l][k] += (
                            self.actual_root_growth[l][k]
                            * root_growth_index
                            / sum(root_growth_index)
                        )

    def distribute_extra_carbon_for_root(
        self, row_space, column_width, per_plant_area, plant_row_column
    ):
        # If extra carbon is available, it is assumed to be added to the taproot.
        if self.extra_carbon > 0:
            # available carbon for taproot growth, in g dry matter per slab.
            # ExtraCarbon is converted to availt (g dry matter per slab).
            availt = self.extra_carbon * 0.01 * row_space / per_plant_area
            # distance from the tip of the taproot, cm.
            sdl = self.taproot_length - self.last_layer_with_root_depth
            # proportionality factors for allocating added dry matter among taproot
            # soil cells.
            tpwt = np.zeros((40, 2))
            # Extra Carbon (availt) is added to soil cells with roots in the columns
            # immediately to the left and to the right of the location of the plant row
            for l in reversed(range(self.taproot_layer_number + 1)):
                # The weighting factors for allocating the carbon (tpwt) are
                # proportional to the volume of each soil cell and its distance (sdl)
                # from the tip of the taproot.
                sdl += depth_of_layer(l)
                tpwt[l][0] = sdl * depth_of_layer(l) * column_width
                tpwt[l][1] = sdl * depth_of_layer(l) * column_width
            # The proportional amount of mass is added to the mass of the last
            # (inactive) root class in each soil cell.
            self.root_weights[
                : self.taproot_layer_number + 1,
                (plant_row_column, plant_row_column + 1),
                -1,
            ] += (
                availt * tpwt[: self.taproot_layer_number + 1, :] / tpwt.sum()
            )

    def root_aging(self, l, k):
        """It updates the variable celage(l,k) for the age of roots in each soil cell
        containing roots. When root age reaches a threshold thtrn(i), a transformation
        of root tissue from class i to class i+1 occurs. The proportion transformed is
        trn(i).

        It has been adapted from the code of GOSSYM, but the threshold age for this
        process is based on the time from when the roots first grew into each soil cell
        (whereas the time from emergence was used in GOSSYM)

        NOTE: only 3 root age groups are assumed here.
        """
        # The following constant parameters are used:
        thtrn = [20.0, 40.0]  # the time threshold, from the initial
        # penetration of roots to a soil cell, after which some of the root mass of
        # class i may be transferred into the next class (i+1).
        trn = [0.0060, 0.0050]  # the daily proportion of this transfer.

        # daily average soil temperature (c) of soil cell.
        stday = self.soil_temperature[l][k] - 273.161
        self.root_age[l][k] += SoilTemOnRootGrowth(stday)

        for i in range(2):
            if self.root_age[l][k] > thtrn[i]:
                # root mass transferred from one class to the next.
                xtr = trn[i] * self.root_weights[l][k][i]
                self.root_weights[l][k][i + 1] += xtr
                self.root_weights[l][k][i] -= xtr

    def redist_root_new_growth(self, l: int, k: int, addwt: float, column_width: float, plant_row_column: int):
        """This function computes the redistribution of new growth of roots into adjacent soil cells. It is called from ActualRootGrowth().

        Redistribution is affected by the factors rgfdn, rgfsd, rgfup.
        And the values of RootGroFactor(l,k) in this soil cell and in the adjacent cells.
        The values of ActualRootGrowth(l,k) for this and for the adjacent soil cells are computed.
        The code of this module is based, with major changes, on the code of GOSSYM."""
        # The following constant parameters are used. These are relative factors for root growth to adjoining cells, downwards, sideways, and upwards, respectively. These factors are relative to the volume of the soil cell from which growth originates.
        rgfdn = 900
        rgfsd = 600
        rgfup = 10
        # Set the number of layer above and below this layer, and the number of columns to the right and to the left of this column.
        
        # layer above and below layer l.
        lp1 = min(40 - 1, l + 1)
        lm1 = max(0, l - 1)

        # column to the left and to the right of column k.
        kp1 = min(20 - 1, k + 1)
        km1 = max(0, k - 1)
        # Compute proportionality factors (efac1, efacl, efacr, efacu, efacd) as the product of RootGroFactor and the geotropic factors in the respective soil cells.
        # Note that the geotropic factors are relative to the volume of the soil cell.
        # Compute the sum srwp of the proportionality factors.
        # product of RootGroFactor and geotropic factor for this cell.
        efac1 = self.layer_depth[l] * column_width * self.root_growth_factor[l, k]
        # as efac1 for the cell to the left of this cell.
        efacl = rgfsd * self.root_growth_factor[l, km1]
        # as efac1 for the cell to the right of this cell.
        efacr = rgfsd * self.root_growth_factor[l, kp1]
        # as efac1 for the cell above this cell.
        efacu = rgfup * self.root_growth_factor[lm1, k]
        # as efac1 for the cell below this cell.
        efacd = rgfdn * self.root_growth_factor[lp1, k]
        # sum of all efac values.
        srwp = efac1 + efacl + efacr + efacu + efacd
        # If srwp is very small, all the added weight will be in the same soil soil cell, and execution of this function is ended.
        if srwp < 1e-10:
            self.actual_root_growth[l][k] = addwt
            return
        # Allocate the added dry matter to this and the adjoining soil cells in proportion to the EFAC factors.
        self.actual_root_growth[l][k] += addwt * efac1 / srwp
        self.actual_root_growth[l][km1] += addwt * efacl / srwp
        self.actual_root_growth[l][kp1] += addwt * efacr / srwp
        self.actual_root_growth[lm1][k] += addwt * efacu / srwp
        self.actual_root_growth[lp1][k] += addwt * efacd / srwp
        # If roots are growing into new soil soil cells, initialize their RootAge to 0.01.
        if self.root_age[l][km1] == 0:
            self.root_age[l][km1] = 0.01
        if self.root_age[l][kp1] == 0:
            self.root_age[l][kp1] = 0.01
        if self.root_age[lm1][k] == 0:
            self.root_age[lm1][k] = 0.01
        # If this new compartmment is in a new layer with roots, also initialize its RootColNumLeft and RootColNumRight values.
        if self.root_age[lp1][k] == 0 and efacd > 0:
            self.root_age[lp1][k] = 0.01
        # If this is in the location of the taproot, and the roots reach a new soil layer, update the taproot parameters taproot_length, self.last_layer_with_root_depth, and self.taproot_layer_number.
        if k == plant_row_column or k == plant_row_column + 1:
            if lp1 > self.taproot_layer_number and efacd > 0:
                self.taproot_length = self.last_layer_with_root_depth + 0.01
                self.last_layer_with_root_depth += self.layer_depth[lp1]
                self.taproot_layer_number = lp1

    def root_death(self, l, k):
        """This function computes the death of root tissue in each soil cell containing roots.

        When root age reaches a threshold thdth(i), a proportion dth(i) of the roots in class i dies. The mass of dead roots is added to DailyRootLoss.

        It has been adapted from GOSSYM, but the threshold age for this process is based on the time from when the roots first grew into each soil cell.

        It is assumed that root death rate is greater in dry soil, for all root classes except class 1. Root death rate is increased to the maximum value in soil saturated with water.
        """
        aa = 0.008  # a parameter in the equation for computing dthfac.
        dth = [0.0001, 0.0002, 0.0001]  # the daily proportion of death of root tissue.
        dthmax = 0.10  # a parameter in the equation for computing dthfac.
        psi0 = -14.5  # a parameter in the equation for computing dthfac.
        thdth = [30.0, 50.0, 100.0]  # the time threshold, from the initial
        # penetration of roots to a soil cell, after which death of root tissue of class i may occur.

        result = 0
        for i in range(3):
            if self.root_age[l][k] > thdth[i]:
                # the computed proportion of roots dying in each class.
                dthfac = dth[i]
                if self.soil_water_content[l, k] >= self.pore_space[l]:
                    dthfac = dthmax
                else:
                    if i <= 1 and self.soil_psi[l, k] <= psi0:
                        dthfac += aa * (psi0 - self.soil_psi[l, k])
                    if dthfac > dthmax:
                        dthfac = dthmax
                result += self.root_weights[l][k][i] * dthfac
                self.root_weights[l][k][i] -= self.root_weights[l][k][i] * dthfac
        return result
