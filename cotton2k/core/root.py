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
