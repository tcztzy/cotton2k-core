from .phenology import Stage


class StemGrowth:  # pylint: disable=too-few-public-methods,no-member
    def potential_stem_growth(  # pylint: disable=too-many-arguments
        self,
        stem_dry_weight: float,
        density_factor: float,
        var12: float,
        var13: float,
        var14: float,
        var15: float,
        var16: float,
        var17: float,
        var18: float,
    ) -> float:
        """Computes and returns the potential stem growth of cotton plants."""
        # There are two periods for computation of potential stem growth:
        # (1) Before the appearance of a square on the third fruiting branch.
        # Potential stem growth is a function of plant age (days from emergence).
        main_stem = self.vegetative_branches[0]  # type: ignore[attr-defined]
        if (
            len(main_stem.fruiting_branches) < 3
            or self.fruiting_nodes_stage[0, 2, 0] == Stage.NotYetFormed  # type: ignore
        ):
            return var12 * (var13 + var14 * self.kday)  # type: ignore
        # (2) After the appearance of a square on the third fruiting branch.
        # It is assumed that all stem tissue that is more than 32 days old is not
        # active.
        # Potential stem growth is a function of active stem tissue weight, and plant
        # density.
        # effect of plant density on stem growth rate.
        return (
            max(1.0 - var15 * (1.0 - density_factor), 0.2)
            * var16
            * (var17 + var18 * stem_dry_weight)
        )

    @staticmethod
    def add_plant_height(  # pylint: disable=too-many-arguments,too-many-locals
        density_factor: float,
        physiological_days_increment: float,
        number_of_pre_fruiting_nodes: int,
        second_fruiting_branch_stage: Stage,
        age_of_last_pre_fruiting_node: float,
        age_of_penultimate_pre_fruiting_node: float,
        average_physiological_age_of_top_three_nodes: float,
        water_stress_of_stem: float,
        carbon_stress: float,
        nitrogen_stress_of_vegetative: float,
        var19: float,
        var20: float,
        var21: float,
        var22: float,
        var23: float,
        var24: float,
        var25: float,
        var26: float,
    ) -> float:
        """This function simulates the growth in height of the main stem of cotton
        plants.
        """
        # The following constant parameters are used:
        vhtpar = [1.0, 0.27, 0.60, 0.20, 0.10, 0.26, 0.32]
        addz = 0.0  # daily plant height growth increment, cm.
        # Calculate vertical growth of main stem before the square on the second
        # fruiting branch has appeared. Added stem height (addz) is a function of the
        # age of the last prefruiting node.
        if second_fruiting_branch_stage == Stage.NotYetFormed:
            addz = vhtpar[0] - vhtpar[1] * age_of_last_pre_fruiting_node
            addz = max(min(addz, vhtpar[2]), 0)
            # It is assumed that the previous prefruiting node is also capable of
            # growth, and its growth (dz2) is added to addz.
            if number_of_pre_fruiting_nodes > 1:
                # plant height growth increment due to growth of the second node from
                # the top.
                dz2 = var19 - var20 * age_of_penultimate_pre_fruiting_node
                dz2 = min(max(dz2, 0), vhtpar[3])
                addz += dz2
            # The effect of water stress on stem height at this stage is less than at a
            # later stage (as modified by vhtpar(4)).
            addz *= 1.0 - vhtpar[4] * (1.0 - water_stress_of_stem)
        else:
            # Calculate vertical growth of main stem after the second square has
            # appeared. Added stem height (addz) is a function of the average age of
            # the upper three main stem nodes.
            addz = var21 + average_physiological_age_of_top_three_nodes * (
                var22 + var23 * average_physiological_age_of_top_three_nodes
            )
            if average_physiological_age_of_top_three_nodes > (-0.5 * var22 / var23):
                addz = var24
            addz = min(max(addz, var24), var25)
            # addz is affected by water, carbohydrate and nitrogen stresses.
            addz *= water_stress_of_stem
            addz *= 1.0 - vhtpar[5] * (1.0 - carbon_stress)
            addz *= 1.0 - vhtpar[6] * (1.0 - nitrogen_stress_of_vegetative)
        # The effect of temperature is expressed by physiological_days_increment. There
        # are also effects of plant density, and of a variety-specific calibration
        # parameter (VarPar(26)).
        return addz * var26 * physiological_days_increment * density_factor
