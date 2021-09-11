import datetime
from enum import IntEnum
from typing import Optional


class DaysToFirstSquare:  # pylint: disable=too-few-public-methods
    """This class is tricky for speed"""

    accumulated_temperature = 0
    days = 0
    accumulated_stress = 0

    def __call__(
        self, temperature, water_stress, nitrogen_stress, calibration_parameter
    ):
        # average temperature from day of emergence.
        self.days += 1
        self.accumulated_temperature += temperature
        average_temperature = min(self.accumulated_temperature / self.days, 34)
        # cumulative effect of water and N stresses on date of first square.
        self.accumulated_stress += (
            0.08 * (1 - water_stress) * 0.3 * (1 - nitrogen_stress)
        )

        return (
            132.2 + average_temperature * (-7 + average_temperature * 0.125)
        ) * calibration_parameter - self.accumulated_stress


days_to_first_square = DaysToFirstSquare()


class Stage(IntEnum):
    NotYetFormed = 0
    Square = 1
    GreenBoll = 2
    MatureBoll = 3
    AbscisedAsBoll = 4
    AbscisedAsSquare = 5
    AbscisedAsFlower = 6
    YoungGreenBoll = 7


class Phenology:  # pylint: disable=no-member,protected-access,attribute-defined-outside-init,too-many-instance-attributes
    def phenology(self):
        """Simulates events of phenology and abscission in the cotton plant."""
        u = (self.date - self._sim.start_date).days
        # The following constant parameters are used:
        vpheno = [0.65, -0.83, -1.67, -0.25, -0.75, 10.0, 15.0, 7.10]

        self.number_of_fruiting_sites = 0
        stemNRatio = (
            self.stem_nitrogen / self.stem_weight
        )  # the ratio of N to dry matter in the stems.
        # The following section is executed if the first square has not yet been formed
        # days_to_first_square() is called to compute the number of days to 1st square,
        # and method pre_fruiting_node() is called to simulate the formation of
        # prefruiting nodes.
        if self._sim.first_square_date is None:
            self._sim.DaysTo1stSqare = days_to_first_square(
                self.average_temperature,
                self.water_stress,
                self.nitrogen_stress_vegetative,
                self._sim.cultivar_parameters[30],
            )
            self.pre_fruiting_node(stemNRatio, *self._sim.cultivar_parameters[31:35])
            # When first square is formed, FirstSquare is assigned the day of year.
            # Function create_first_square() is called for formation of first square.
            if self.kday >= int(self._sim.DaysTo1stSqare):
                self._sim.first_square_date = self.date
                self.create_first_square(stemNRatio, self._sim.cultivar_parameters[34])
            # if a first square has not been formed, call LeafAbscission() and exit.
            else:
                self.leaf_abscission(
                    self._sim.per_plant_area,
                    self._sim.first_square_date,
                    self._sim.defoliate_date,
                )
                return
        # The following is executed after the appearance of the first square.
        # If there are only one or two vegetative branches, and if plant population
        # allows it, call _add_vegetative_branch() to decide if a new vegetative branch
        # is to be added. Note that dense plant populations (large per_plant_area)
        # prevent new vegetative branch formation.
        if len(self.vegetative_branches) == 1 and self._sim.per_plant_area >= vpheno[5]:
            self._sim._add_vegetative_branch(u, stemNRatio, self._sim.DaysTo1stSqare)
        if len(self.vegetative_branches) == 1 and self._sim.per_plant_area >= vpheno[6]:
            self._sim._add_vegetative_branch(u, stemNRatio, self._sim.DaysTo1stSqare)
        # The maximum number of nodes per fruiting branch (nidmax) is affected by plant
        # density. It is computed as a function of density_factor.
        nidmax = min(
            int(vpheno[7] * self._sim.density_factor + 0.5), 5
        )  # maximum number of nodes per fruiting branch.
        # Start loop over all existing vegetative branches.
        # Call AddFruitingBranch() to decide if a new node (and a new fruiting branch)
        # is to be added on this stem.
        for k, vb in enumerate(self.vegetative_branches):
            if len(vb.fruiting_branches) < 30:
                self.add_fruiting_branch(
                    k,
                    self._sim.density_factor,
                    stemNRatio,
                    self._sim.cultivar_parameters[35],
                    self._sim.cultivar_parameters[34],
                    self._sim.topping_date,
                )
            # Loop over all existing fruiting branches, and call add_fruiting_node() to
            # decide if a new node on this fruiting branch is to be added.
            for l, fb in enumerate(vb.fruiting_branches):
                if len(fb.nodes) < nidmax:
                    self.add_fruiting_node(
                        k,
                        l,
                        stemNRatio,
                        self._sim.density_factor,
                        self._sim.cultivar_parameters[34],
                        self._sim.cultivar_parameters[36],
                        self._sim.cultivar_parameters[37],
                    )
                # Loop over all existing fruiting nodes, and call
                # simulate_fruiting_site() to simulate the condition of each fruiting
                # node.
                for m in range(len(fb.nodes)):
                    first_bloom = self.simulate_fruiting_site(
                        k,
                        l,
                        m,
                        self._sim.defoliate_date,
                        self._sim.first_bloom_date,
                        self._sim.climate[u]["Tmin"],
                        self._sim.climate[u]["Tmax"],
                        self._sim.plant_population,
                        *self._sim.cultivar_parameters[38:43]
                    )
                    if first_bloom is not None:
                        self._sim.first_bloom_date = first_bloom
        # Call FruitingSitesAbscission() to simulate the abscission of fruiting parts.
        self.fruiting_sites_abscission()
        # Call LeafAbscission() to simulate the abscission of leaves.
        self.leaf_abscission(
            self._sim.per_plant_area,
            self._sim.first_square_date,
            self._sim.defoliate_date,
        )

    def simulate_fruiting_site(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        self,
        k,
        l,
        m,
        defoliate_date,
        first_bloom_date,
        min_temperature,
        max_temperature,
        plant_population,
        var38,
        var39,
        var40,
        var41,
        var42,
    ) -> Optional[datetime.date]:
        """Simulates the development of each fruiting site."""
        # The following constant parameters are used:
        vfrsite = [
            0.60,
            0.40,
            12.25,
            0.40,
            33.0,
            0.20,
            0.04,
            0.45,
            26.10,
            9.0,
            0.10,
            3.0,
            1.129,
            0.043,
            0.26,
        ]
        # FruitingCode = 0 indicates that this node has not yet been formed.
        # In this case, assign zero to boltmp and return.
        site = (
            self.vegetative_branches[k]  # type: ignore[attr-defined]
            .fruiting_branches[l]
            .nodes[m]
        )
        if site.stage == Stage.NotYetFormed:
            site.boll.cumulative_temperature = 0
            return None
        self.number_of_fruiting_sites += 1  # Increment site number.
        # LeafAge(k,l,m) is the age of the leaf at this site.
        # it is updated by adding the physiological age of this day,
        # the effect of water and nitrogen stresses (agefac).

        # effect of water and nitrogen stresses on leaf aging.
        agefac = (1 - self.water_stress) * vfrsite[0] + (  # type: ignore[attr-defined]
            1 - self.nitrogen_stress_vegetative  # type: ignore[attr-defined]
        ) * vfrsite[1]
        site.leaf.age += self.day_inc + agefac  # type: ignore[attr-defined]
        # After the application of defoliation, add the effect of defoliation on leaf
        # age.
        if defoliate_date and self.date > defoliate_date:  # type: ignore[attr-defined]
            site.leaf.age += var38
        # FruitingCode = 3, 4, 5 or 6 indicates that this node has an open boll,
        # or has been completely abscised. Return in this case.
        if site.stage in (
            Stage.MatureBoll,
            Stage.AbscisedAsBoll,
            Stage.AbscisedAsSquare,
            Stage.AbscisedAsFlower,
        ):
            return None
        # Age of node is modified for low minimum temperatures and for high maximum
        # temperatures.
        ageinc = self.day_inc  # type: ignore[attr-defined]
        # Adjust leaf aging for low minimum temperature.
        if min_temperature < vfrsite[2]:
            ageinc += vfrsite[3] * (vfrsite[2] - min_temperature)
        # Adjust leaf aging for high maximum temperature.
        if max_temperature > vfrsite[4]:
            ageinc -= min(vfrsite[6] * (max_temperature - vfrsite[4]), vfrsite[5])
        ageinc = max(ageinc, vfrsite[7])
        # Compute average temperature of this site since formation.
        site.average_temperature = (
            site.average_temperature * site.age
            + self.average_temperature * ageinc  # type: ignore[attr-defined]
        ) / (site.age + ageinc)
        # Update the age of this node, AgeOfSite(k,l,m), by adding ageinc.
        site.age += ageinc
        # The following is executed if this node is a square (Stage.Sqaure):
        # If square is old enough, make it a green boll: initialize the computations of
        # average boll temperature (boltmp) and boll age (AgeOfBoll). Stage will now be
        # Stage.YoungGreenBoll.
        if site.stage == Stage.Square:
            if site.age >= vfrsite[8]:  # type: ignore[attr-defined]
                (
                    site.boll.cumulative_temperature
                ) = self.average_temperature  # type: ignore[attr-defined]
                site.boll.age = self.day_inc  # type: ignore[attr-defined]
                site.stage = Stage.YoungGreenBoll  # type: ignore[attr-defined]
                self.new_boll_formation(site)  # type: ignore[attr-defined]
                # If this is the first flower, define FirstBloom.
                if (
                    first_bloom_date is None
                    and self.green_bolls_weight > 0  # type: ignore[attr-defined]
                ):
                    return self.date  # type: ignore[attr-defined]
            return None
        # If there is a boll at this site:
        # Calculate average boll temperature (boltmp), and boll age
        # (AgeOfBoll) which is its physiological age, modified by water stress.
        # If leaf area index is low, dum is calculated as an intermediate
        # variable. It is used to increase boll temperature and to accelerate
        # boll aging when leaf cover is decreased. Boll age is also modified
        # by nitrogen stress (state.nitrogen_stress_fruiting).
        if site.boll.weight > 0:
            # effect of leaf area index on boll temperature and age.
            if (
                self.leaf_area_index <= vfrsite[11]  # type: ignore[attr-defined]
                and self.kday > 100  # type: ignore[attr-defined]
            ):
                dum = (
                    vfrsite[12]
                    - vfrsite[13] * self.leaf_area_index  # type: ignore[attr-defined]
                )
            else:
                dum = 1
            # added physiological age of boll on this day.
            dagebol = (
                self.day_inc * dum  # type: ignore[attr-defined]
                + vfrsite[14] * (1 - self.water_stress)  # type: ignore[attr-defined]
                + vfrsite[10] * (1 - self.nitrogen_stress_fruiting)  # type: ignore
            )
            site.boll.cumulative_temperature = (
                site.boll.cumulative_temperature * site.boll.age
                + self.average_temperature * dagebol  # type: ignore[attr-defined]
            ) / (site.boll.age + dagebol)
            site.boll.age += dagebol
        # if this node is a young green boll (Stage.YoungGreenBoll):
        # Check boll age and after a fixed age convert it to an "old" green boll
        # (Stage.GreenBoll).
        if site.stage == Stage.YoungGreenBoll:
            if site.boll.age >= vfrsite[9]:
                site.stage = Stage.GreenBoll
            return None
        if site.stage == Stage.GreenBoll:
            self.boll_opening(  # type: ignore[attr-defined]
                site,
                defoliate_date,
                site.boll.cumulative_temperature,
                plant_population,
                var39,
                var40,
                var41,
                var42,
            )
        return None

    def create_first_square(self, stemNRatio, first_square_leaf_area):
        """Initiates the first square."""
        # FruitFraction and FruitingCode are assigned 1 for the first fruiting site.
        self.vegetative_branches[0].number_of_fruiting_branches = 1
        self.vegetative_branches[0].fruiting_branches[0].number_of_fruiting_nodes = 1
        first_node = self.vegetative_branches[0].fruiting_branches[0].nodes[0]
        first_node.stage = Stage.Square
        first_node.fraction = 1
        # Initialize a new leaf at this position. define its initial weight and area.
        # VarPar[34] is the initial area of a new leaf. The mass and nitrogen of the
        # new leaf are substacted from the stem.
        if self.version >= 0x500:
            leaf_weight = min(
                first_square_leaf_area * self.leaf_weight_area_ratio,
                self.stem_weight - 0.2,
            )
            leaf_area = leaf_weight / self.leaf_weight_area_ratio
        else:
            leaf_area = first_square_leaf_area
            leaf_weight = leaf_area * self.leaf_weight_area_ratio
        first_node.leaf.area = leaf_area
        first_node.leaf.weight = leaf_weight
        self.stem_weight -= leaf_weight
        self.leaf_weight += leaf_weight
        self.leaf_nitrogen += leaf_weight * stemNRatio
        self.stem_nitrogen -= leaf_weight * stemNRatio
        first_node.average_temperature = self.average_temperature
        # Define the initial values of NumFruitBranches, NumNodes,
        # self.fruit_growth_ratio, and AvrgNodeTemper.
        self.fruit_growth_ratio = 1
        # It is assumed that the cotyledons are dropped at time of first square.
        # Compute changes in AbscisedLeafWeight, self.leaf_weight, self.leaf_nitrogen
        # and self.cumulative_nitrogen_loss caused by the abscission of the cotyledons.
        cotylwt = 0.20  # cotylwt is the leaf weight of the cotyledons.
        self.leaf_weight -= cotylwt
        self.cumulative_nitrogen_loss += cotylwt * self.leaf_nitrogen / self.leaf_weight
        self.leaf_nitrogen -= cotylwt * self.leaf_nitrogen / self.leaf_weight

    def boll_opening(  # pylint: disable=too-many-arguments
        self,
        site,
        defoliate_date: Optional[datetime.date],
        tmpboll: float,
        plant_population: float,
        var39,
        var40,
        var41,
        var42,
    ):
        "Simulates the transition of each fruiting site from green to dehisced boll."
        # The following constant parameters are used:
        ddpar1 = 1
        ddpar2 = 0.8  # constant parameters for computing fdhslai.
        vboldhs = [
            30.0,
            41.189,
            -1.6057,
            0.020743,
            70.0,
            0.994,
        ]
        # Assign atn as the average boll temperature (tmpboll), and check that it is
        # not higher than a maximum value.
        atn = min(tmpboll, vboldhs[0])  # modified average temperature of this boll.
        # Compute dehiss as a function of boll temperature.
        # days from flowering to boll opening.
        dehiss = var39 + atn * (vboldhs[1] + atn * (vboldhs[2] + atn * vboldhs[3]))
        dehiss *= var40
        dehiss = min(dehiss, vboldhs[4])
        # Dehiss is decreased after a defoliation.
        if defoliate_date and self.date > defoliate_date:  # type: ignore[attr-defined]
            dehiss *= pow(
                vboldhs[5],
                (self.date - defoliate_date).days,  # type: ignore[attr-defined]
            )
        # If leaf area index is less than dpar1, decrease dehiss.
        if self.leaf_area_index < ddpar1:  # type: ignore[attr-defined]
            # effect of small lai on dehiss
            fdhslai = (
                ddpar2
                + self.leaf_area_index  # type: ignore[attr-defined]
                * (1 - ddpar2)
                / ddpar1
            )
            fdhslai = min(max(fdhslai, 0), 1)
            dehiss *= fdhslai
        if site.boll.age < dehiss:
            return
        # If green boll is old enough (AgeOfBoll greater than dehiss), make it an open
        # boll, set stage to MatureBoll, and update boll and burr weights.
        site.stage = Stage.MatureBoll
        self.open_bolls_weight += site.boll.weight  # type: ignore[attr-defined]
        self.open_bolls_burr_weight += site.burr.weight  # type: ignore[attr-defined]
        self.green_bolls_weight -= site.boll.weight  # type: ignore[attr-defined]
        self.green_bolls_burr_weight -= site.burr.weight  # type: ignore[attr-defined]
        # Compute the ginning percentage as a function of boll temperature.
        # Compute the average ginning percentage of all the bolls opened until now
        # (self.ginning_percent).
        site.ginning_percent = (var41 - var42 * atn) / 100
        self.ginning_percent = (
            self.ginning_percent  # type: ignore[has-type]
            * self.number_of_open_bolls  # type: ignore[attr-defined]
            + site.ginning_percent * site.fraction
        ) / (
            self.number_of_open_bolls + site.fraction  # type: ignore[attr-defined]
        )  # type: ignore[attr-defined]
        # Cumulative lint yield (LintYield) is computed in kg per ha.
        self.lint_yield += (  # type: ignore[attr-defined]
            site.ginning_percent * site.boll.weight * plant_population * 0.001
        )
        self.fiber_quality(atn, site.fraction)
        # Update the number of open bolls per plant (nopen).
        self.number_of_open_bolls += site.fraction  # type: ignore[attr-defined]

    def fiber_quality(self, atn, fraction):
        """Computation of fiber properties is as in GOSSYM, it is not used in COTTON2K,
        and it has not been tested. It is included here for compatibility, and it may
        be developed in future versions."""
        # fsx (fiber strength in g / tex at 1/8 inch) is computed, and averaged (as
        # fiber_strength) for all open bolls.
        fsx = 56.603 + atn * (
            -2.921 + 0.059 * atn
        )  # fiber strength (g / tex at 1/8 inch) of this boll.
        # flx (fiber length in inches, 2.5% span) is computed, and averaged (as
        # fiber_length) for all open bolls.
        flx = 1.219 - 0.0065 * atn  # fiber length (inches, 2.5% span) of this boll.
        self.fiber_strength = (  # type: ignore[attr-defined]
            self.fiber_strength * self.number_of_open_bolls
            + fsx * fraction  # type: ignore[attr-defined]
        ) / (
            self.number_of_open_bolls + fraction  # type: ignore[attr-defined]
        )
        self.fiber_length = (  # type: ignore[attr-defined]
            self.fiber_length * self.number_of_open_bolls
            + flx * fraction  # type: ignore[attr-defined]
        ) / (
            self.number_of_open_bolls + fraction  # type: ignore[attr-defined]
        )

    # pylint: disable=access-member-before-definition
    def leaf_abscission(self, per_plant_area, first_square_date, defoliate_date):
        # If there are almost no leaves, this routine is not executed.
        if self.leaf_area_index <= 0.0001:
            return
        # Compute droplf as a function of LeafAreaIndex.
        p0 = 140 if self.version < 0x500 else 100
        p1 = -1
        droplf = p0 + p1 * self.leaf_area_index  # leaf age until its abscission.
        # Simulate the physiological abscission of prefruiting node leaves.
        self.pre_fruiting_node_leaf_abscission(
            droplf, first_square_date, defoliate_date
        )
        # Loop for all vegetative branches and fruiting branches, and call
        # main_stem_leaf_abscission() for each fruiting branch to simulate the
        # physiological abscission of the other leaves.
        for k in range(self.number_of_vegetative_branches):
            for l in range(self.vegetative_branches[k].number_of_fruiting_branches):
                self.main_stem_leaf_abscission(k, l, droplf)
        # Call defoliation_leaf_abscission() to simulate leaf abscission caused by
        # defoliants.
        if defoliate_date is not None and self.date >= defoliate_date:
            self.defoliation_leaf_abscission(defoliate_date)
        # If the reserves in the leaf are too high, add the lost reserves to
        # AbscisedLeafWeight and adjust reserve_carbohydrate.
        if self.reserve_carbohydrate > 0:
            self.reserve_carbohydrate = min(
                self.reserve_carbohydrate,
                0.2 * self.leaf_weight,
            )
        # Compute the resulting LeafAreaIndex but do not let it get too small.
        self.leaf_area_index = max(0.0001, self.leaf_area / per_plant_area)

    def main_stem_leaf_abscission(self, k, l, droplf):
        """Simulate the abscission of main stem leaves on node l of vegetative branch k

        Arguments
        ---------
        droplf
            leaf age until it is abscised.
        k, l
            numbers of this vegetative branch and fruiting branch.
        """
        # The leaf on this main stem node is abscised if its age has reached droplf,
        # and if there is a leaf here, and if LeafAreaIndex is not too small:
        # Update self.leaf_area, AbscisedLeafWeight, self.leaf_weight,
        # self.petiole_weight, state.leaf_nitrogen, CumPlantNLoss.
        # Assign zero to LeafAreaMainStem, PetioleWeightMainStem and LeafWeightMainStem
        # of this leaf.
        # If this is after defoliation.
        main_stem_leaf = self.vegetative_branches[k].fruiting_branches[l].main_stem_leaf
        if (
            self.vegetative_branches[k].fruiting_branches[l].nodes[0].leaf.age > droplf
            and main_stem_leaf.area > 0
            and self.leaf_area_index > 0.1
        ):
            self.leaf_weight -= main_stem_leaf.weight
            self.petiole_weight -= main_stem_leaf.petiole_weight
            self.leaf_nitrogen -= (
                main_stem_leaf.weight * self.leaf_nitrogen_concentration
            )
            self.petiole_nitrogen -= (
                main_stem_leaf.petiole_weight * self.petiole_nitrogen_concentration
            )
            self.cumulative_nitrogen_loss += (
                main_stem_leaf.weight * self.leaf_nitrogen_concentration
                + main_stem_leaf.petiole_weight * self.petiole_nitrogen_concentration
            )
            self.leaf_area -= main_stem_leaf.area
            main_stem_leaf.area = 0
            main_stem_leaf.weight = 0
            main_stem_leaf.petiole_weight = 0
        # Loop over all nodes on this fruiting branch and call
        # fruit_node_leaf_abscission().
        for m in range(
            self.vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes
        ):
            self.fruit_node_leaf_abscission(k, l, m, droplf)

    def fruit_node_leaf_abscission(self, k, l, m, droplf):
        """Simulates the abscission of fruiting node leaves on node m of fruiting
        branch l of vegetative branch k.

        Arguments
        ---------
        droplf
            leaf age until it is abscised.
        k, l
            numbers of this vegetative branch and fruiting branch.
        m
            node number on this fruiting branch.
        """
        # The leaf on this fruiting node is abscised if its age has reached droplf, and
        # if there is a leaf here, and if LeafAreaIndex is not too small:

        # Update state.leaf_area, AbscisedLeafWeight, self.leaf_weight,
        # self.petiole_weight, state.leaf_nitrogen, CumPlantNLoss,
        # Assign zero to LeafAreaNodes, PetioleWeightNodes and LeafWeightNodes of this
        # leaf.
        # If this is after defoliation.
        site = self.vegetative_branches[k].fruiting_branches[l].nodes[m]
        if (
            site.leaf.age >= droplf
            and site.leaf.area > 0
            and self.leaf_area_index > 0.1
        ):
            self.leaf_weight -= site.leaf.weight
            self.petiole_weight -= site.petiole.weight
            self.leaf_nitrogen -= site.leaf.weight * self.leaf_nitrogen_concentration
            self.petiole_nitrogen -= (
                site.petiole.weight * self.petiole_nitrogen_concentration
            )
            self.cumulative_nitrogen_loss += (
                site.leaf.weight * self.leaf_nitrogen_concentration
                + site.petiole.weight * self.petiole_nitrogen_concentration
            )
            self.leaf_area -= site.leaf.area
            site.leaf.area = 0
            site.leaf.weight = 0
            site.petiole.weight = 0
