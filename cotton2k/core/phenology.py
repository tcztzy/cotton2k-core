import datetime
from collections.abc import Sequence
from enum import IntEnum
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


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
    """code indicating the developmental state of each fruiting site"""

    NotYetFormed = 0
    Square = 1
    GreenBoll = 2  # not susceptible to shedding
    MatureBoll = 3
    AbscisedAsBoll = 4
    AbscisedAsSquare = 5
    AbscisedAsFlower = 6
    YoungGreenBoll = 7  # susceptible to shedding


# pylint: disable=E1101,E0203,R0902,R0912,R0913,R0914,W0201
class Phenology:
    _sim: Any
    average_temperature: float
    burr_weight: npt.NDArray[np.double]
    date: datetime.date
    day_inc: float
    fruiting_nodes_age: npt.NDArray[np.double]
    fruiting_nodes_average_temperature: npt.NDArray[np.double]
    fruiting_nodes_boll_cumulative_temperature: npt.NDArray[np.double]
    fruiting_nodes_boll_age: npt.NDArray[np.double]
    fruiting_nodes_boll_weight: npt.NDArray[np.double]
    fruiting_nodes_fraction: npt.NDArray[np.double]
    fruiting_nodes_ginning_percent: npt.NDArray[np.double]
    fruiting_nodes_stage: npt.NDArray[np.int_]
    node_leaf_age: npt.NDArray[np.double]
    ginning_percent: float
    green_bolls_burr_weight: float
    green_bolls_weight: float
    kday: int
    nitrogen_stress_fruiting: float
    nitrogen_stress_vegetative: float
    number_of_open_bolls: float
    open_bolls_burr_weight: float
    water_stress: float

    @property
    def phenological_delay_for_vegetative_by_carbon_stress(self):
        """delay in formation of new fruiting branches caused by carbon stress."""
        delay = np.polynomial.Polynomial(
            [self._sim.cultivar_parameters[27], -0.25, -0.75]
        )(self.carbon_stress)
        return min(max(delay, 0), 1)

    @property
    def phenological_delay_for_fruiting_by_carbon_stress(self):
        """delay in formation of new fruiting sites caused by carbon stress."""
        delay = np.polynomial.Polynomial(
            [self._sim.cultivar_parameters[28], -0.83, -1.67]
        )(self.carbon_stress)
        return min(max(delay, 0), self._sim.cultivar_parameters[29])

    def phenology(self):
        """Simulates events of phenology and abscission in the cotton plant."""
        u = (self.date - self._sim.start_date).days
        # The following constant parameters are used:
        vpheno = [0.65, -0.83, -1.67, -0.25, -0.75, 10.0, 15.0, 7.10]

        stemNRatio = (
            self.stem_nitrogen / self.stem_weight
        )  # the ratio of N to dry matter in the stems.
        # The following section is executed if the first square has not yet been formed
        # days_to_first_square() is called to compute the number of days to 1st square,
        # and method pre_fruiting_node() is called to simulate the formation of
        # prefruiting nodes.
        if self._sim.first_square_date is None:
            DaysTo1stSqare = days_to_first_square(
                self.average_temperature,
                self.water_stress,
                self.nitrogen_stress_vegetative,
                self._sim.cultivar_parameters[30],
            )
            self.pre_fruiting_node(stemNRatio, *self._sim.cultivar_parameters[31:35])
            # When first square is formed, FirstSquare is assigned the day of year.
            # Function create_first_square() is called for formation of first square.
            if self.kday >= int(DaysTo1stSqare):
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
        # allows it, call add_vegetative_branch() to decide if a new vegetative branch
        # is to be added. Note that dense plant populations (large per_plant_area)
        # prevent new vegetative branch formation.
        if (
            self.fruiting_nodes_stage[1] == Stage.NotYetFormed
        ).all() and self._sim.per_plant_area >= vpheno[5]:
            self.add_vegetative_branch(u, stemNRatio, DaysTo1stSqare)
        if (
            self.fruiting_nodes_stage[2] == Stage.NotYetFormed
        ).all() and self._sim.per_plant_area >= vpheno[6]:
            self.add_vegetative_branch(u, stemNRatio, DaysTo1stSqare)
        # The maximum number of nodes per fruiting branch (nidmax) is affected by plant
        # density. It is computed as a function of density_factor.
        nidmax = min(
            int(vpheno[7] * self._sim.density_factor + 0.5), 5
        )  # maximum number of nodes per fruiting branch.
        # Start loop over all existing vegetative branches.
        # Call AddFruitingBranch() to decide if a new node (and a new fruiting branch)
        # is to be added on this stem.
        for k in range(3):
            number_of_branches = (
                (self.fruiting_nodes_stage[k] != Stage.NotYetFormed).sum(axis=1) > 0
            ).sum()
            if number_of_branches < 30:
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
            number_of_branches = (
                (self.fruiting_nodes_stage[k] != Stage.NotYetFormed).sum(axis=1) > 0
            ).sum()
            for l in range(number_of_branches):
                number_of_nodes = (
                    self.fruiting_nodes_stage[k, l] != Stage.NotYetFormed
                ).sum()
                if number_of_nodes < nidmax:
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
                number_of_nodes = (
                    self.fruiting_nodes_stage[k, l] != Stage.NotYetFormed
                ).sum()
                for m in range(number_of_nodes):
                    first_bloom = self.simulate_fruiting_site(
                        k,
                        l,
                        m,
                        self._sim.defoliate_date,
                        self._sim.first_bloom_date,
                        self._sim.meteor[self.date]["tmin"],
                        self._sim.meteor[self.date]["tmax"],
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

    def simulate_fruiting_site(
        self,
        k,
        l,
        m,
        defoliate_date,
        first_bloom_date,
        min_temperature,
        max_temperature,
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
        if self.fruiting_nodes_stage[k, l, m] == Stage.NotYetFormed:
            self.fruiting_nodes_boll_cumulative_temperature[k, l, m] = 0
            return None
        # LeafAge(k,l,m) is the age of the leaf at this site.
        # it is updated by adding the physiological age of this day,
        # the effect of water and nitrogen stresses (agefac).

        # effect of water and nitrogen stresses on leaf aging.
        agefac = (1 - self.water_stress) * vfrsite[0] + (
            1 - self.nitrogen_stress_vegetative
        ) * vfrsite[1]
        self.node_leaf_age[k, l, m] += self.day_inc + agefac
        # After the application of defoliation, add the effect of defoliation on leaf
        # age.
        if defoliate_date and self.date > defoliate_date:
            self.node_leaf_age[k, l, m] += var38
        # FruitingCode = 3, 4, 5 or 6 indicates that this node has an open boll,
        # or has been completely abscised. Return in this case.
        if self.fruiting_nodes_stage[k, l, m] in (
            Stage.MatureBoll,
            Stage.AbscisedAsBoll,
            Stage.AbscisedAsSquare,
            Stage.AbscisedAsFlower,
        ):
            return None
        # Age of node is modified for low minimum temperatures and for high maximum
        # temperatures.
        ageinc = self.day_inc
        # Adjust leaf aging for low minimum temperature.
        if min_temperature < vfrsite[2]:
            ageinc += vfrsite[3] * (vfrsite[2] - min_temperature)
        # Adjust leaf aging for high maximum temperature.
        if max_temperature > vfrsite[4]:
            ageinc -= min(vfrsite[6] * (max_temperature - vfrsite[4]), vfrsite[5])
        ageinc = max(ageinc, vfrsite[7])
        # Compute average temperature of this site since formation.
        self.fruiting_nodes_average_temperature[k, l, m] = (
            self.fruiting_nodes_average_temperature[k, l, m]
            * self.fruiting_nodes_age[k, l, m]
            + self.average_temperature * ageinc
        ) / (self.fruiting_nodes_age[k, l, m] + ageinc)
        # Update the age of this node, AgeOfSite(k,l,m), by adding ageinc.
        self.fruiting_nodes_age[k, l, m] += ageinc
        # The following is executed if this node is a square (Stage.Sqaure):
        # If square is old enough, make it a green boll: initialize the computations of
        # average boll temperature (boltmp) and boll age (AgeOfBoll). Stage will now be
        # Stage.YoungGreenBoll.
        if self.fruiting_nodes_stage[k, l, m] == Stage.Square:
            if self.fruiting_nodes_age[k, l, m] >= vfrsite[8]:
                self.fruiting_nodes_boll_cumulative_temperature[
                    k, l, m
                ] = self.average_temperature
                self.fruiting_nodes_boll_age[k, l, m] = self.day_inc
                self.fruiting_nodes_stage[k, l, m] = Stage.YoungGreenBoll
                self.new_boll_formation((k, l, m))
                # If this is the first flower, define FirstBloom.
                if first_bloom_date is None and self.green_bolls_weight > 0:
                    return self.date
            return None
        # If there is a boll at this site:
        # Calculate average boll temperature (boltmp), and boll age
        # (AgeOfBoll) which is its physiological age, modified by water stress.
        # If leaf area index is low, dum is calculated as an intermediate
        # variable. It is used to increase boll temperature and to accelerate
        # boll aging when leaf cover is decreased. Boll age is also modified
        # by nitrogen stress (state.nitrogen_stress_fruiting).
        if self.fruiting_nodes_boll_weight[k, l, m] > 0:
            # effect of leaf area index on boll temperature and age.
            if self.leaf_area_index <= vfrsite[11] and self.kday > 100:
                dum = vfrsite[12] - vfrsite[13] * self.leaf_area_index
            else:
                dum = 1
            # added physiological age of boll on this day.
            dagebol = (
                self.day_inc * dum
                + vfrsite[14] * (1 - self.water_stress)
                + vfrsite[10] * (1 - self.nitrogen_stress_fruiting)
            )
            self.fruiting_nodes_boll_cumulative_temperature[k, l, m] = (
                self.fruiting_nodes_boll_cumulative_temperature[k, l, m]
                * self.fruiting_nodes_boll_age[k, l, m]
                + self.average_temperature * dagebol
            ) / (self.fruiting_nodes_boll_age[k, l, m] + dagebol)
            self.fruiting_nodes_boll_age[k, l, m] += dagebol
        # if this node is a young green boll (Stage.YoungGreenBoll):
        # Check boll age and after a fixed age convert it to an "old" green boll
        # (Stage.GreenBoll).
        if self.fruiting_nodes_stage[k, l, m] == Stage.YoungGreenBoll:
            if self.fruiting_nodes_boll_age[k, l, m] >= vfrsite[9]:
                self.fruiting_nodes_stage[k, l, m] = Stage.GreenBoll
            return None
        if self.fruiting_nodes_stage[k, l, m] == Stage.GreenBoll:
            self.boll_opening(
                (k, l, m),
                defoliate_date,
                var39,
                var40,
                var41,
                var42,
            )
        return None

    @property
    def pollination_switch(self):
        """pollination switch: false = no pollination, true = yes."""
        # Set 'pollination switch' for rainy days (as in GOSSYM).
        return self.rain < 2.5

    def new_boll_formation(self, index):
        """Simulates the formation of a new boll at a fruiting site."""
        # The following constant parameters are used:
        seedratio = 0.64  # ratio of seeds in seedcotton weight.
        vnewboll = [0.31, 0.02]
        # If bPollinSwitch is false accumulate number of blooms to be dropped, and
        # define FruitingCode as 6.
        if not self.pollination_switch:
            self.fruiting_nodes_stage[index] = Stage.AbscisedAsFlower
            self.fruiting_nodes_fraction[index] = 0
            self.square_weights[index] = 0
            return
        # The initial weight of the new boll (BollWeight) and new burr (burr_weight)
        # will be a fraction of the square weight, and the rest will be added to
        # BloomWeightLoss. 80% of the initial weight will be in the burr.
        # The nitrogen in the square is partitioned in the same proportions. The
        # nitrogen that was in the square is transferred to the burrs. Update
        # green_bolls_burr_weight. assign zero to SquareWeight at this site.
        # initial weight of boll after flowering.
        bolinit = vnewboll[0] * self.square_weights[index]
        self.fruiting_nodes_boll_weight[index] = 0.2 * bolinit
        self.burr_weight[index] = bolinit - self.fruiting_nodes_boll_weight[index]

        # the nitrogen content of one square before flowering.
        sqr1n = self.square_nitrogen_concentration * self.square_weights[index]
        self.square_nitrogen -= sqr1n
        sqr1n = sqr1n * vnewboll[0]

        # the nitrogen content of seeds in a new boll on flowering.
        seed1n = min(
            self.fruiting_nodes_boll_weight[index] * seedratio * vnewboll[1], sqr1n
        )
        self.seed_nitrogen += seed1n
        self.burr_nitrogen += sqr1n - seed1n

        self.green_bolls_burr_weight += self.burr_weight[index]
        self.square_weights[index] = 0

    def add_vegetative_branch(self, stemNRatio, DaysTo1stSqare, initial_leaf_area):
        """Decides whether a new vegetative branch is to be added, and then forms it."""
        for k in reversed(range(3)):
            if (self.fruiting_nodes_stage[k] != Stage.NotYetFormed).any():
                break
        if k == 2:
            return
        # TimeToNextVegBranch is computed as a function of this average temperature.
        # time, in physiological days, for the next vegetative branch to be formed.
        TimeToNextVegBranch = np.polynomial.Polynomial([13.39, -0.696, 0.012])(
            self.fruiting_nodes_average_temperature[k, 0, 0]
        )
        # Compare the age of the first fruiting site of the last formed vegetative
        # branch with TimeToNextVegBranch plus DaysTo1stSqare and the delays caused by
        # stresses, in order to decide if a new vegetative branch is to be formed.
        if (
            self.fruiting_nodes_age[k, 0, 0]
            < TimeToNextVegBranch
            + self.phenological_delay_for_vegetative_by_carbon_stress
            + self.phenological_delay_by_nitrogen_stress
            + DaysTo1stSqare
        ):
            return
        # Assign 1 to FruitFraction and FruitingCode of the first site of this branch.
        index = (k + 1, 0, 0)
        self.fruiting_nodes_fraction[index] = 1
        self.fruiting_nodes_stage[index] = Stage.Square
        # Add a new leaf to the first site of this branch.
        self.node_leaf_area[index] = initial_leaf_area
        self.node_leaf_weight[index] = initial_leaf_area * self.leaf_weight_area_ratio
        # Add a new mainstem leaf to the first node of this branch.
        self.main_stem_leaf_area[index[:2]] = initial_leaf_area
        self.main_stem_leaf_weight[index[:2]] = (
            initial_leaf_area * self.leaf_weight_area_ratio
        )
        # The initial mass and nitrogen in the new leaves are substracted from the stem.
        self.stem_weight -= (
            self.node_leaf_weight[index] + self.main_stem_leaf_weight[index[:2]]
        )
        self.leaf_weight += (
            self.node_leaf_weight[index] + self.main_stem_leaf_weight[index[:2]]
        )
        # nitrogen moved to new leaves from stem.
        addlfn = (
            self.node_leaf_weight[index] + self.main_stem_leaf_weight[index[:2]]
        ) * stemNRatio
        self.leaf_nitrogen += addlfn
        self.stem_nitrogen -= addlfn
        # Assign the initial value of the average temperature of the first site.
        # Define initial NumFruitBranches and NumNodes for the new vegetative branch.
        self.fruiting_nodes_average_temperature[index] = self.average_temperature

    def add_fruiting_branch(
        self,
        k,
        density_factor,
        stemNRatio,
        time_to_new_fruiting_branch,
        new_node_initial_leaf_area,
        topping_date=None,
    ):
        """Decides if a new fruiting branch is to be added to a vegetative branch, and
        forms it."""
        if topping_date is not None and self.date >= topping_date:
            return
        # The following constant parameters are used:
        vfrtbr = [0.8, 0.95, 33.0, 4.461, -0.1912, 0.00265, 1.8, -1.32]
        # Compute the cumulative delay for the appearance of the next caused by
        # carbohydrate, nitrogen, and water stresses.
        self.delay_of_new_fruiting_branch[k] += (
            self.phenological_delay_for_vegetative_by_carbon_stress
            + vfrtbr[0] * self.phenological_delay_by_nitrogen_stress
        )
        self.delay_of_new_fruiting_branch[k] += vfrtbr[1] * (1 - self.water_stress)
        # Define nbrch and compute TimeToNextFruBranch, the time in physiological days
        # needed for the formation of each successive fruiting branch, as a function of
        # the average temperature. This function is derived from data of K. R. Reddy,
        # CSRU, adjusted for age expressed in physiological days.

        # It is different for the main stem (k = 0) than for the other vegetative
        # branches. TimeToNextFruNode is modified for plant density.
        # modified average daily temperature.
        for l in reversed(range(30)):
            if (self.fruiting_nodes_stage[k, l] != Stage.NotYetFormed).any():
                break
        else:
            l = -1
        tav = min(
            self.fruiting_nodes_average_temperature[k, l, 0],
            vfrtbr[2],
        )
        # TimeToNextFruBranch is the time, in physiological days, for the next fruiting
        # branch to be formed.
        # Add DelayNewFruBranch to TimeToNextFruNode.
        TimeToNextFruBranch = time_to_new_fruiting_branch + tav * (
            vfrtbr[3] + tav * (vfrtbr[4] + tav * vfrtbr[5])
        )
        if k > 0:
            TimeToNextFruBranch = TimeToNextFruBranch * vfrtbr[6]
        TimeToNextFruBranch = (
            TimeToNextFruBranch * (1 + vfrtbr[7] * (1 - density_factor))
            + self.delay_of_new_fruiting_branch[k]
        )
        # Check if the the age of the last fruiting branch exceeds TimeToNextFruBranch.
        # If so, form the new fruiting branch:
        if self.fruiting_nodes_age[k, l, 0] < TimeToNextFruBranch:
            return
        # Increment NumFruitBranches, define newbr, and assign 1 to NumNodes,
        # FruitFraction and FruitingCode.
        if l >= 29:
            return
        if self.version >= 0x500:
            leaf_weight = max(
                min(
                    new_node_initial_leaf_area * self.leaf_weight_area_ratio,
                    self.stem_weight - 0.2,
                ),
                0,
            )
            leaf_area = leaf_weight / self.leaf_weight_area_ratio
        else:
            leaf_area = new_node_initial_leaf_area
            leaf_weight = leaf_area * self.leaf_weight_area_ratio
        # the index number of the new fruiting branch on this vegetative branch, after
        # a new branch has been added.
        l += 1
        self.fruiting_nodes_fraction[k, l, 0] = 1
        self.fruiting_nodes_stage[k, l, 0] = Stage.Square
        # Initiate new leaves at the first node of the new fruiting branch, and at the
        # corresponding main stem node. The mass and nitrogen in the new leaves is
        # substacted from the stem.
        self.node_leaf_area[k, l, 0] = leaf_area
        self.node_leaf_weight[k, l, 0] = leaf_weight

        self.main_stem_leaf_area[k, l] = leaf_area
        self.main_stem_leaf_weight[k, l] = leaf_weight
        self.stem_weight -= (
            self.main_stem_leaf_weight[k, l] + self.node_leaf_weight[k, l, 0]
        )
        self.leaf_weight += (
            self.main_stem_leaf_weight[k, l] + self.node_leaf_weight[k, l, 0]
        )
        # addlfn is the nitrogen added to new leaves from stem.
        addlfn = (
            self.main_stem_leaf_weight[k, l] + self.node_leaf_weight[k, l, 0]
        ) * stemNRatio
        self.leaf_nitrogen += addlfn
        self.stem_nitrogen -= addlfn
        # Begin computing AvrgNodeTemper of the new node and assign zero to
        # DelayNewFruBranch.
        self.fruiting_nodes_average_temperature[k, l, 0] = self.average_temperature
        self.delay_of_new_fruiting_branch[k] = 0

    def add_fruiting_node(self, k, l, stemNRatio, density_factor, var34, var36, var37):
        """Decide if a new node is to be added to a fruiting branch, and forms it."""
        # The following constant parameters are used:
        vfrtnod = [1.32, 0.90, 33.0, 7.6725, -0.3297, 0.004657]
        # Compute the cumulative delay for the appearance of the next node on the
        # fruiting branch, caused by carbohydrate, nitrogen, and water stresses.
        self.node_delay[k, l] += (
            self.phenological_delay_for_fruiting_by_carbon_stress
            + vfrtnod[0] * self.phenological_delay_by_nitrogen_stress
        )
        self.node_delay[k, l] += vfrtnod[1] * (1 - self.water_stress)
        # Define nnid, and compute the average temperature of the last node of this
        # fruiting branch, from the time it was formed.
        for nnid in reversed(range(5)):
            if self.fruiting_nodes_stage[k, l, nnid] != Stage.NotYetFormed:
                break
        tav = min(
            self.fruiting_nodes_average_temperature[k, l, nnid],
            vfrtnod[2],
        )  # modified daily average temperature.

        # Compute TimeToNextFruNode, the time (in physiological days) needed for the
        # formation of each successive node on the fruiting branch. This is a function
        # of temperature, derived from data of K. R. Reddy, CSRU, adjusted for age in
        # physiological days. It is modified for plant density.

        # time, in physiological days, for the next node on the fruiting branch to be
        # formed
        TimeToNextFruNode = var36 + tav * (
            vfrtnod[3] + tav * (vfrtnod[4] + tav * vfrtnod[5])
        )
        TimeToNextFruNode = (
            TimeToNextFruNode * (1 + var37 * (1 - density_factor))
            + self.node_delay[k, l]
        )
        # Check if the the age of the last node on the fruiting branch exceeds
        # TimeToNextFruNode.
        # If so, form the new node:
        if self.fruiting_nodes_age[k, l, nnid] < TimeToNextFruNode or nnid >= 4:
            return
        # Increment NumNodes, define newnod, and assign 1 to FruitFraction and
        # FruitingCode.
        if self.version >= 0x500:
            leaf_weight = min(
                var34 * self.leaf_weight_area_ratio, self.stem_weight - 0.2
            )
            if leaf_weight <= 0:
                return
            leaf_area = leaf_weight / self.leaf_weight_area_ratio
        else:
            leaf_area = var34
            leaf_weight = leaf_area * self.leaf_weight_area_ratio
        self.fruiting_nodes_fraction[k, l, nnid + 1] = 1
        self.fruiting_nodes_stage[k, l, nnid + 1] = Stage.Square
        # Initiate a new leaf at the new node. The mass and nitrogen in the new leaf is
        # substacted from the stem.
        self.node_leaf_area[k, l, nnid + 1] = leaf_area
        self.node_leaf_weight[k, l, nnid + 1] = leaf_weight
        self.stem_weight -= leaf_weight
        self.leaf_weight += leaf_weight
        self.leaf_nitrogen += leaf_weight * stemNRatio
        self.stem_nitrogen -= leaf_weight * stemNRatio
        # Begin computing AvrgNodeTemper of the new node, and assign zero to
        # DelayNewNode.
        self.fruiting_nodes_average_temperature[
            k, l, nnid + 1
        ] = self.average_temperature
        self.node_delay[k, l] = 0

    def create_first_square(self, stemNRatio, first_square_leaf_area):
        """Initiates the first square."""
        # FruitFraction and FruitingCode are assigned 1 for the first fruiting site.
        self.fruiting_nodes_stage[0, 0, 0] = Stage.Square
        self.fruiting_nodes_fraction[0, 0, 0] = 1
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
        self.node_leaf_area[0, 0, 0] = leaf_area
        self.node_leaf_weight[0, 0, 0] = leaf_weight
        self.stem_weight -= leaf_weight
        self.leaf_weight += leaf_weight
        self.leaf_nitrogen += leaf_weight * stemNRatio
        self.stem_nitrogen -= leaf_weight * stemNRatio
        self.fruiting_nodes_average_temperature[0, 0, 0] = self.average_temperature
        # Define the initial values of NumFruitBranches, NumNodes,
        # self.fruit_growth_ratio, and AvrgNodeTemper.
        self.fruit_growth_ratio = 1
        # It is assumed that the cotyledons are dropped at time of first square.
        # Compute changes in AbscisedLeafWeight, self.leaf_weight, self.leaf_nitrogen
        # caused by the abscission of the cotyledons.
        cotylwt = 0.20  # cotylwt is the leaf weight of the cotyledons.
        self.leaf_weight -= cotylwt
        self.leaf_nitrogen -= cotylwt * self.leaf_nitrogen / self.leaf_weight

    def boll_opening(
        self,
        site_index: tuple[int, int, int],
        defoliate_date: Optional[datetime.date],
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
        atn = min(  # modified average temperature of this boll.
            self.fruiting_nodes_boll_cumulative_temperature[site_index], vboldhs[0]
        )
        # Compute dehiss as a function of boll temperature.
        # days from flowering to boll opening.
        dehiss = var39 + atn * (vboldhs[1] + atn * (vboldhs[2] + atn * vboldhs[3]))
        dehiss *= var40
        dehiss = min(dehiss, vboldhs[4])
        # Dehiss is decreased after a defoliation.
        if defoliate_date and self.date > defoliate_date:
            dehiss *= pow(
                vboldhs[5],
                (self.date - defoliate_date).days,
            )
        # If leaf area index is less than dpar1, decrease dehiss.
        if self.leaf_area_index < ddpar1:
            # effect of small lai on dehiss
            fdhslai = ddpar2 + self.leaf_area_index * (1 - ddpar2) / ddpar1
            fdhslai = min(max(fdhslai, 0), 1)
            dehiss *= fdhslai
        if self.fruiting_nodes_boll_age[site_index] < dehiss:
            return
        # If green boll is old enough (AgeOfBoll greater than dehiss), make it an open
        # boll, set stage to MatureBoll, and update boll and burr weights.
        self.fruiting_nodes_stage[site_index] = Stage.MatureBoll
        self.open_bolls_burr_weight += self.burr_weight[site_index]
        self.green_bolls_burr_weight -= self.burr_weight[site_index]
        # Compute the ginning percentage as a function of boll temperature.
        # Compute the average ginning percentage of all the bolls opened until now
        # (self.ginning_percent).
        self.fruiting_nodes_ginning_percent[site_index] = (var41 - var42 * atn) / 100
        self.ginning_percent = (
            self.ginning_percent * self.number_of_open_bolls
            + self.fruiting_nodes_ginning_percent[site_index]
            * self.fruiting_nodes_fraction[site_index]
        ) / (self.number_of_open_bolls + self.fruiting_nodes_fraction[site_index])
        self.fiber_quality(atn, self.fruiting_nodes_fraction[site_index])

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
        self.fiber_strength = (
            self.fiber_strength * self.number_of_open_bolls + fsx * fraction
        ) / (self.number_of_open_bolls + fraction)
        self.fiber_length = (
            self.fiber_length * self.number_of_open_bolls + flx * fraction
        ) / (self.number_of_open_bolls + fraction)

    def leaf_abscission(self, per_plant_area, first_square_date, defoliate_date):
        # If there are almost no leaves, this routine is not executed.
        if self.leaf_area_index <= 0.0001:
            return
        # Compute droplf as a function of LeafAreaIndex.
        p0 = 140
        p1 = -1
        droplf = p0 + p1 * self.leaf_area_index  # leaf age until its abscission.
        # Simulate the physiological abscission of prefruiting node leaves.
        self.pre_fruiting_node_leaf_abscission(
            droplf, first_square_date, defoliate_date
        )
        # Loop for all vegetative branches and fruiting branches, and call
        # main_stem_leaf_abscission() for each fruiting branch to simulate the
        # physiological abscission of the other leaves.
        for i, w in np.ndenumerate(self.main_stem_leaf_weight):
            if w:
                self.main_stem_leaf_abscission(*i, droplf)
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
        # Update AbscisedLeafWeight, self.leaf_weight,
        # self.petiole_weight, state.leaf_nitrogen, CumPlantNLoss.
        # Assign zero to LeafAreaMainStem, PetioleWeightMainStem and LeafWeightMainStem
        # of this leaf.
        # If this is after defoliation.
        if (
            self.node_leaf_age[k, l, 0] > droplf
            and self.main_stem_leaf_area[k, l] > 0
            and self.leaf_area_index > 0.1
        ):
            self.leaf_nitrogen -= (
                self.main_stem_leaf_weight[k, l] * self.leaf_nitrogen_concentration
            )
            self.leaf_weight -= self.main_stem_leaf_weight[k, l]
            self.petiole_nitrogen -= (
                self.main_stem_leaf_petiole_weight[k, l]
                * self.petiole_nitrogen_concentration
            )
            self.petiole_weight -= self.main_stem_leaf_petiole_weight[k, l]
            self.main_stem_leaf_area[k, l] = 0
            self.main_stem_leaf_weight[k, l] = 0
            self.main_stem_leaf_petiole_weight[k, l] = 0
        # Loop over all nodes on this fruiting branch and call
        # fruit_node_leaf_abscission().
        for m in range(5):
            if self.node_leaf_weight[k, l, m]:
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

        # Update AbscisedLeafWeight, self.leaf_weight,
        # self.petiole_weight, state.leaf_nitrogen, CumPlantNLoss,
        # Assign zero to LeafAreaNodes, PetioleWeightNodes and LeafWeightNodes of this
        # leaf.
        # If this is after defoliation.
        if (
            self.node_leaf_age[k, l, m] >= droplf
            and self.node_leaf_area[k, l, m] > 0
            and self.leaf_area_index > 0.1
        ):
            self.leaf_weight -= self.node_leaf_weight[k, l, m]
            self.petiole_weight -= self.node_petiole_weight[k, l, m]
            self.leaf_nitrogen -= (
                self.node_leaf_weight[k, l, m] * self.leaf_nitrogen_concentration
            )
            self.petiole_nitrogen -= (
                self.node_petiole_weight[k, l, m] * self.petiole_nitrogen_concentration
            )
            self.node_leaf_area[k, l, m] = 0
            self.node_leaf_weight[k, l, m] = 0
            self.node_petiole_weight[k, l, m] = 0
