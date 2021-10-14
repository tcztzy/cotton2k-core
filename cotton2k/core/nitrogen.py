from functools import cached_property
from typing import Sequence

import numpy as np
import numpy.typing as npt
from numpy.polynomial import Polynomial

from .phenology import Stage


def petno3r(age: float) -> float:
    """ratio of :math:`NO_3` to total `N` in an individual petiole.

    Parameters
    ----------
    age : float
        physiological age of petiole.

    Returns
    -------
    float

    Examples
    --------
    >>> petno3r(1)
    0.945
    >>> petno3r(2)
    0.93
    >>> petno3r(62)
    0.03
    >>> petno3r(63)
    0.02
    """
    return max(Polynomial((0.96, -0.015))(age), 0.02)


class PlantNitrogen:  # pylint: disable=too-few-public-methods,no-member,attribute-defined-outside-init,too-many-instance-attributes
    """Module for nitrogen cycle computation.

    Attributes
    ----------
        pre_fruiting_nodes_age : list[double]
        burres : float
            reserve N in burrs, in g per plant.
        leafrs : float
            reserve N in leaves, in g per plant.
        petrs : float
            reserve N in petioles, in g per plant.
        stemrs : float
            reserve N in stems, in g per plant.
        rootrs : float
            reserve N in roots, in g per plant.
        rqnbur : float
            nitrogen requirement for burr growth.
        rqnlef : float
            nitrogen requirement for leaf growth.
        rqnpet : float
            nitrogen requirement for petiole growth.
        rqnrut : float
            nitrogen requirement for root growth.
        rqnsed : float
            nitrogen requirement for seed growth.
        rqnsqr : float
            nitrogen requirement for square growth.
        rqnstm : float
            nitrogen requirement for stem growth.
        reqv : float
            nitrogen requirement for vegetative shoot growth.
        reqf : float
            nitrogen requirement for fruit growth.
        reqtot : float
            total nitrogen requirement for plant growth.
        npool : float
            total nitrogen available for growth.
        uptn : float
            nitrogen uptake from the soil, g per plant.
        xtran : float
            amount of nitrogen not used for growth of plant parts.
        addnf : float
            daily added nitrogen to fruit, g per plant.
        addnr : float
            daily added nitrogen to root, g per plant.
        addnv : float
            daily added nitrogen to vegetative shoot, g per plant.
    """

    leafrs = 0.0
    burres = 0.0
    petrs = 0.0
    stemrs = 0.0
    rootrs = 0.0
    npool = 0.0
    uptn = 0.0
    xtran = 0.0
    addnf = 0.0
    addnr = 0.0
    addnv = 0.0
    node_leaf_age: npt.NDArray[np.double]

    @cached_property
    def actual_square_growth(self):
        """total actual growth of squares, g plant-1 day-1."""
        return (self.square_potential_growth * self.fruit_growth_ratio)[
            self.fruiting_nodes_stage == Stage.Square
        ].sum()

    @cached_property
    def petiole_nitrate_nitrogen(self) -> float:
        """This function computes the ratio of :math:`NO_3` nitrogen to total N in the
        petioles.
        """
        # The ratio of NO3 to total N in each individual petiole is computed as a
        # linear function of leaf age. It is assumed that this ratio is maximum for
        # young leaves and is declining with leaf age.
        spetno3 = 0.0  # sum of petno3r.
        # Loop of prefruiting node leaves.
        for i in range(self.number_of_pre_fruiting_nodes):  # type: ignore[attr-defined]
            spetno3 += petno3r(self.pre_fruiting_nodes_age[i])  # type: ignore
        # Loop of all the other leaves, with the same computations.

        # number of petioles computed.
        numl = self.number_of_pre_fruiting_nodes  # type: ignore[attr-defined]
        for i, stage in np.ndenumerate(self.fruiting_nodes_stage):
            if stage != Stage.NotYetFormed:
                numl += 1
                spetno3 += petno3r(self.node_leaf_age[i])
        # The return value of the function is the average ratio of NO3 to total N for
        # all the petioles in the plant.
        return spetno3 / numl

    @property
    def green_bolls_weight(self):
        """total weight of seedcotton in green bolls, g plant-1."""
        return self.fruiting_nodes_boll_weight[
            np.isin(self.fruiting_nodes_stage, (Stage.GreenBoll, Stage.YoungGreenBoll))
        ].sum()

    @property
    def reqtot(self):
        return self.rqnrut + self.reqv + self.reqf

    @property
    def rqnrut(self):
        # Add ExtraCarbon to CarbonAllocatedForRootGrowth to compute the total supply of
        # carbohydrates for root growth.
        rootcn0 = 0.018  # maximum N content for roots
        return rootcn0 * (
            self.carbon_allocated_for_root_growth + self.extra_carbon
        )  # for root

    @property
    def reqv(self):
        return self.rqnlef + self.rqnpet + self.rqnstm

    @property
    def rqnlef(self):
        lefcn0 = 0.064  # maximum N content for leaves
        return lefcn0 * self.total_actual_leaf_growth  # for leaf blade

    @property
    def rqnpet(self):
        petcn0 = 0.032  # maximum N content for petioles
        return petcn0 * self.total_actual_petiole_growth  # for petiole

    @property
    def rqnstm(self):
        stmcn0 = 0.036  # maximum N content for stems
        return stmcn0 * self.actual_stem_growth  # for stem

    @property
    def reqf(self):
        return self.rqnsqr + self.rqnsed + self.rqnbur

    @property
    def rqnbur(self):
        burcn0 = 0.026  # maximum N content for burrs
        return self.actual_burr_growth * burcn0  # for burrs

    @property
    def rqnsqr(self):
        sqrcn0 = 0.024  # maximum N content for squares
        return self.actual_square_growth * sqrcn0  # for squares

    @property
    def rqnsed(self):
        seedcn0 = 0.036  # maximum N content for seeds
        seedcn1 = 0.045  # additional requirement of N for existing seed tissue.
        seedratio = 0.64  # ratio of seeds to seedcotton in green bolls.
        # components of seed N requirements.
        rqnsed1 = self.actual_boll_growth * seedratio * seedcn0  # for seed growth
        rqnsed2 = 0
        # The N required for replenishing the N content of existing seed tissue
        # (rqnsed2) is added to seed growth requirement.
        if self.green_bolls_weight > self.actual_boll_growth:
            # existing ratio of N to dry matter in the seeds.
            rseedn = self.seed_nitrogen / (
                (self.green_bolls_weight - self.actual_boll_growth) * seedratio
            )
            rqnsed2 = max(
                (self.green_bolls_weight - self.actual_boll_growth)
                * seedratio
                * (seedcn1 - rseedn),
                0,
            )

        return rqnsed1 + rqnsed2  # total requirement for seeds

    def plant_nitrogen(self, emerge_date, growing_stem_weight):
        """\
        This function simulates the nitrogen accumulation and distribution in cotton
        plants, and computes nitrogen stresses.

        The maximum and minimum N concentrations for the various organs are modified
        from those reported by: Jones et. al. (1974): Development of a nitrogen balance
        for cotton growth models: a first approximation. Crop Sci. 14:541-546.

        The following parameters are used in all plant N routines:
                  Growth requirement   Uptake requirement  Minimum content
        leaves      lefcn0    = .064    vnreqlef  = .042    vlfnmin   = .018
        petioles    petcn0    = .032    vnreqpet  = .036    vpetnmin  = .005
        stems       stmcn0    = .036    vnreqstm  = .012    vstmnmin  = .006
        roots       rootcn0   = .018    vnreqrt   = .010    vrtnmin   = .010
        burrs       burcn0    = .026    vnreqbur  = .012    vburnmin  = .006
        seeds       seedcn0   = .036    seedcn1   = .045
        squares     sqrcn0    = .024    vnreqsqr  = .024
        """
        # The following subroutines are now called:
        # The following constant parameters are used:
        petcn0 = 0.032  # maximum N content for petioles
        # On emergence, assign initial values to petiole N concentrations.
        if self.date <= emerge_date:
            self.petiole_nitrogen_concentration = petcn0
            self.petiole_nitrate_nitrogen_concentration = petcn0
        self.nitrogen_supply()  # computes the supply of N from uptake and reserves.
        self.nitrogen_allocation()  # computes the allocation of N in the plant.
        if self.xtran > 0:
            # computes the further allocation of N in the plant
            self.extra_nitrogen_allocation()
        # computes the concentrations of N in plant dry matter.
        self.plant_nitrogen_content()
        self.get_nitrogen_stress()  # computes nitrogen stress factors.
        self.nitrogen_uptake_requirement(
            growing_stem_weight
        )  # computes N requirements for uptake

    def nitrogen_supply(self):
        """This function computes the supply of N by uptake from the soil reserves."""
        # The following constant parameters are used:
        MobilizNFractionBurrs = 0.08  # fraction of N mobilizable for burrs
        MobilizNFractionLeaves = (
            0.09  # fraction of N mobilizable for leaves and petioles
        )
        MobilizNFractionStemRoot = 0.40  # fraction of N mobilizable for stems and roots
        vburnmin = 0.006  # minimum N contents of burrs
        vlfnmin = 0.018  # minimum N contents of leaves
        vpetnmin = 0.005  # minimum N contents of petioles, non-nitrate fraction.
        vpno3min = 0.002  # minimum N contents of petioles, nitrate fraction.
        vrtnmin = 0.010  # minimum N contents of roots
        vstmnmin = 0.006  # minimum N contents of stems
        # uptn is the total supply of nitrogen to the plant by uptake of nitrate and
        # ammonium.
        self.uptn = self.supplied_nitrate_nitrogen + self.supplied_ammonium_nitrogen
        # If total N requirement is less than the supply, define npool as the supply
        # and assign zero to the N reserves in all organs.
        if self.reqtot <= self.uptn:
            self.npool = self.uptn
            self.leafrs = 0
            self.petrs = 0
            self.stemrs = 0
            self.rootrs = 0
            self.burres = 0
            self.xtran = 0
        else:
            # If total N requirement exceeds the supply, compute the nitrogen  reserves
            # in the plant. The reserve N in an organ is defined as a fraction  of the
            # nitrogen content exceeding a minimum N content in it.
            # The N reserves in leaves, petioles, stems, roots and burrs of green bolls
            # are computed, and their N content updated.
            self.leafrs = max(
                (self.leaf_nitrogen - vlfnmin * self.leaf_weight)
                * MobilizNFractionLeaves,
                0,
            )
            self.leaf_nitrogen -= self.leafrs
            # The petiole N content is subdivided to nitrate and non-nitrate.
            # The nitrate ratio in the petiole N is computed by property
            # petiole_nitrate_nitrogen
            # Note that the nitrate fraction is more available for redistribution.
            # ratio of NO3 N to total N in petioles.
            rpetno3 = self.petiole_nitrate_nitrogen
            # components of reserve N in petioles, for non-NO3 and NO3 origin,
            # respectively.
            petrs1 = max(
                (self.petiole_nitrogen * (1 - rpetno3) - vpetnmin * self.petiole_weight)
                * MobilizNFractionLeaves,
                0,
            )
            petrs2 = max(
                (self.petiole_nitrogen * rpetno3 - vpno3min * self.petiole_weight)
                * MobilizNFractionLeaves,
                0,
            )
            self.petrs = petrs1 + petrs2
            self.petiole_nitrogen -= self.petrs
            # Stem N reserves.
            self.stemrs = max(
                (self.stem_nitrogen - vstmnmin * self.stem_weight)
                * MobilizNFractionStemRoot,
                0,
            )
            self.stem_nitrogen -= self.stemrs
            # Root N reserves
            self.rootrs = max(
                (self.root_nitrogen - vrtnmin * self.root_weight)
                * MobilizNFractionStemRoot,
                0,
            )
            self.root_nitrogen -= self.rootrs
            # Burr N reserves
            if self.green_bolls_burr_weight > 0:
                self.burres = max(
                    (self.burr_nitrogen - vburnmin * self.green_bolls_burr_weight)
                    * MobilizNFractionBurrs,
                    0,
                )
                self.burr_nitrogen -= self.burres
            else:
                self.burres = 0
            # The total reserves, resn, are added to the amount taken up from the soil,
            # for computing npool. Note that N of seeds or squares is not available for
            # redistribution in the plant.
            resn = (
                self.leafrs + self.petrs + self.stemrs + self.rootrs + self.burres
            )  # total reserve N, in g per plant.
            self.npool = self.uptn + resn

    def plant_nitrogen_content(self):
        """This function computes the concentrations of nitrogen in the dry matter of
        the plant parts."""
        # The following constant parameter is used:
        seedratio = 0.64
        if self.petiole_weight > 0.00001:
            self.petiole_nitrogen_concentration = (
                self.petiole_nitrogen / self.petiole_weight
            )
            self.petiole_nitrate_nitrogen_concentration = (
                self.petiole_nitrogen_concentration * self.petiole_nitrate_nitrogen
            )
        if self.stem_weight > 0:
            self.stem_nitrogen_concentration = self.stem_nitrogen / self.stem_weight
        if self.root_weight > 0:
            self.root_nitrogen_concentration = self.root_nitrogen / self.root_weight
        if self.square_weight > 0:
            self.square_nitrogen_concentration = (
                self.square_nitrogen / self.square_weight
            )
        # weight of seeds in green and mature bolls.
        xxseed = (
            self.open_bolls_weight * (1 - self.ginning_percent)
            + self.green_bolls_weight * seedratio
        )
        if xxseed > 0:
            self.seed_nitrogen_concentration = self.seed_nitrogen / xxseed
        # weight of burrs in green and mature bolls.
        xxbur = self.open_bolls_burr_weight + self.green_bolls_burr_weight
        if xxbur > 0:
            self.burr_nitrogen_concentration = self.burr_nitrogen / xxbur

    def nitrogen_allocation(self):  # pylint: disable=too-many-statements
        """This function computes the allocation of supplied nitrogen to the plant
        parts."""
        # The following constant parameters are used:

        # maximum proportion of N pool that can be added to seeds
        vseednmax: float = 0.70
        # maximum proportion of N pool that can be added to squares
        vsqrnmax: float = 0.65
        # maximum proportion of N pool that can be added to burrs
        vburnmax: float = 0.65
        # maximum proportion of N pool that can be added to leaves
        vlfnmax: float = 0.90
        # maximum proportion of N pool that can be added to stems
        vstmnmax: float = 0.70
        # maximum proportion of N pool that can be added to petioles
        vpetnmax: float = 0.75
        # If total N requirement is less than npool, add N required for growth to the N
        # in each organ, compute added N to vegetative parts, fruiting parts and roots,
        # and compute xtran as the difference between npool and the total N requirements
        if self.reqtot <= self.npool:
            self.leaf_nitrogen += self.rqnlef
            self.petiole_nitrogen += self.rqnpet
            self.stem_nitrogen += self.rqnstm
            self.root_nitrogen += self.rqnrut
            self.square_nitrogen += self.rqnsqr
            self.seed_nitrogen += self.rqnsed
            self.burr_nitrogen += self.rqnbur
            self.addnv = self.rqnlef + self.rqnstm + self.rqnpet
            self.addnf = self.rqnsqr + self.rqnsed + self.rqnbur
            self.addnr = self.rqnrut
            self.xtran = self.npool - self.reqtot
            return
        # If N requirement is greater than npool, execute the following:
        # First priority is nitrogen supply to the growing seeds. It is assumed that up
        # to vseednmax = 0.70 of the supplied N can be used by the seeds. Update seed N
        # and addnf by the amount of nitrogen used for seed growth, and decrease npool
        # by this amount.
        # The same procedure is used for each organ, consecutively.
        usage = min(vseednmax * self.npool, self.rqnsed)
        self.seed_nitrogen += usage
        self.addnf += usage
        self.npool -= usage
        # Next priority is for burrs, which can use N up to vburnmax = 0.65 of the
        # remaining N pool, and for squares, which can use N up to vsqrnmax = 0.65
        usage = min(vburnmax * self.npool, self.rqnbur)
        self.burr_nitrogen += usage
        self.addnf += usage
        self.npool -= usage
        usage = min(vsqrnmax * self.npool, self.rqnsqr)
        self.square_nitrogen += usage
        self.addnf += usage
        self.npool -= usage
        # Next priority is for leaves, which can use N up to vlfnmax = 0.90 of the
        # remaining N pool, for stems, up to vstmnmax = 0.70, and for petioles, up to
        # vpetnmax = 0.75
        usage = min(vlfnmax * self.npool, self.rqnlef)
        self.leaf_nitrogen += usage
        self.addnv += usage
        self.npool -= usage

        usage = min(vstmnmax * self.npool, self.rqnstm)
        self.stem_nitrogen += usage
        self.addnv += usage
        self.npool -= usage

        usage = min(vpetnmax * self.npool, self.rqnpet)
        self.petiole_nitrogen += usage
        self.addnv += usage
        self.npool -= usage
        # The remaining npool goes to root growth. If any npool remains it is defined
        # as xtran.
        usage = min(self.npool, self.rqnrut)
        self.root_nitrogen += usage
        self.addnr += usage
        self.npool -= usage
        self.xtran = self.npool

    def extra_nitrogen_allocation(self):
        """Computes the allocation of extra nitrogen to the plant parts."""
        # If there are any N reserves in the plant, allocate remaining xtran in
        # proportion to the N reserves in each of these organs. Note: all reserves are
        # in g per plant units.
        addbur: float = 0  # reserve N to be added to the burrs.
        addlfn: float = 0  # reserve N to be added to the leaves.
        addpetn: float = 0  # reserve N to be added to the petioles.
        addrt: float = 0  # reserve N to be added to the roots.
        addstm: float = 0  # reserve N to be added to the stem.
        # sum of existing reserve N in plant parts.
        rsum = self.leafrs + self.petrs + self.stemrs + self.rootrs + self.burres
        if rsum > 0:
            addlfn = self.xtran * self.leafrs / rsum
            addpetn = self.xtran * self.petrs / rsum
            addstm = self.xtran * self.stemrs / rsum
            addrt = self.xtran * self.rootrs / rsum
            addbur = self.xtran * self.burres / rsum
        else:
            # If there are no reserves, allocate xtran in proportion to the dry weights
            # in each of these organs.
            # weight of vegetative plant parts, plus burrs.
            vegwt = (
                self.leaf_weight
                + self.petiole_weight
                + self.stem_weight
                + self.root_weight
                + self.green_bolls_burr_weight
            )
            addlfn = self.xtran * self.leaf_weight / vegwt
            addpetn = self.xtran * self.petiole_weight / vegwt
            addstm = self.xtran * self.stem_weight / vegwt
            addrt = self.xtran * self.root_weight / vegwt
            addbur = self.xtran * self.green_bolls_burr_weight / vegwt
        # Update N content in these plant parts. Note that at this stage of nitrogen
        # allocation, only vegetative parts and burrs are updated (not seeds or squares)
        self.leaf_nitrogen += addlfn
        self.petiole_nitrogen += addpetn
        self.stem_nitrogen += addstm
        self.root_nitrogen += addrt
        self.burr_nitrogen += addbur

    def get_nitrogen_stress(self):
        """This function computes the nitrogen stress factors."""
        # Set the default values for the nitrogen stress coefficients to 1.
        self.nitrogen_stress_vegetative = 1
        self.nitrogen_stress_root = 1
        self.nitrogen_stress_fruiting = 1
        self.nitrogen_stress = 1
        # Compute the nitrogen stress coefficients.
        # nitrogen_stress_fruiting is the ratio of N added actually to the fruits, to
        # their N requirements. state.nitrogen_stress_vegetative is the same for
        # vegetative shoot growth, and state.nitrogen_stress_root for roots. Also, an
        # average stress coefficient for vegetative and reproductive organs is computed
        # as nitrogen_stress.
        # Each stress coefficient has a value between 0 and 1.
        if self.reqf > 0:
            self.nitrogen_stress_fruiting = min(max(self.addnf / self.reqf, 0), 1)
        if self.reqv > 0:
            self.nitrogen_stress_vegetative = min(max(self.addnv / self.reqv, 0), 1)
        if self.rqnrut > 0:
            self.nitrogen_stress_root = min(max(self.addnr / self.rqnrut, 0), 1)
        if self.reqf + self.reqv > 0:
            self.nitrogen_stress = (self.addnf + self.addnv) / (self.reqf + self.reqv)
            self.nitrogen_stress = min(max(self.nitrogen_stress, 0), 1)

    def nitrogen_uptake_requirement(self, growing_stem_weight):
        """Computes total_required_nitrogen, the nitrogen requirements of the plant
        to be used for simulating the N uptake from the soil (in method
        nitrogen_uptake()) in the next day."""
        # The following constant parameters are used:
        seedcn1 = 0.045  # further requirement for existing seed tissue.
        seedratio = 0.64  # the ratio of seeds to seedcotton in green bolls.
        vnreqlef = 0.042  # coefficient for computing N uptake requirements of leaves
        vnreqpet = 0.036  # coefficient for computing N uptake requirements of petioles
        vnreqstm = 0.012  # coefficient for computing N uptake requirements of stems
        vnreqrt = 0.010  # coefficient for computing N uptake requirements of roots
        vnreqsqr = 0.024  # coefficient for computing N uptake requirements of squares
        vnreqbur = 0.012  # coefficient for computing N uptake requirements of burrs

        self.total_required_nitrogen = self.reqtot
        # After the requirements of today's growth are supplied, N is also required for
        # supplying necessary functions in other active plant tissues.
        # Add nitrogen uptake required for leaf and petiole tissue to TotalRequiredN.
        if self.leaf_nitrogen_concentration < vnreqlef:
            self.total_required_nitrogen += self.leaf_weight * (
                vnreqlef - self.leaf_nitrogen_concentration
            )
        if self.petiole_nitrogen_concentration < vnreqpet:
            self.total_required_nitrogen += self.petiole_weight * (
                vnreqpet - self.petiole_nitrogen_concentration
            )
        # Add stem requirement to TotalRequiredN.
        if self.stem_nitrogen_concentration < vnreqstm:
            self.total_required_nitrogen += growing_stem_weight * (
                vnreqstm - self.stem_nitrogen_concentration
            )
        # Compute nitrogen uptake requirement for existing tissues of roots, squares,
        # and seeds and burrs of green bolls. Add it to TotalRequiredN.
        if self.root_nitrogen_concentration < vnreqrt:
            self.total_required_nitrogen += self.root_weight * (
                vnreqrt - self.root_nitrogen_concentration
            )
        if self.square_nitrogen_concentration < vnreqsqr:
            self.total_required_nitrogen += self.square_weight * (
                vnreqsqr - self.square_nitrogen_concentration
            )
        if self.seed_nitrogen_concentration < seedcn1:
            self.total_required_nitrogen += (
                self.green_bolls_weight
                * seedratio
                * (seedcn1 - self.seed_nitrogen_concentration)
            )
        if self.burr_nitrogen_concentration < vnreqbur:
            self.total_required_nitrogen += self.green_bolls_burr_weight * (
                vnreqbur - self.burr_nitrogen_concentration
            )
