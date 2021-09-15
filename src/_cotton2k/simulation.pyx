from datetime import date, timedelta
from math import sin, cos, acos, sqrt, pi, atan

from libc.math cimport exp, log
from libc.stdlib cimport malloc
from libc.stdint cimport uint32_t
from libcpp cimport bool as bool_t

cimport numpy
import numpy as np

from _cotton2k.climate import compute_day_length, compute_incoming_long_wave_radiation, radiation, delta, gamma, refalbed, clcor, cloudcov, sunangle, clearskyemiss, dayrh, VaporPressure, tdewest, compute_hourly_wind_speed
from _cotton2k.fruit import TemperatureOnFruitGrowthRate
from _cotton2k.leaf import temperature_on_leaf_growth_rate, leaf_resistance_for_transpiration
from _cotton2k.soil import compute_soil_surface_albedo, compute_incoming_short_wave_radiation, root_psi, SoilTemOnRootGrowth, SoilAirOnRootGrowth, SoilNitrateOnRootGrowth, PsiOnTranspiration
from _cotton2k.utils import date2doy, doy2date
from _cotton2k.thermology import canopy_balance
from .climate cimport ClimateStruct
from .cxx cimport (
    cSimulation,
    SandVolumeFraction,
    ClayVolumeFraction,
    ElCondSatSoilToday,
    LocationColumnDrip,
    LocationLayerDrip,
    PotGroAllSquares,
    PotGroAllBolls,
    PotGroAllBurrs,
    PoreSpace,
    SoilPsi,
    SoilHorizonNum,
    AverageSoilPsi,
    VolNh4NContent,
    VolUreaNContent,
    noitr,
    thts,
    thetar,
    HeatCondDrySoil,
    HeatCapacitySoilSolid,
    MarginalWaterContent,
    HumusOrganicMatter,
    NO3FlowFraction,
    MaxWaterCapacity,
)
from .irrigation cimport Irrigation
from .rs cimport (
    SlabLoc,
    dl,
    wk,
    SoilMechanicResistance,
    wcond,
    PsiOsmotic,
    psiq,
    qpsi,
    form,
)
from .soil cimport cRoot
from .state cimport cState, cVegetativeBranch, cFruitingBranch, cMainStemLeaf, StateBase
from .fruiting_site cimport FruitingSite, Leaf, cBoll, cBurr, SquareStruct, cPetiole


cdef int LateralRootFlag[40] # flags indicating presence of lateral roots in soil layers: 0 = no lateral roots are possible. 1 = lateral roots may be initiated. 2 = lateral roots have been initiated.

AbscissionLag = np.zeros(20)  # the time (in physiological days) from tagging fruiting sites for shedding.
ShedByWaterStress = np.zeros(20)  # the effect of moisture stress on shedding.
ShedByNitrogenStress = np.zeros(20)  # the effect of nitrogen stress on shedding.
ShedByCarbonStress = np.zeros(20)  # the effect of carbohydrate stress on shedding
NumSheddingTags = 0  # number of 'box-car' units used for moving values in arrays defining fruit shedding (AbscissionLag, ShedByCarbonStress, ShedByNitrogenStress and ShedByWaterStress).

SOIL = np.array([], dtype=[
    ("depth", np.double),  # depth from soil surface to the end of horizon layers, cm.
])
cdef double condfc[9]  # hydraulic conductivity at field capacity of horizon layers, cm per day.
cdef double h2oint[14]  # initial soil water content, percent of field capacity,
# defined by input for consecutive 15 cm soil layers.
cdef double oma[14]  # organic matter at the beginning of the season, percent of soil weight,
# defined by input for consecutive 15 cm soil layers.
cdef double pclay[9]  # percentage of clay in soil horizon of horizon layers.
cdef double psand[9]  # percentage of sand in soil horizon of horizon layers.
cdef double psidra  # soil matric water potential, bars, for which immediate drainage
# will be simulated (suggested value -0.25 to -0.1).
cdef double psisfc  # soil matric water potential at field capacity,
# bars (suggested value -0.33 to -0.1).
cdef double rnnh4[14]  # residual nitrogen as ammonium in soil at beginning of season, kg per ha.
# defined by input for consecutive 15 cm soil layers.
cdef double rnno3[14]  # residual nitrogen as nitrate in soil at beginning of season, kg per ha.
# defined by input for consecutive 15 cm soil layers.
cdef double LayerDepth = 15
cdef double AverageLwp = 0  # running average of state.min_leaf_water_potential + state.max_leaf_water_potential for the last 3 days.
PercentDefoliation = 0

LwpMinX = np.zeros(3, dtype=np.double)  # array of values of min_leaf_water_potential for the last 3 days.
LwpX = np.zeros(3, dtype=np.double)  # array of values of min_leaf_water_potential + max_leaf_water_potential for the last 3 days.
FoliageTemp = np.ones(20, dtype=np.double) * 295  # average foliage temperature (oK).

DefoliationDate = np.zeros(5, dtype=np.int_)  # Dates (DOY) of defoliant applications.
DefoliationMethod = np.zeros(5, dtype=np.int_)  # code number of method of application of defoliants:  0 = 'banded'; 1 = 'sprinkler'; 2 = 'broaddcast'.
DefoliantAppRate = np.zeros(5, dtype=np.double)  # rate of defoliant application in pints per acre.


PetioleWeightPreFru = np.zeros(9, dtype=np.double)  # weight of prefruiting node petioles, g.
PotGroLeafAreaPreFru = np.zeros(9, dtype=np.double)  # potentially added area of a prefruiting node leaf, dm2 day-1.
PotGroLeafWeightPreFru = np.zeros(9, dtype=np.double)  # potentially added weight of a prefruiting node leaf, g day-1.
PotGroPetioleWeightPreFru = np.zeros(9, dtype=np.double)  # potentially added weight of a prefruiting node petiole, g day-1.
RootImpede = np.zeros((40, 20), dtype=np.double)  # root mechanical impedance for a soil cell, kg cm-2.


cdef void InitializeSoilTemperature():
    """Initializes the variables needed for the simulation of soil temperature, and variables used by functions ThermalCondSoil() and SoilHeatFlux().

    It is executed once at the beginning of the simulation.
    """
    cdef double bsand = 20    # heat conductivity of sand and silt (mcal cm-1 s-1 C-1).
    cdef double bclay = 7     # heat conductivity of clay (mcal cm-1 s-1 C-1).
    cdef double cka = 0.0615  # heat conductivity of air (mcal cm-1 s-1 C-1).
    cdef double ckw = 1.45    # heat conductivity of water (mcal cm-1 s-1 C-1).
    cdef double cmin = 0.46   # heat capacity of the mineral fraction of the soil.
    cdef double corg = 0.6    # heat capacity of the organic fraction of the soil.
    cdef double ga = 0.144    # shape factor for air in pore spaces.
    cdef double rm = 2.65     # specific weight of mineral fraction of soil.
    cdef double ro = 1.3      # specific weight of organic fraction of soil.
    # Compute aggregation factors:
    dsand = form(bsand, ckw, ga)  # aggregation factor for sand in water
    dclay = form(bclay, ckw, ga)  # aggregation factor for clay in water
    cdef double dsandair = form(bsand, cka, ga)  # aggregation factor for sand in air
    cdef double dclayair = form(bclay, cka, ga)  # aggregation factor for clay in air
    # Loop over all soil layers, and define indices for some soil arrays.
    cdef double sumdl = 0  # sum of depth of consecutive soil layers.
    for l in range(40):
        sumdl += dl(l)
        j = int((sumdl + LayerDepth - 1) / LayerDepth) - 1  # layer definition for oma
        if j > 13:
            j = 13
        # Using the values of the clay and organic matter percentages in the soil, compute mineral and organic fractions of the soil, by weight and by volume.
        mmo = oma[j] / 100  # organic matter fraction of dry soil (by weight).
        mm = 1 - mmo  # mineral fraction of dry soil (by weight).
        # MarginalWaterContent is set as a function of the sand fraction of the soil.
        i1 = SoilHorizonNum[l]  # layer definition as in soil hydrology input file.
        MarginalWaterContent[l] = 0.1 - 0.07 * psand[i1] / 100
        # The volume fractions of clay (ClayVolumeFraction) and of sand plus silt (SandVolumeFraction), are calculated.
        ra = (mmo / ro) / (mm / rm)  # volume ratio of organic to mineral soil fractions.
        xo = (1 - PoreSpace[l]) * ra / (1 + ra)  # organic fraction of soil (by volume).
        xm = (1 - PoreSpace[l]) - xo  # mineral fraction of soil (by volume).
        ClayVolumeFraction[l] = pclay[i1] * xm / mm / 100
        SandVolumeFraction[l] = 1 - PoreSpace[l] - ClayVolumeFraction[l]
        # Heat capacity of the solid soil fractions (mineral + organic, by volume )
        HeatCapacitySoilSolid[l] = xm * cmin + xo * corg
        # The heat conductivity of dry soil (HeatCondDrySoil) is computed using the procedure suggested by De Vries.
        HeatCondDrySoil[l] = (
            1.25
            * (
                PoreSpace[l] * cka
                + dsandair * bsand * SandVolumeFraction[l]
                + dclayair * bclay * ClayVolumeFraction[l]
            )
            / (
                PoreSpace[l]
                + dsandair * SandVolumeFraction[l]
                + dclayair * ClayVolumeFraction[l]
            )
        )


cdef class SoilInit:
    cdef unsigned int number_of_layers
    def __init__(self, initial, hydrology):
        for i, layer in enumerate(initial):
            rnnh4[i] = layer["ammonium_nitrogen"]
            rnno3[i] = layer["nitrate_nitrogen"]
            oma[i] = layer["organic_matter"]
            h2oint[i] = layer["water"]
        self.hydrology = hydrology
        self.number_of_layers = len(hydrology["layers"])

    @property
    def lyrsol(self):
        return self.number_of_layers

    @property
    def hydrology(self):
        return {
            "ratio_implicit": RatioImplicit,
            "max_conductivity": conmax,
            "field_capacity_water_potential": psisfc,
            "immediate_drainage_water_potential": psidra,
            "layers": [
                {
                    "depth": SOIL["depth"][i],
                    "air_dry": airdr[i],
                    "theta": thetas[i],
                    "alpha": alpha[i],
                    "beta": vanGenuchtenBeta,
                    "saturated_hydraulic_conductivity": SaturatedHydCond[i],
                    "field_capacity_hydraulic_conductivity": condfc[i],
                    "bulk_density": BulkDensity[i],
                    "clay": pclay[i],
                    "sand": psand[i],
                }
                for i in range(self.number_of_layers)
            ]
        }

    @hydrology.setter
    def hydrology(self, soil_hydrology):
        global RatioImplicit, conmax, psisfc, psidra, SOIL
        RatioImplicit = soil_hydrology["ratio_implicit"]
        conmax = soil_hydrology["max_conductivity"]
        psisfc = soil_hydrology["field_capacity_water_potential"]
        psidra = soil_hydrology["immediate_drainage_water_potential"]
        for i, layer in enumerate(soil_hydrology["layers"]):
            SOIL = np.append(SOIL, np.array([layer["depth"]], dtype=[("depth", np.double)]))
            airdr[i] = layer["air_dry"]
            thetas[i] = layer["theta"]
            alpha[i] = layer["alpha"]
            vanGenuchtenBeta[i] = layer["beta"]
            SaturatedHydCond[i] = layer["saturated_hydraulic_conductivity"]
            condfc[i] = layer["field_capacity_hydraulic_conductivity"]
            BulkDensity[i] = layer["bulk_density"]
            pclay[i] = layer["clay"]
            psand[i] = layer["sand"]

cdef class Climate:
    cdef ClimateStruct *climate
    cdef unsigned int start_day
    cdef unsigned int days
    cdef unsigned int current

    def __init__(self, start_date, climate):
        self.start_day = date2doy(start_date)
        self.current = self.start_day
        self.days = len(climate)
        self.climate = <ClimateStruct *> malloc(sizeof(ClimateStruct) * len(climate))
        for i, daily_climate in enumerate(climate):
            self.climate[i].Rad = daily_climate["radiation"]
            self.climate[i].Tmax = daily_climate["max"]
            self.climate[i].Tmin = daily_climate["min"]
            self.climate[i].Wind = daily_climate["wind"]
            self.climate[i].Rain = daily_climate["rain"]
            self.climate[i].Tdew = daily_climate.get("dewpoint",
                                                     tdewest(daily_climate["max"],
                                                             SitePar[5], SitePar[6]))

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.days
            step = key.step or 1
            if isinstance(start, date):
                start = date2doy(start) - self.start_day
            if isinstance(stop, date):
                stop = date2doy(stop) - self.start_day
            return [{
                "radiation": self.climate[i].Rad,
                "max": self.climate[i].Tmax,
                "min": self.climate[i].Tmin,
                "wind": self.climate[i].Wind,
                "rain": self.climate[i].Rain,
                "dewpoint": self.climate[i].Tdew,
            } for i in range(start, stop, step)]
        else:
            if not isinstance(key, int):
                key = date2doy(key) - self.start_day
            climate = self.climate[key]
            return {
                "radiation": climate["Rad"],
                "max": climate["Tmax"],
                "min": climate["Tmin"],
                "wind": climate["Wind"],
                "rain": climate["Rain"],
                "dewpoint": climate["Tdew"],
            }


cdef class Root:
    cdef cRoot *_
    cdef unsigned int l
    cdef unsigned int k

    @property
    def weight_capable_uptake(self):
        return self._[0].weight_capable_uptake

    @weight_capable_uptake.setter
    def weight_capable_uptake(self, value):
        self._[0].weight_capable_uptake = value

    @property
    def growth_factor(self):
        return self._[0].growth_factor

    @staticmethod
    cdef Root from_ptr(cRoot *_ptr, unsigned int l, unsigned int k):
        cdef Root root = Root.__new__(Root)
        root._ = _ptr
        root.l = l
        root.k = k
        return root


cdef class SoilCell:
    cdef cSoilCell *_
    cdef unsigned int l
    cdef unsigned int k
    cdef public Root root

    @staticmethod
    cdef SoilCell from_ptr(cSoilCell *_ptr, unsigned int l, unsigned int k):
        cdef SoilCell cell = SoilCell.__new__(SoilCell)
        cell._ = _ptr
        cell.l = l
        cell.k = k
        cell.root = Root.from_ptr(&_ptr[0].root, l, k)
        return cell

    @property
    def water_content(self):
        return self._[0].water_content

    @water_content.setter
    def water_content(self, value):
        self._[0].water_content = value

    @property
    def nitrate_nitrogen_content(self):
        return self._[0].nitrate_nitrogen_content

    @nitrate_nitrogen_content.setter
    def nitrate_nitrogen_content(self, value):
        self._[0].nitrate_nitrogen_content = value

    @property
    def fresh_organic_matter(self):
        return self._[0].fresh_organic_matter

    @fresh_organic_matter.setter
    def fresh_organic_matter(self, value):
        self._[0].fresh_organic_matter = value


cdef class NodeLeaf:
    cdef Leaf *_

    @property
    def age(self):
        return self._[0].age

    @age.setter
    def age(self, value):
        self._[0].age = value

    @property
    def area(self):
        return self._[0].area

    @area.setter
    def area(self, value):
        self._[0].area = value

    @property
    def potential_growth(self):
        return self._[0].potential_growth

    @potential_growth.setter
    def potential_growth(self, value):
        self._[0].potential_growth = value

    @property
    def weight(self):
        return self._[0].weight

    @weight.setter
    def weight(self, value):
        self._[0].weight = value

    @staticmethod
    cdef NodeLeaf from_ptr(Leaf *_ptr):
        cdef NodeLeaf leaf = NodeLeaf.__new__(NodeLeaf)
        leaf._ = _ptr
        return leaf


cdef class Petiole:
    cdef cPetiole *_

    @property
    def potential_growth(self):
        return self._[0].potential_growth

    @potential_growth.setter
    def potential_growth(self, value):
        self._[0].potential_growth = value

    @property
    def weight(self):
        return self._[0].weight

    @weight.setter
    def weight(self, value):
        self._[0].weight = value

    @staticmethod
    cdef Petiole from_ptr(cPetiole *_ptr):
        cdef Petiole petiole = Petiole.__new__(Petiole)
        petiole._ = _ptr
        return petiole


cdef class Boll:
    cdef cBoll *_
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int m

    @property
    def age(self):
        return self._[0].age

    @age.setter
    def age(self, value):
        self._[0].age = value

    @property
    def weight(self):
        return self._[0].weight

    @weight.setter
    def weight(self, value):
        self._[0].weight = value

    @property
    def cumulative_temperature(self):
        return self._[0].cumulative_temperature

    @cumulative_temperature.setter
    def cumulative_temperature(self, value):
        self._[0].cumulative_temperature = value

    @property
    def potential_growth(self):
        return self._[0].potential_growth

    @staticmethod
    cdef Boll from_ptr(cBoll *_ptr, unsigned int k, unsigned int l, unsigned int m):
        cdef Boll boll = Boll.__new__(Boll)
        boll._ = _ptr
        boll.k = k
        boll.l = l
        boll.m = m
        return boll


cdef class Burr:
    cdef cBurr *_
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int m

    @property
    def potential_growth(self):
        return self._[0].potential_growth

    @property
    def weight(self):
        return self._[0].weight

    @weight.setter
    def weight(self, value):
        self._[0].weight = value

    @staticmethod
    cdef Burr from_ptr(cBurr *_ptr, unsigned int k, unsigned int l, unsigned int m):
        cdef Burr burr = Burr.__new__(Burr)
        burr._ = _ptr
        burr.k = k
        burr.l = l
        burr.m = m
        return burr


cdef class Square:
    cdef SquareStruct *_
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int m

    @property
    def weight(self):
        return self._[0].weight

    @weight.setter
    def weight(self, value):
        self._[0].weight = value

    @property
    def potential_growth(self):
        return self._[0].potential_growth

    @staticmethod
    cdef Square from_ptr(SquareStruct *_ptr, unsigned int k, unsigned int l, unsigned int m):
        cdef Square square = Square.__new__(Square)
        square._ = _ptr
        square.k = k
        square.l = l
        square.m = m
        return square


cdef class FruitingNode:
    cdef FruitingSite *_
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int m
    cdef public Petiole petiole

    @property
    def average_temperature(self):
        return self._[0].average_temperature

    @average_temperature.setter
    def average_temperature(self, value):
        self._[0].average_temperature = value

    @property
    def age(self):
        return self._[0].age

    @age.setter
    def age(self, value):
        self._[0].age = value

    @property
    def fraction(self):
        return self._[0].fraction

    @fraction.setter
    def fraction(self, value):
        self._[0].fraction = value

    @property
    def ginning_percent(self):
        return self._[0].ginning_percent

    @ginning_percent.setter
    def ginning_percent(self, value):
        self._[0].ginning_percent = value

    @property
    def leaf(self):
        return NodeLeaf.from_ptr(&self._.leaf)

    @property
    def boll(self):
        return Boll.from_ptr(&self._[0].boll, self.k, self.l, self.m)

    @property
    def burr(self):
        return Burr.from_ptr(&self._[0].burr, self.k, self.l, self.m)

    @property
    def square(self):
        return Square.from_ptr(&self._[0].square, self.k, self.l, self.m)

    @property
    def stage(self):
        return self._[0].stage

    @stage.setter
    def stage(self, value):
        self._[0].stage = value

    @staticmethod
    cdef FruitingNode from_ptr(FruitingSite *_ptr, unsigned int k, unsigned int l, unsigned int m):
        cdef FruitingNode node = FruitingNode.__new__(FruitingNode)
        node._ = _ptr
        node.k = k
        node.l = l
        node.m = m
        node.petiole = Petiole.from_ptr(&_ptr[0].petiole)
        return node


cdef class PreFruitingNode:
    cdef double *_age

    @property
    def age(self):
        return self._age[0]

    @age.setter
    def age(self, value):
        self._age[0] = value

    @staticmethod
    cdef PreFruitingNode from_ptr(double *age):
        cdef PreFruitingNode node = PreFruitingNode.__new__(PreFruitingNode)
        node._age = age
        return node


cdef class MainStemLeaf:
    cdef cMainStemLeaf *_

    @property
    def area(self):
        return self._.leaf_area

    @area.setter
    def area(self, value):
        self._.leaf_area = value

    @property
    def potential_growth_of_area(self):
        return self._[0].potential_growth_for_leaf_area

    @potential_growth_of_area.setter
    def potential_growth_of_area(self, value):
        self._[0].potential_growth_for_leaf_area = value

    @property
    def potential_growth_of_weight(self):
        return self._[0].potential_growth_for_leaf_area

    @potential_growth_of_weight.setter
    def potential_growth_of_weight(self, value):
        self._[0].potential_growth_for_leaf_weight = value

    @property
    def potential_growth_of_petiole(self):
        return self._[0].potential_growth_for_petiole_weight

    @potential_growth_of_petiole.setter
    def potential_growth_of_petiole(self, value):
        self._[0].potential_growth_for_petiole_weight = value

    @property
    def weight(self):
        return self._.leaf_weight

    @weight.setter
    def weight(self, value):
        self._.leaf_weight = value

    @property
    def petiole_weight(self):
        return self._[0].petiole_weight

    @petiole_weight.setter
    def petiole_weight(self, value):
        self._[0].petiole_weight = value

    @staticmethod
    cdef MainStemLeaf from_ptr(cMainStemLeaf *_ptr):
        """Factory function to create WrapperClass objects from
        given my_c_struct pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Call to __new__ bypasses __init__ constructor
        cdef MainStemLeaf main_stem_leaf = MainStemLeaf.__new__(MainStemLeaf)
        main_stem_leaf._ = _ptr
        return main_stem_leaf


cdef class FruitingBranch:
    cdef cFruitingBranch *_
    cdef unsigned int k
    cdef unsigned int l

    @property
    def number_of_fruiting_nodes(self):
        return self._[0].number_of_fruiting_nodes

    @number_of_fruiting_nodes.setter
    def number_of_fruiting_nodes(self, value):
        self._[0].number_of_fruiting_nodes = value

    @property
    def delay_for_new_node(self):
        return self._[0].delay_for_new_node

    @delay_for_new_node.setter
    def delay_for_new_node(self, value):
        self._[0].delay_for_new_node = value

    @property
    def main_stem_leaf(self):
        return MainStemLeaf.from_ptr(&self._[0].main_stem_leaf)

    @property
    def nodes(self):
        return [FruitingNode.from_ptr(&self._.nodes[i], self.k, self.l, i) for i in
                range(self._.number_of_fruiting_nodes)]

    @staticmethod
    cdef FruitingBranch from_ptr(cFruitingBranch *_ptr, unsigned int k, unsigned int l):
        """Factory function to create WrapperClass objects from
        given my_c_struct pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Call to __new__ bypasses __init__ constructor
        cdef FruitingBranch fruiting_branch = FruitingBranch.__new__(FruitingBranch)
        fruiting_branch._ = _ptr
        fruiting_branch.k = k
        fruiting_branch.l = l
        return fruiting_branch


cdef class VegetativeBranch:
    cdef cVegetativeBranch *_
    cdef unsigned int k

    @property
    def number_of_fruiting_branches(self):
        return self._[0].number_of_fruiting_branches

    @number_of_fruiting_branches.setter
    def number_of_fruiting_branches(self, value):
        self._[0].number_of_fruiting_branches = value

    @property
    def delay_for_new_fruiting_branch(self):
        return self._[0].delay_for_new_fruiting_branch

    @delay_for_new_fruiting_branch.setter
    def delay_for_new_fruiting_branch(self, value):
        self._[0].delay_for_new_fruiting_branch = value

    @property
    def fruiting_branches(self):
        return [FruitingBranch.from_ptr(&self._[0].fruiting_branches[i], self.k, i) for i in
                range(self._[0].number_of_fruiting_branches)]

    @staticmethod
    cdef VegetativeBranch from_ptr(cVegetativeBranch *_ptr, unsigned int k):
        """Factory function to create WrapperClass objects from
        given my_c_struct pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Call to __new__ bypasses __init__ constructor
        cdef VegetativeBranch vegetative_branch = VegetativeBranch.__new__(VegetativeBranch)
        vegetative_branch._ = _ptr
        vegetative_branch.k = k
        return vegetative_branch


cdef class Hour:
    cdef public double albedo  # hourly albedo of a reference crop.
    cdef public double cloud_cor  # hourly cloud type correction.
    cdef public double cloud_cov  # cloud cover ratio (0 to 1).
    cdef public double dew_point  # hourly dew point temperatures, C.
    cdef public double et1  # part of hourly Penman evapotranspiration affected by net radiation, in mm per hour.
    cdef public double et2  # part of hourly Penman evapotranspiration affected by wind and vapor pressure deficit, in mm per hour.
    cdef public double humidity  # hourly values of relative humidity (%).
    cdef public double radiation  # hourly global radiation, W / m2.
    cdef public double ref_et  # reference evapotranspiration, mm per hour.
    cdef public double temperature  # hourly air temperatures, C.
    cdef public double wind_speed  # Hourly wind velocity, m per second.

cdef double[3] cgind = [1, 1, 0.10]  # the index for the capability of growth of class I roots (0 to 1).

cdef double gh2oc[10]  # input gravimetric soil water content, g g-1, in the soil mechanical impedance table. values have been read from the soil impedance file.
cdef double tstbd[10][10]  # input bulk density in the impedance table, g cm-3.
cdef double impede[10][10]  # input table of soil impedance to root growth
cdef int inrim  # number of input bulk-density data points for the impedance curve
cdef unsigned int ncurve  # number of input soil-moisture curves in the impedance table.


cdef class SoilImpedance:
    @property
    def curves(self):
        global gh2oc, tstbd, impede, inrim, ncurve
        return {gh2oc[i]: {tstbd[j][i]: impede[j][i] for j in range(inrim)} for i in
                range(ncurve)}

    @curves.setter
    def curves(self, impedance_table):
        global gh2oc, tstbd, impede, inrim, ncurve
        ncurve = len(impedance_table)
        inrim = len(impedance_table[0])
        for i, row in enumerate(impedance_table):
            gh2oc[i] = row.pop("water")
            for j, pair in enumerate(sorted(row.items())):
                tstbd[j][i], impede[j][i] = pair


cdef class Soil:
    cdef cSoil *_
    cells = np.empty((40, 20), dtype=object)

    @property
    def number_of_layers_with_root(self):
        return self._[0].number_of_layers_with_root

    @number_of_layers_with_root.setter
    def number_of_layers_with_root(self, value):
        self._[0].number_of_layers_with_root = value

    @staticmethod
    cdef Soil from_ptr(cSoil *_ptr):
        cdef Soil soil = Soil.__new__(Soil)
        soil._ = _ptr
        for l in range(40):
            for k in range(20):
                soil.cells[l][k] = SoilCell.from_ptr(&_ptr[0].cells[l][k], l, k)
        return soil

    def root_impedance(self):
        """This function calculates soil mechanical impedance to root growth, rtimpd(l,k), for all soil cells. It is called from PotentialRootGrowth(). The impedance is a function of bulk density and water content in each soil soil cell. No changes have been made in the original GOSSYM code."""
        global gh2oc, tstbd, impede, inrim, ncurve
        for l in range(nl):
            j = SoilHorizonNum[l]
            Bd = BulkDensity[j]  # bulk density for this layer

            for jj in range(inrim):
                if Bd <= tstbd[jj][0]:
                    break
            j1 = min(jj, inrim - 1)
            j0 = max(0, jj - 1)

            for k in range(nk):
                Vh2o = self._[0].cells[l][k].water_content / Bd
                for ik in range(ncurve):
                    if Vh2o <= gh2oc[ik]:
                        break
                i1 = min(ncurve - 1, ik)
                i0 = max(0, ik - 1)

                if j1 == 0:
                    if i1 == 0 or Vh2o <= gh2oc[i1]:
                        RootImpede[l][k] = impede[j1][i1]
                    else:
                        RootImpede[l][k] = impede[j1][i0] - (impede[j1][i0] - impede[j1][i1]) * (Vh2o - gh2oc[i0]) / (gh2oc[i1] - gh2oc[i0])
                else:
                    if i1 == 0 or Vh2o <= gh2oc[i1]:
                        RootImpede[l][k] = impede[j0][i1] - (impede[j0][i1] - impede[j1][i1]) * (tstbd[j0][i1] - Bd) / (tstbd[j0][i1] - tstbd[j1][i1])
                    else:
                        temp1 = impede[j0][i1] - (impede[j0][i1] - impede[j1][i1]) * (tstbd[j0][i1] - Bd) / (tstbd[j0][i1] - tstbd[j1][i1])
                        temp2 = impede[j0][i0] - (impede[j0][i0] - impede[j1][i1]) * (tstbd[j0][i0] - Bd) / (tstbd[j0][i0] - tstbd[j1][i0])
                        RootImpede[l][k] = temp2 + (temp1 - temp2) * (Vh2o - gh2oc[i0]) / (gh2oc[i1] - gh2oc[i0])


cdef class State(StateBase):
    cdef Simulation _sim
    cdef public Soil soil
    cdef public numpy.ndarray root_weights
    pre_fruiting_nodes = []

    @staticmethod
    cdef State from_ptr(cState *_ptr, Simulation _sim, unsigned int version):
        """Factory function to create WrapperClass objects from
        given my_c_struct pointer.

        Setting ``owner`` flag to ``True`` causes
        the extension type to ``free`` the structure pointed to by ``_ptr``
        when the wrapper object is deallocated."""
        # Call to __new__ bypasses __init__ constructor
        cdef State state = State.__new__(State)
        state._ = _ptr
        state._sim = _sim
        state.version = version
        state.soil = Soil.from_ptr(&_ptr[0].soil)
        for i in range(24):
            state.hours[i] = Hour()
        for i in range(_ptr[0].number_of_pre_fruiting_nodes):
            state.pre_fruiting_nodes.append(PreFruitingNode.from_ptr(&_ptr[0].age_of_pre_fruiting_nodes[i]))
        return state

    @property
    def phenological_delay_for_vegetative_by_carbon_stress(self):
        """delay in formation of new fruiting branches caused by carbon stress."""
        delay = np.polynomial.Polynomial([self._sim.cultivar_parameters[27], -0.25, -0.75])(self.carbon_stress)
        return min(max(delay, 0), 1)

    @property
    def phenological_delay_for_fruiting_by_carbon_stress(self):
        """delay in formation of new fruiting sites caused by carbon stress."""
        delay = np.polynomial.Polynomial([self._sim.cultivar_parameters[28], -0.83, -1.67])(self.carbon_stress)
        return min(max(delay, 0), self._sim.cultivar_parameters[29])

    @property
    def vegetative_branches(self):
        return [VegetativeBranch.from_ptr(&self._[0].vegetative_branches[k], k) for k in range(self.number_of_vegetative_branches)]

    def roots_capable_of_uptake(self):
        """This function computes the weight of roots capable of uptake for all soil cells."""
        cuind = [1, 0.5, 0]  # the indices for the relative capability of uptake (between 0 and 1) of water and nutrients by root age classes.
        for l in range(40):
            for k in range(20):
                self.soil.cells[l][k].root.weight_capable_uptake = 0
        # Loop for all soil soil cells with roots. compute for each soil cell root-weight capable of uptake (RootWtCapblUptake) as the sum of products of root weight and capability of uptake index (cuind) for each root class in it.
        for l in range(self.soil.number_of_layers_with_root):
            for k in range(self.soil._[0].layers[l].number_of_left_columns_with_root, self.soil._[0].layers[l].number_of_right_columns_with_root + 1):
                for i in range(3):
                    if self.root_weights[l][k][i] > 1.e-15:
                        self.soil.cells[l][k].root.weight_capable_uptake += self.root_weights[l][k][i] * cuind[i]

    def predict_emergence(self, plant_date, hour, plant_row_column):
        """This function predicts date of emergence."""
        cdef double dpl = 5  # depth of planting, cm (assumed 5).
        # Define some initial values on day of planting.
        if self.date == plant_date and hour == 0:
            self.delay_of_emergence = 0
            self.hypocotyl_length = 0.3
            self.seed_moisture = 8
            # Compute soil layer number for seed depth.
            sumdl = 0  # depth to the bottom of a soil layer.
            for l in range(40):
                sumdl += dl(l)
                if sumdl >= dpl:
                    self.seed_layer_number = l
                    break
        # Compute matric soil moisture potential at seed location.
        # Define te as soil temperature at seed location, C.
        cdef double psi  # matric soil moisture potential at seed location.
        cdef double te  # soil temperature at seed depth, C.
        psi = SoilPsi[self.seed_layer_number][plant_row_column]
        te = SoilTemp[self.seed_layer_number][plant_row_column] - 273.161
        te = max(te, 10)

        # Phase 1 of of germination - imbibition. This phase is executed when the moisture content of germinating seeds is not more than 80%.
        cdef double dw  # rate of moisture addition to germinating seeds, percent per hour.
        cdef double xkl  # a function of temperature and moisture, used to calculate dw.
        if self.seed_moisture <= 80:
            xkl = .0338 + .0000855 * te * te - 0.003479 * psi
            if xkl < 0:
                xkl = 0
            # Compute the rate of moisture addition to germinating seeds, percent per hour.
            dw = xkl * (80 - self.seed_moisture)
            # Compute delw, the marginal value of dw, as a function of soil temperature and soil water potential.
            delw = 0  # marginal value of dw.
            if te < 21.2:
                delw = -0.1133 + .000705 * te ** 2 - .001348 * psi + .001177 * psi ** 2
            elif te < 26.66:
                delw = -.3584 + .001383 * te ** 2 - .03509 * psi + .003507 * psi ** 2
            elif te < 32.3:
                delw = -.6955 + .001962 * te ** 2 - .08335 * psi + .007627 * psi ** 2 - .006411 * psi * te
            else:
                delw = 3.3929 - .00197 * te ** 2 - .36935 * psi + .00865 * psi ** 2 + .007306 * psi * te
            if delw < 0.01:
                delw = 0.01
            # Add dw to tw, or if dw is less than delw assign 100% to tw.
            if dw > delw:
                self.seed_moisture += dw
            else:
                self.seed_moisture = 100
            return

        # Phase 2 of of germination - hypocotyl elongation.
        cdef double xt  # a function of temperature, used to calculate de.
        if te > 39.9:
            xt = 0
        else:
            xt = 0.0853 - 0.0057 * (te - 34.44) * (te - 34.44) / (41.9 - te)
        # At low soil temperatures, when negative values of xt occur, compute the delay in germination rate.
        if xt < 0 and te < 14:
            self.delay_of_emergence += xt / 2
            return
        else:
            if self.delay_of_emergence < 0:
                if self.delay_of_emergence + xt < 0:
                    self.delay_of_emergence += xt
                    return
                else:
                    xt += self.delay_of_emergence
                    self.delay_of_emergence = 0
        # Compute elongation rate of hypocotyl, de, as a sigmoid function of HypocotylLength. Add de to HypocotylLength.
        cdef double de  # rate of hypocotyl growth, cm per hour.
        de = 0.0567 * xt * self.hypocotyl_length * (10 - self.hypocotyl_length)
        self.hypocotyl_length += de
        # Check for completion of emergence (when HypocotylLength exceeds planting depth) and report germination to output.
        if self.hypocotyl_length > dpl:
            self.kday = 1
            return self.date

    def pre_fruiting_node(self, stemNRatio, time_to_next_pre_fruiting_node, time_factor_for_first_two_pre_fruiting_nodes, time_factor_for_third_pre_fruiting_node, initial_pre_fruiting_nodes_leaf_area):
        """This function checks if a new prefruiting node is to be added, and then sets it."""
        # The following constant parameter is used:
        cdef double MaxAgePreFrNode = 66  # maximum age of a prefruiting node (constant)
        # When the age of the last prefruiting node exceeds MaxAgePreFrNode, this function is not activated.
        if self.pre_fruiting_nodes[-1].age > MaxAgePreFrNode:
            return
        # Loop over all existing prefruiting nodes.
        # Increment the age of each prefruiting node in physiological days.
        for node in self.pre_fruiting_nodes:
            node.age += self.day_inc
        # For the last prefruiting node (if there are less than 9 prefruiting nodes):
        # The period (timeToNextPreFruNode) until the formation of the next node is VarPar(31), but it is modified for the first three nodes.
        # If the physiological age of the last prefruiting node is more than timeToNextPreFruNode, form a new prefruiting node - increase state.number_of_pre_fruiting_nodes, assign the initial average temperature for the new node, and initiate a new leaf on this node.
        if self.number_of_pre_fruiting_nodes >= 9:
            return
        # time, in physiological days, for the next prefruiting node to be formed.
        if self.number_of_pre_fruiting_nodes <= 2:
            time_to_next_pre_fruiting_node *= time_factor_for_first_two_pre_fruiting_nodes
        elif self.number_of_pre_fruiting_nodes == 3:
            time_to_next_pre_fruiting_node *= time_factor_for_third_pre_fruiting_node

        if self.pre_fruiting_nodes[-1].age >= time_to_next_pre_fruiting_node:
            if self.version >= 0x500:
                leaf_weight = min(initial_pre_fruiting_nodes_leaf_area * self.leaf_weight_area_ratio, self.stem_weight - 0.2)
                if leaf_weight <= 0:
                    return
                leaf_area = leaf_weight / self.leaf_weight_area_ratio
            else:
                leaf_area = initial_pre_fruiting_nodes_leaf_area
                leaf_weight = leaf_area * self.leaf_weight_area_ratio
            self.number_of_pre_fruiting_nodes += 1
            self._[0].leaf_area_pre_fruiting[self.number_of_pre_fruiting_nodes - 1] = leaf_area
            self._[0].leaf_weight_pre_fruiting[self.number_of_pre_fruiting_nodes - 1] = leaf_weight
            self.leaf_weight += leaf_weight
            self.stem_weight -= leaf_weight
            self.leaf_nitrogen += leaf_weight * stemNRatio
            self.stem_nitrogen -= leaf_weight * stemNRatio

    def add_fruiting_node(self, int k, int l, double stemNRatio, double density_factor, double var34, double var36, double var37):
        """Decide if a new node is to be added to a fruiting branch, and forms it. It is called from function CottonPhenology()."""
        # The following constant parameters are used:
        cdef double[6] vfrtnod = [1.32, 0.90, 33.0, 7.6725, -0.3297, 0.004657]
        # Compute the cumulative delay for the appearance of the next node on the fruiting branch, caused by carbohydrate, nitrogen, and water stresses.
        self.vegetative_branches[k].fruiting_branches[l].delay_for_new_node += self.phenological_delay_for_fruiting_by_carbon_stress + vfrtnod[0] * self.phenological_delay_by_nitrogen_stress
        self.vegetative_branches[k].fruiting_branches[l].delay_for_new_node += vfrtnod[1] * (1 - self.water_stress)
        # Define nnid, and compute the average temperature of the last node of this fruiting branch, from the time it was formed.
        cdef int nnid = self._[0].vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes - 1  # the number of the last node on this fruiting branche.
        cdef double tav = min(self._[0].vegetative_branches[k].fruiting_branches[l].nodes[nnid].average_temperature, vfrtnod[2])  # modified daily average temperature.
        # Compute TimeToNextFruNode, the time (in physiological days) needed for the formation of each successive node on the fruiting branch. This is a function of temperature, derived from data of K. R. Reddy, CSRU, adjusted for age in physiological days. It is modified for plant density.
        cdef double TimeToNextFruNode  # time, in physiological days, for the next node on the fruiting branch to be formed
        TimeToNextFruNode = var36 + tav * (vfrtnod[3] + tav * (vfrtnod[4] + tav * vfrtnod[5]))
        TimeToNextFruNode = TimeToNextFruNode * (1 + var37 * (1 - density_factor)) + self.vegetative_branches[k].fruiting_branches[l].delay_for_new_node
        # Check if the the age of the last node on the fruiting branch exceeds TimeToNextFruNode.
        # If so, form the new node:
        if self._[0].vegetative_branches[k].fruiting_branches[l].nodes[nnid].age < TimeToNextFruNode or self._[0].vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes == 5:
            return
        # Increment NumNodes, define newnod, and assign 1 to FruitFraction and FruitingCode.
        if self.version >= 0x500:
            leaf_weight = min(var34 * self.leaf_weight_area_ratio, self.stem_weight - 0.2)
            if leaf_weight <= 0:
                return
            leaf_area = leaf_weight / self.leaf_weight_area_ratio
        else:
            leaf_area = var34
            leaf_weight = leaf_area * self.leaf_weight_area_ratio
        self._[0].vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes += 1
        cdef int newnod = nnid + 1  # the number of the new node on this fruiting branche.
        self._[0].vegetative_branches[k].fruiting_branches[l].nodes[newnod].fraction = 1
        self._[0].vegetative_branches[k].fruiting_branches[l].nodes[newnod].stage = Stage.Square
        # Initiate a new leaf at the new node. The mass and nitrogen in the new leaf is substacted from the stem.
        self._[0].vegetative_branches[k].fruiting_branches[l].nodes[newnod].leaf.area = leaf_area
        self._[0].vegetative_branches[k].fruiting_branches[l].nodes[newnod].leaf.weight = leaf_weight
        self.stem_weight -= leaf_weight
        self.leaf_weight += leaf_weight
        self.leaf_nitrogen += leaf_weight * stemNRatio
        self.stem_nitrogen -= leaf_weight * stemNRatio
        # Begin computing AvrgNodeTemper of the new node, and assign zero to DelayNewNode.
        self._[0].vegetative_branches[k].fruiting_branches[l].nodes[newnod].average_temperature = self.average_temperature
        self.vegetative_branches[k].fruiting_branches[l].delay_for_new_node = 0

    def leaf_water_potential(self, double row_space):
        """This function simulates the leaf water potential of cotton plants.

        It has been adapted from the model of Moshe Meron (The relation of cotton leaf water potential to soil water content in the irrigated management range. PhD dissertation, UC Davis, 1984).
        """
        # Constant parameters used:
        cdef double cmg = 3200  # length in cm per g dry weight of roots, based on an average
        # root diameter of 0.06 cm, and a specific weight of 0.11 g dw per cubic cm.
        cdef double psild0 = -1.32  # maximum values of min_leaf_water_potential
        cdef double psiln0 = -0.40  # maximum values of self.max_leaf_water_potential.
        cdef double rtdiam = 0.06  # average root diameter in cm.
        cdef double[13] vpsil = [0.48, -5.0, 27000., 4000., 9200., 920., 0.000012, -0.15, -1.70, -3.5, 0.1e-9, 0.025, 0.80]
        # Leaf water potential is not computed during 10 days after emergence. Constant values are assumed for this period.
        if self.kday <= 10:
            self.max_leaf_water_potential = psiln0
            self.min_leaf_water_potential = psild0
            return
        # Compute shoot resistance (rshoot) as a function of plant height.
        cdef double rshoot  # shoot resistance, Mpa hours per cm.
        rshoot = vpsil[0] * self.plant_height / 100
        # Assign zero to summation variables
        cdef double psinum = 0  # sum of RootWtCapblUptake for all soil cells with roots.
        cdef double rootvol = 0  # sum of volume of all soil cells with roots.
        cdef double rrlsum = 0  # weighted sum of reciprocals of rrl.
        cdef double rroot = 0  # root resistance, Mpa hours per cm.
        cdef double sumlv = 0  # weighted sum of root length, cm, for all soil cells with roots.
        cdef double vh2sum = 0  # weighted sum of soil water content, for all soil cells with roots.
        # Loop over all soil cells with roots. Check if RootWtCapblUptake is greater than vpsil[10].
        # All average values computed for the root zone, are weighted by RootWtCapblUptake (root weight capable of uptake), but the weight assigned will not be greater than vpsil[11].
        cdef double rrl  # root resistance per g of active roots.
        for l in range(self.soil.number_of_layers_with_root):
            for k in range(self._[0].soil.layers[l].number_of_left_columns_with_root, self._[0].soil.layers[l].number_of_right_columns_with_root):
                if self.soil.cells[l][k].root.weight_capable_uptake >= vpsil[10]:
                    psinum += min(self.soil.cells[l][k].root.weight_capable_uptake, vpsil[11])
                    sumlv += min(self.soil.cells[l][k].root.weight_capable_uptake, vpsil[11]) * cmg
                    rootvol += dl(l) * wk(k, row_space)
                    if SoilPsi[l][k] <= vpsil[1]:
                        rrl = vpsil[2] / cmg
                    else:
                        rrl = (vpsil[3] - SoilPsi[l][k] * (vpsil[4] + vpsil[5] * SoilPsi[l][k])) / cmg
                    rrlsum += min(self.soil.cells[l][k].root.weight_capable_uptake, vpsil[11]) / rrl
                    vh2sum += self.soil.cells[l][k].water_content * min(self.soil.cells[l][k].root.weight_capable_uptake, vpsil[11])
        # Compute average root resistance (rroot) and average soil water content (vh2).
        cdef double dumyrs  # intermediate variable for computing cond.
        cdef double vh2  # average of soil water content, for all soil soil cells with roots.
        if psinum > 0 and sumlv > 0:
            rroot = psinum / rrlsum
            vh2 = vh2sum / psinum
            dumyrs = max(sqrt(1 / (pi * sumlv / rootvol)) / rtdiam, 1.001)
        else:
            rroot = 0
            vh2 = thad[0]
            dumyrs = 1.001
        # Compute hydraulic conductivity (cond), and soil resistance near the root surface  (rsoil).
        cdef double cond  # soil hydraulic conductivity near the root surface.
        cond = wcond(vh2, thad[0], thts[0], vanGenuchtenBeta[0], SaturatedHydCond[0], PoreSpace[0]) / 24
        cond = cond * 2 * sumlv / rootvol / log(dumyrs)
        cond = max(cond, vpsil[6])
        cdef double rsoil = 0.0001 / (2 * pi * cond)  # soil resistance, Mpa hours per cm.
        # Compute leaf resistance (leaf_resistance_for_transpiration) as the average of the resistances of all existing leaves.
        # The resistance of an individual leaf is a function of its age.
        # Function leaf_resistance_for_transpiration is called to compute it. This is executed for all the leaves of the plant.
        cdef int numl = 0  # number of leaves.
        cdef double sumrl = 0  # sum of leaf resistances for all the plant.
        for j in range(self.number_of_pre_fruiting_nodes):  # loop prefruiting nodes
            numl += 1
            sumrl += leaf_resistance_for_transpiration(self.pre_fruiting_nodes[j].age)

        for k in range(self.number_of_vegetative_branches):  # loop for all other nodes
            for l in range(self.vegetative_branches[k].number_of_fruiting_branches):
                for m in range(self.vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes):
                    numl += 1
                    sumrl += leaf_resistance_for_transpiration(self.vegetative_branches[k].fruiting_branches[l].nodes[m].leaf.age)
        cdef double rleaf = sumrl / numl  # leaf resistance, Mpa hours per cm.

        cdef double rtotal = rsoil + rroot + rshoot + rleaf  # The total resistance to transpiration, MPa hours per cm, (rtotal) is computed.
        # Compute maximum (early morning) leaf water potential, max_leaf_water_potential, from soil water potential (AverageSoilPsi, converted from bars to MPa).
        # Check for minimum and maximum values.
        self.max_leaf_water_potential = min(max(vpsil[7] + 0.1 * AverageSoilPsi, vpsil[8]), psiln0)
        # Compute minimum (at time of maximum transpiration rate) leaf water potential, min_leaf_water_potential, from maximum transpiration rate (etmax) and total resistance to transpiration (rtotal).
        cdef double etmax = 0  # the maximum hourly rate of evapotranspiration for this day.
        for ihr in range(24):  # hourly loop
            if self.hours[ihr].ref_et > etmax:
                etmax = self.hours[ihr].ref_et
        self.min_leaf_water_potential = min(max(self.max_leaf_water_potential - 0.1 * max(etmax, vpsil[12]) * rtotal, vpsil[9]), psild0)

    def actual_leaf_growth(self, vratio):
        """This function simulates the actual growth of leaves of cotton plants. It is called from PlantGrowth()."""
        # Loop for all prefruiting node leaves. Added dry weight to each leaf is proportional to PotGroLeafWeightPreFru. Update leaf weight (state.leaf_weight_pre_fruiting) and leaf area (state.leaf_area_pre_fruiting) for each prefruiting node leaf. added dry weight to each petiole is proportional to PotGroPetioleWeightPreFru. update petiole weight (PetioleWeightPreFru) for each prefruiting node leaf.
        # Compute total leaf weight (state.leaf_weight), total petiole weight (PetioleWeightNodes), and state.leaf_area.
        for j in range(self.number_of_pre_fruiting_nodes): # loop by prefruiting node.
            self._[0].leaf_weight_pre_fruiting[j] += PotGroLeafWeightPreFru[j] * vratio
            self.leaf_weight += self._[0].leaf_weight_pre_fruiting[j]
            PetioleWeightPreFru[j] += PotGroPetioleWeightPreFru[j] * vratio
            self.petiole_weight += PetioleWeightPreFru[j]
            self._[0].leaf_area_pre_fruiting[j] += PotGroLeafAreaPreFru[j] * vratio
            self.leaf_area += self._[0].leaf_area_pre_fruiting[j]
        # Loop for all fruiting branches on each vegetative branch, to compute actual growth of mainstem leaves.
        # Added dry weight to each leaf is proportional to PotGroLeafWeightMainStem, added dry weight to each petiole is proportional to PotGroPetioleWeightMainStem, and added area to each leaf is proportional to PotGroLeafAreaMainStem.
        # Update leaf weight (LeafWeightMainStem), petiole weight (PetioleWeightMainStem) and leaf area(LeafAreaMainStem) for each main stem node leaf.
        # Update the total leaf weight (state.leaf_weight), total petiole weight (state.petiole_weight) and total area (state.leaf_area).
        for k in range(self.number_of_vegetative_branches):  # loop of vegetative branches
            for l in range(self.vegetative_branches[k].number_of_fruiting_branches):  # loop of fruiting branches
                main_stem_leaf = self.vegetative_branches[k].fruiting_branches[l].main_stem_leaf
                main_stem_leaf.weight += main_stem_leaf.potential_growth_of_weight * vratio
                self.leaf_weight += main_stem_leaf.weight
                main_stem_leaf.petiole_weight += main_stem_leaf.potential_growth_of_petiole * vratio
                self.petiole_weight += main_stem_leaf.petiole_weight
                main_stem_leaf.area += main_stem_leaf.potential_growth_of_area * vratio
                self.leaf_area += main_stem_leaf.area
                # Loop for all fruiting nodes on each fruiting branch. to compute actual growth of fruiting node leaves.
                # Added dry weight to each leaf is proportional to PotGroLeafWeightNodes, added dry weight to each petiole is proportional to PotGroPetioleWeightNodes, and added area to each leaf is proportional to PotGroLeafAreaNodes.
                # Update leaf weight (LeafWeightNodes), petiole weight (PetioleWeightNodes) and leaf area (LeafAreaNodes) for each fruiting node leaf.
                # Compute total leaf weight (state.leaf_weight), total petiole weight (PetioleWeightNodes) and total area (state.leaf_area).
                for m in range(self._[0].vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes):  # loop of nodes on a fruiting branch
                    site = self.vegetative_branches[k].fruiting_branches[l].nodes[m]
                    site.leaf.weight += site.leaf.potential_growth * self.leaf_weight_area_ratio * vratio
                    self.leaf_weight += site.leaf.weight
                    site.petiole.weight += site.petiole.potential_growth * vratio
                    self.petiole_weight += site.petiole.weight
                    site.leaf.area += site.leaf.potential_growth * vratio
                    self.leaf_area += site.leaf.area

    def actual_fruit_growth(self):
        """This function simulates the actual growth of squares and bolls of cotton plants."""
        # Assign zero to all the sums to be computed.
        self.square_weight = 0
        self.green_bolls_weight = 0
        self.green_bolls_burr_weight = 0
        self.actual_square_growth = 0
        self.actual_boll_growth = 0
        self.actual_burr_growth = 0
        # Begin loops over all fruiting sites.
        for vegetative_branch in self.vegetative_branches:
            for fruiting_branch in vegetative_branch.fruiting_branches:
                for site in fruiting_branch.nodes:
                    # If this site is a square, the actual dry weight added to it (dwsq) is proportional to its potential growth.
                    # Update the weight of this square (SquareWeight), sum of today's added dry weight to squares (state.actual_square_growth), and total weight of squares (self.square_weight).
                    if site.stage == Stage.Square:
                        dwsq = site.square.potential_growth * self.fruit_growth_ratio  # dry weight added to square.

                        site.square.weight += dwsq
                        self.actual_square_growth += dwsq
                        self.square_weight += site.square.weight
                    # If this site is a green boll, the actual dry weight added to seedcotton and burrs is proportional to their respective potential growth.
                    if site.stage == Stage.GreenBoll or site.stage == Stage.YoungGreenBoll:
                        # dry weight added to seedcotton in a boll.
                        dwboll = site.boll.potential_growth * self.fruit_growth_ratio
                        site.boll.weight += dwboll
                        self.actual_boll_growth += dwboll
                        self.green_bolls_weight += site.boll.weight
                        # dry weight added to the burrs in a boll.
                        dwburr = site.burr.potential_growth * self.fruit_growth_ratio
                        site.burr.weight += dwburr
                        self.actual_burr_growth += dwburr
                        self.green_bolls_burr_weight += site.burr.weight

    def dry_matter_balance(self, per_plant_area) -> float:
        """This function computes the cotton plant dry matter (carbon) balance, its allocation to growing plant parts, and carbon stress. It is called from PlantGrowth()."""
        # The following constant parameters are used:
        vchbal = [6.0, 2.5, 1.0, 5.0, 0.20, 0.80, 0.48, 0.40, 0.2072, 0.60651, 0.0065, 1.10, 4.0, 0.25, 4.0]
        # Assign values for carbohydrate requirements for growth of stems, roots, leaves, petioles, squares and bolls. Potential growth of all plant parts is modified by nitrogen stresses.
        # carbohydrate requirement for square growth, g per plant per day.
        cdsqar = PotGroAllSquares * (self.nitrogen_stress_fruiting + vchbal[0]) / (vchbal[0] + 1)
        # carbohydrate requirement for boll and burr growth, g per plant per day.
        cdboll = (PotGroAllBolls + PotGroAllBurrs) * (self.nitrogen_stress_fruiting + vchbal[0]) / (vchbal[0] + 1)
        # cdleaf is carbohydrate requirement for leaf growth, g per plant per day.
        cdleaf = self.leaf_potential_growth * (self.nitrogen_stress_vegetative + vchbal[1]) / (vchbal[1] + 1)
        # cdstem is carbohydrate requirement for stem growth, g per plant per day.
        cdstem = self.stem_potential_growth * (self.nitrogen_stress_vegetative + vchbal[2]) / (vchbal[2] + 1)
        # cdroot is carbohydrate requirement for root growth, g per plant per day.
        cdroot = self.root_potential_growth * (self.nitrogen_stress_root + vchbal[3]) / (vchbal[3] + 1)
        # cdpet is carbohydrate requirement for petiole growth, g per plant per day.
        cdpet = self.petiole_potential_growth * (self.nitrogen_stress_vegetative + vchbal[14]) / (vchbal[14] + 1)
        # total carbohydrate requirement for plant growth, g per plant per day.
        cdsum = cdstem + cdleaf + cdpet + cdroot + cdsqar + cdboll
        # Compute CarbonStress as the ratio of available to required carbohydrates.
        if cdsum <= 0:
            self.carbon_stress = 1
            return 0  # Exit function if cdsum is 0.
        # total available carbohydrates for growth (cpool, g per plant).
        # cpool is computed as: net photosynthesis plus a fraction (vchbal(13) ) of the stored reserves (reserve_carbohydrate).
        cpool = self.net_photosynthesis + self.reserve_carbohydrate * vchbal[13]
        self.carbon_stress = min(1, cpool / cdsum)
        # When carbohydrate supply is sufficient for growth requirements, CarbonStress will be assigned 1, and the carbohydrates actually supplied for plant growth (total_actual_leaf_growth, total_actual_petiole_growth, actual_stem_growth, carbon_allocated_for_root_growth, pdboll, pdsq) will be equal to the required amounts.
        # pdboll is amount of carbohydrates allocated to boll growth.
        # pdsq is amount of carbohydrates allocated to square growth.
        if self.carbon_stress == 1:
            self.total_actual_leaf_growth = cdleaf
            self.total_actual_petiole_growth = cdpet
            self.actual_stem_growth = cdstem
            self.carbon_allocated_for_root_growth = cdroot
            pdboll = cdboll
            pdsq = cdsqar
            xtrac1 = 0
        # When carbohydrate supply is less than the growth requirements, set priorities for allocation of carbohydrates.
        else:
            # cavail remaining available carbohydrates.
            # First priority is for fruit growth. Compute the ratio of available carbohydrates to the requirements for boll and square growth (bsratio).
            if cdboll + cdsqar > 0:
                # ratio of available carbohydrates to the requirements for boll and square growth.
                bsratio = cpool / (cdboll + cdsqar)
                # ffr is ratio of actual supply of carbohydrates to the requirement for boll and square growth.
                # The factor ffr is a function of bsratio and WaterStress. It is assumed that water stress increases allocation of carbohydrates to bolls. Check that ffr is not less than zero, or greater than 1 or than bsratio.
                ffr = min(max((vchbal[5] + vchbal[6] * (1 - self.water_stress)) * bsratio, 0), 1)
                ffr = min(bsratio, ffr)
                # Now compute the actual carbohydrates used for boll and square growth, and the remaining available carbohydrates.
                pdboll = cdboll * ffr
                pdsq = cdsqar * ffr
                cavail = cpool - pdboll - pdsq
            else:
                cavail = cpool
                pdboll = 0
                pdsq = 0
            # The next priority is for leaf and petiole growth. Compute the factor flf for leaf growth allocation, and check that it is not less than zero or greater than 1.
            if cdleaf + cdpet > 0:
                # ratio of actual supply of carbohydrates to the requirement for leaf growth.
                flf = min(max(vchbal[7] * cavail / (cdleaf + cdpet), 0), 1)
                # Compute the actual carbohydrates used for leaf and petiole growth, and the
                # remaining available carbohydrates.
                self.total_actual_leaf_growth = cdleaf * flf
                self.total_actual_petiole_growth = cdpet * flf
                cavail -= self.total_actual_leaf_growth + self.total_actual_petiole_growth
            else:
                self.total_actual_leaf_growth = 0
                self.total_actual_petiole_growth = 0
            # The next priority is for root growth.
            if cdroot > 0:
                # ratio between carbohydrate supply to root and to stem growth.
                # At no water stress conditions, ratio is an exponential function of dry weight of vegetative shoot (stem + leaves). This equation is based on data from Avi Ben-Porath's PhD thesis.
                # ratio is modified (calibrated) by vchbal[11].
                ratio = vchbal[8] + vchbal[9] * exp(-vchbal[10] * (self.stem_weight + self.leaf_weight + self.petiole_weight) *
                                                    per_plant_area)
                ratio *= vchbal[11]
                # rtmax is the proportion of remaining available carbohydrates that can be supplied to root growth. This is increased by water stress.
                rtmax = ratio / (ratio + 1)
                rtmax = rtmax * (1 + vchbal[12] * (1 - self.water_stress))
                rtmax = min(rtmax, 1)
                # Compute the factor frt for root growth allocation, as a function of rtmax, and check that it is not less than zero or greater than 1.
                # ratio of actual supply of carbohydrates to the requirement for root growth.
                frt = min(max(rtmax * cavail / cdroot, 0), 1)
                # Compute the actual carbohydrates used for root growth, and the remaining available carbohydrates.
                self.carbon_allocated_for_root_growth = max((cdroot * frt), (cavail - cdstem))
                cavail -= self.carbon_allocated_for_root_growth
            else:
                self.carbon_allocated_for_root_growth = 0
            # The remaining available carbohydrates are used for stem growth. Compute thefactor fst and the actual carbohydrates used for stem growth.
            if cdstem > 0:
                # ratio of actual supply of carbohydrates to the requirement for stem growth.
                fst = min(max(cavail / cdstem, 0), 1)
                self.actual_stem_growth = cdstem * fst
            else:
                self.actual_stem_growth = 0
            # If there are any remaining available unused carbohydrates, define them as xtrac1.
            xtrac1 = max(cavail - self.actual_stem_growth, 0)
        # Check that the amounts of carbohydrates supplied to each organ will not be less than zero.
        self.actual_stem_growth = max(self.actual_stem_growth, 0)
        self.total_actual_leaf_growth = max(self.total_actual_leaf_growth, 0)
        self.total_actual_petiole_growth = max(self.total_actual_petiole_growth, 0)
        self.carbon_allocated_for_root_growth = max(self.carbon_allocated_for_root_growth, 0)
        pdboll = max(pdboll, 0)
        pdsq = max(pdsq, 0)
        # Update the amount of reserve carbohydrates (reserve_carbohydrate) in the leaves.
        self.reserve_carbohydrate += self.net_photosynthesis - (self.actual_stem_growth + self.total_actual_leaf_growth + self.total_actual_petiole_growth + self.carbon_allocated_for_root_growth + pdboll + pdsq)
        # maximum possible amount of carbohydrate reserves that can be stored in the leaves.
        # resmax is a fraction (vchbal[4])) of leaf weight. Excessive reserves are defined as xtrac2.
        resmax = vchbal[4] * self.leaf_weight
        if self.reserve_carbohydrate > resmax:
            xtrac2 = self.reserve_carbohydrate - resmax
            self.reserve_carbohydrate = resmax
        else:
            xtrac2 = 0
        # ExtraCarbon is computed as total excessive carbohydrates.
        self.extra_carbon = xtrac1 + xtrac2
        # Compute state.fruit_growth_ratio as the ratio of carbohydrates supplied to square and boll growth to their carbohydrate requirements.
        if PotGroAllSquares + PotGroAllBolls + PotGroAllBurrs > 0:
            self.fruit_growth_ratio = (pdsq + pdboll) / (PotGroAllSquares + PotGroAllBolls + PotGroAllBurrs)
        else:
            self.fruit_growth_ratio = 1
        # Compute vratio as the ratio of carbohydrates supplied to leaf and petiole growth to their carbohydrate requirements.
        if self.leaf_potential_growth + self.petiole_potential_growth > 0:
            vratio = (self.total_actual_leaf_growth + self.total_actual_petiole_growth) / (self.leaf_potential_growth + self.petiole_potential_growth)
        else:
            vratio = 1
        return vratio

    def init_root_data(self, uint32_t plant_row_column, double mul):
        for l in range(40):
            for k in range(20):
                self._[0].soil.cells[l][k].root = {
                    "growth_factor": 1,
                    "weight_capable_uptake": 0,
                }
        # FIXME: I consider the value is incorrect
        self.root_weights = np.zeros((40, 20, 3), dtype=np.float64)
        self.root_weights[0,(plant_row_column - 1, plant_row_column + 2),0] = 0.0020
        self.root_weights[0,(plant_row_column, plant_row_column + 1),0] = 0.0070
        self.root_weights[1,(plant_row_column - 1, plant_row_column + 2),0] = 0.0040
        self.root_weights[1,(plant_row_column, plant_row_column + 1),0] = 0.0140
        self.root_weights[2,(plant_row_column - 1, plant_row_column + 2),0] = 0.0060
        self.root_weights[2,(plant_row_column, plant_row_column + 1),0] = 0.0210
        for l, w in zip(range(3, 7), (0.0200, 0.0150, 0.0100, 0.0050)):
            self.root_weights[l, (plant_row_column, plant_row_column + 1), 0] = w
        self.root_weights[:] *= mul
        self.root_age[:3,plant_row_column - 1:plant_row_column + 3] = 0.01
        self.root_age[3:7,plant_row_column:plant_row_column + 2] = 0.01

    def root_death(self, l, k):
        """This function computes the death of root tissue in each soil cell containing roots.

        When root age reaches a threshold thdth(i), a proportion dth(i) of the roots in class i dies. The mass of dead roots is added to DailyRootLoss.

        It has been adapted from GOSSYM, but the threshold age for this process is based on the time from when the roots first grew into each soil cell.

        It is assumed that root death rate is greater in dry soil, for all root classes except class 1. Root death rate is increased to the maximum value in soil saturated with water.
        """
        cdef double aa = 0.008  # a parameter in the equation for computing dthfac.
        cdef double[3] dth = [0.0001, 0.0002, 0.0001]  # the daily proportion of death of root tissue.
        cdef double dthmax = 0.10  # a parameter in the equation for computing dthfac.
        cdef double psi0 = -14.5  # a parameter in the equation for computing dthfac.
        cdef double[3] thdth = [30.0, 50.0, 100.0]  # the time threshold, from the initial
        # penetration of roots to a soil cell, after which death of root tissue of class i may occur.

        result = 0
        for i in range(3):
            if self.root_age[l][k] > thdth[i]:
                # the computed proportion of roots dying in each class.
                dthfac = dth[i]
                if self._[0].soil.cells[l][k].water_content >= PoreSpace[l]:
                    dthfac = dthmax
                else:
                    if i <= 1 and SoilPsi[l][k] <= psi0:
                        dthfac += aa * (psi0 - SoilPsi[l][k])
                    if dthfac > dthmax:
                        dthfac = dthmax
                result += self.root_weights[l][k][i] * dthfac
                self.root_weights[l][k][i] -= self.root_weights[l][k][i] * dthfac
        return result

    def tap_root_growth(self, int NumRootAgeGroups, unsigned int plant_row_column):
        """This function computes the elongation of the taproot."""
        # Call function TapRootGrowth() for taproot elongation, if the taproot has not already reached the bottom of the slab.
        if self.taproot_layer_number >= nl - 1 and self.taproot_length >= self.last_layer_with_root_depth:
            return
        # The following constant parameters are used:
        cdef double p1 = 0.10  # constant parameter.
        cdef double rtapr = 4  # potential growth rate of the taproot, cm/day.
        # It is assumed that taproot elongation takes place irrespective of the supply of carbon to the roots. This elongation occurs in the two columns of the slab where the plant is located.
        # Tap root elongation does not occur in water logged soil (water table).
        cdef int klocp1 = plant_row_column + 1  # the second column in which taproot growth occurs.
        if self._[0].soil.cells[self.taproot_layer_number][plant_row_column].water_content >= PoreSpace[self.taproot_layer_number] or self._[0].soil.cells[self.taproot_layer_number][klocp1].water_content >= PoreSpace[self.taproot_layer_number]:
            return
        # Average soil resistance (avres) is computed at the root tip.
        # avres = average value of RootGroFactor for the two soil cells at the tip of the taproot.
        cdef double avres = 0.5 * (self._[0].soil.cells[self.taproot_layer_number][plant_row_column].root.growth_factor + self._[0].soil.cells[self.taproot_layer_number][klocp1].root.growth_factor)
        # It is assumed that a linear empirical function of avres controls the rate of taproot elongation. The potential elongation rate of the taproot is also modified by soil temperature (SoilTemOnRootGrowth function), soil resistance, and soil moisture near the root tip.
        # Actual growth is added to the taproot_length.
        cdef double stday  # daily average soil temperature (C) at root tip.
        stday = 0.5 * (self.soil_temperature[self.taproot_layer_number][plant_row_column] + self.soil_temperature[self.taproot_layer_number][klocp1]) - 273.161
        cdef double addtaprt  # added taproot length, cm
        addtaprt = rtapr * (1 - p1 + avres * p1) * SoilTemOnRootGrowth(stday)
        self.taproot_length += addtaprt
        # last_layer_with_root_depth, the depth (in cm) to the end of the last layer with roots, is used to check if the taproot reaches a new soil layer.
        # When the new value of taproot_length is greater than last_layer_with_root_depth - it means that the roots penetrate to a new soil layer.
        # In this case, and if this is not the last layer in the slab, the following is executed:
        # self.taproot_layer_number and last_layer_with_root_depth are incremented. If this is a new layer with roots, state.soil.number_of_layers_with_root is also redefined and two soil cells of the new layer are defined as containing roots (by initializing RootColNumLeft and RootColNumRight).
        if self.taproot_layer_number > nl - 2 or self.taproot_length <= self.last_layer_with_root_depth:
            return
        # The following is executed when the taproot reaches a new soil layer.
        self.taproot_layer_number += 1
        self.last_layer_with_root_depth += dl(self.taproot_layer_number)
        if self.taproot_layer_number > self._[0].soil.number_of_layers_with_root - 1:
            self._[0].soil.number_of_layers_with_root = self.taproot_layer_number + 1
            if self._[0].soil.number_of_layers_with_root > nl:
                self._[0].soil.number_of_layers_with_root = nl
        if (self._[0].soil.layers[self.taproot_layer_number].number_of_left_columns_with_root == 0 or
            self._[0].soil.layers[self.taproot_layer_number].number_of_left_columns_with_root > plant_row_column):
            self._[0].soil.layers[self.taproot_layer_number].number_of_left_columns_with_root = plant_row_column
        if (self._[0].soil.layers[self.taproot_layer_number].number_of_right_columns_with_root == 0 or
            self._[0].soil.layers[self.taproot_layer_number].number_of_right_columns_with_root < klocp1):
            self._[0].soil.layers[self.taproot_layer_number].number_of_right_columns_with_root = klocp1
        # RootAge is initialized for these soil cells.
        self.root_age[self.taproot_layer_number][plant_row_column] = 0.01
        self.root_age[self.taproot_layer_number][klocp1] = 0.01
        # Some of the mass of class 1 roots is transferred downwards to the new cells.
        # The transferred mass is proportional to 2 cm of layer width, but it is not more than half the existing mass in the last layer.
        for i in range(NumRootAgeGroups):
            # root mass transferred to the cell below when the elongating taproot
            # reaches a new soil layer.
            # first column
            tran = self.root_weights[self.taproot_layer_number - 1][plant_row_column][i] * 2 / dl(self.taproot_layer_number - 1)
            if tran > 0.5 * self.root_weights[self.taproot_layer_number - 1][plant_row_column][i]:
                tran = 0.5 * self.root_weights[self.taproot_layer_number - 1][plant_row_column][i]
            self.root_weights[self.taproot_layer_number][plant_row_column][i] += tran
            self.root_weights[self.taproot_layer_number - 1][plant_row_column][i] -= tran
            # second column
            tran = self.root_weights[self.taproot_layer_number - 1][klocp1][i] * 2 / dl(self.taproot_layer_number - 1)
            if tran > 0.5 * self.root_weights[self.taproot_layer_number - 1][klocp1][i]:
                tran = 0.5 * self.root_weights[self.taproot_layer_number - 1][klocp1][i]
            self.root_weights[self.taproot_layer_number][klocp1][i] += tran
            self.root_weights[self.taproot_layer_number - 1][klocp1][i] -= tran

    def add_fruiting_branch(self, k, density_factor, stemNRatio, time_to_new_fruiting_branch, new_node_initial_leaf_area, topping_date=None):
        """
        This function decides if a new fruiting branch is to be added to a vegetative branch, and forms it. It is called from function CottonPhenology().
        """
        if topping_date is not None and self.date >= topping_date:
            return
        # The following constant parameters are used:
        cdef double[8] vfrtbr = [0.8, 0.95, 33.0, 4.461, -0.1912, 0.00265, 1.8, -1.32]
        # Compute the cumulative delay for the appearance of the next caused by carbohydrate, nitrogen, and water stresses.
        vegetative_branch = self.vegetative_branches[k]
        vegetative_branch.delay_for_new_fruiting_branch += self.phenological_delay_for_vegetative_by_carbon_stress + vfrtbr[0] * self.phenological_delay_by_nitrogen_stress
        vegetative_branch.delay_for_new_fruiting_branch += vfrtbr[1] * (1 - self.water_stress)
        # Define nbrch and compute TimeToNextFruBranch, the time (in physiological days) needed for the formation of each successive fruiting branch, as a function of the average temperature. This function is derived from data of K. R. Reddy, CSRU, adjusted for age expressed in physiological days.
        # It is different for the main stem (k = 0) than for the other vegetative branches. TimeToNextFruNode is modified for plant density. Add DelayNewFruBranch to TimeToNextFruNode.
        last_fruiting_branch = vegetative_branch.fruiting_branches[-1]
        cdef double tav = last_fruiting_branch.nodes[0].average_temperature  # modified average daily temperature.
        if tav > vfrtbr[2]:
            tav = vfrtbr[2]
        # TimeToNextFruBranch is the time, in physiological days, for the next fruiting branch to be formed.
        cdef double TimeToNextFruBranch = time_to_new_fruiting_branch + tav * (vfrtbr[3] + tav * (vfrtbr[4] + tav * vfrtbr[5]))
        if k > 0:
            TimeToNextFruBranch = TimeToNextFruBranch * vfrtbr[6]
        TimeToNextFruBranch = TimeToNextFruBranch * (1 + vfrtbr[7] * (1 - density_factor)) + vegetative_branch.delay_for_new_fruiting_branch
        # Check if the the age of the last fruiting branch exceeds TimeToNextFruBranch. If so, form the new fruiting branch:
        if last_fruiting_branch.nodes[0].age < TimeToNextFruBranch:
            return
        # Increment NumFruitBranches, define newbr, and assign 1 to NumNodes, FruitFraction and FruitingCode.
        vegetative_branch.number_of_fruiting_branches += 1
        if vegetative_branch.number_of_fruiting_branches > 30:
            vegetative_branch.number_of_fruiting_branches = 30
            return
        if self.version >= 0x500:
            leaf_weight = min(new_node_initial_leaf_area * self.leaf_weight_area_ratio, self.stem_weight - 0.2)
            if leaf_weight <= 0:
                return
            leaf_area = leaf_weight / self.leaf_weight_area_ratio
        else:
            leaf_area = new_node_initial_leaf_area
            leaf_weight = leaf_area * self.leaf_weight_area_ratio
        cdef int newbr  # the index number of the new fruiting branch on this vegetative branch, after a new branch has been added.
        newbr = vegetative_branch.number_of_fruiting_branches - 1
        new_branch = vegetative_branch.fruiting_branches[-1]
        new_branch.number_of_fruiting_nodes = 1
        new_node = new_branch.nodes[0]
        new_node.fraction = 1
        new_node.stage = Stage.Square
        # Initiate new leaves at the first node of the new fruiting branch, and at the corresponding main stem node. The mass and nitrogen in the new leaves is substacted from the stem.
        new_node.leaf.area = leaf_area
        new_node.leaf.weight = leaf_weight
        main_stem_leaf = new_branch.main_stem_leaf

        main_stem_leaf.area = leaf_area
        main_stem_leaf.weight = leaf_weight
        self.stem_weight -= main_stem_leaf.weight + new_node.leaf.weight
        self.leaf_weight += main_stem_leaf.weight + new_node.leaf.weight
        # addlfn is the nitrogen added to new leaves from stem.
        cdef double addlfn = (main_stem_leaf.weight + new_node.leaf.weight) * stemNRatio
        self.leaf_nitrogen += addlfn
        self.stem_nitrogen -= addlfn
        # Begin computing AvrgNodeTemper of the new node and assign zero to DelayNewFruBranch.
        new_node.average_temperature = self.average_temperature
        vegetative_branch.delay_for_new_fruiting_branch = 0

    #begin root
    #                  THE COTTON ROOT SUB-MODEL.
    # The following is a documentation of the root sub-model used in COTTON2K. It is
    # derived from the principles of RHIZOS, as implemented in GOSSYM and in GLYCIM,
    # and from some principles of ROOTSIMU (Hoogenboom and Huck, 1986). It is devised
    # to be generally applicable, and may be used with root systems of different crops by
    # redefining the parameters, which are set here as constants, and some of them are
    # set in function InitializeRootData(). These parameters are of course specific for the
    # crop species, and perhaps also for cultivars or cultivar groups.
    #
    # This is a two-dimensional model and it may be used with soil cells of different
    # sizes. The grid can be defined by the modeler. The maximum numbers of layers
    # and columns are given by the parameters maxl and maxk, respectively. These are set
    # to 40 and 20, in this version of COTTON2K. The grid is set in function InitializeGrid().
    #
    # The whole slab is being simulated. Thus, non-symmetrical processes (such as
    # side-dressing of fertilizers or drip-irrigation) can be handled. The plant is assumed to
    # be situated at the center of the soil slab, or off-center for skip-rows. Adjoining soil
    # slabs are considered as mirror-images of each other. Alternate-row drip systems (or any
    # other agricultural input similarly situated) are located at one edge of the slab.
    #
    # The root mass in each cell is made up of NumRootAgeGroups classes, whose number is to be
    # defined by the modeler. The maximum number of classes is 3 in this version of COTTON2K.
    def lateral_root_growth_left(self, int l, int NumRootAgeGroups, unsigned int plant_row_column, double row_space):
        """This function computes the elongation of the lateral roots in a soil layer(l) to the left."""
        # The following constant parameters are used:
        cdef double p1 = 0.10  # constant parameter.
        cdef double rlatr = 3.6  # potential growth rate of lateral roots, cm/day.
        cdef double rtran = 0.2  # the ratio of root mass transferred to a new soil
        # soil cell, when a lateral root grows into it.
        # On its initiation, lateral root length is assumed to be equal to the width of a soil column soil cell at the location of the taproot.
        if self.rlat1[l] <= 0:
            self.rlat1[l] = wk(plant_row_column, row_space)
        cdef double stday  # daily average soil temperature (C) at root tip.
        stday = self.soil_temperature[l][plant_row_column] - 273.161
        cdef double temprg  # the effect of soil temperature on root growth.
        temprg = SoilTemOnRootGrowth(stday)
        # Define the column with the tip of this lateral root (ktip)
        cdef int ktip = 0  # column with the tips of the laterals to the left
        cdef double sumwk = 0  # summation of columns width
        for k in reversed(range(plant_row_column + 1)):
            sumwk += wk(k, row_space)
            if sumwk >= self.rlat1[l]:
                ktip = k
                break
        # Compute growth of the lateral root to the left.
        # Potential growth rate (u) is modified by the soil temperature function,
        # and the linearly modified effect of soil resistance (RootGroFactor).
        # Lateral root elongation does not occur in water logged soil.
        if self._[0].soil.cells[l][ktip].water_content < PoreSpace[l]:
            self.rlat1[l] += rlatr * temprg * (1 - p1 + self._[0].soil.cells[l][ktip].root.growth_factor * p1)
            # If the lateral reaches a new soil soil cell: a proportion (tran) of mass of roots is transferred to the new soil cell.
            if self.rlat1[l] > sumwk and ktip > 0:
                # column into which the tip of the lateral grows to left.
                newktip = ktip - 1
                for i in range(NumRootAgeGroups):
                    tran = self.root_weights[l][ktip][i] * rtran
                    self.root_weights[l][ktip][i] -= tran
                    self.root_weights[l][newktip][i] += tran
                # RootAge is initialized for this soil cell.
                # RootColNumLeft of this layer idefi
                if self.root_age[l][newktip] == 0:
                    self.root_age[l][newktip] = 0.01
                if newktip < self._[0].soil.layers[l].number_of_left_columns_with_root:
                    self._[0].soil.layers[l].number_of_left_columns_with_root = newktip

    def lateral_root_growth_right(self, int l, int NumRootAgeGroups, unsigned int plant_row_column, double row_space):
        # The following constant parameters are used:
        cdef double p1 = 0.10  # constant parameter.
        cdef double rlatr = 3.6  # potential growth rate of lateral roots, cm/day.
        cdef double rtran = 0.2  # the ratio of root mass transferred to a new soil
        # soil cell, when a lateral root grows into it.
        # On its initiation, lateral root length is assumed to be equal to the width of a soil column soil cell at the location of the taproot.
        cdef int klocp1 = plant_row_column + 1
        if self.rlat2[l] <= 0:
            self.rlat2[l] = wk(klocp1, row_space)
        cdef double stday  # daily average soil temperature (C) at root tip.
        stday = self.soil_temperature[l][klocp1] - 273.161
        cdef double temprg  # the effect of soil temperature on root growth.
        temprg = SoilTemOnRootGrowth(stday)
        # define the column with the tip of this lateral root (ktip)
        cdef int ktip = 0  # column with the tips of the laterals to the right
        cdef double sumwk = 0
        for k in range(klocp1, nk):
            sumwk += wk(k, row_space)
            if sumwk >= self.rlat2[l]:
                ktip = k
                break
        # Compute growth of the lateral root to the right. Potential growth rate is modified by the soil temperature function, and the linearly modified effect of soil resistance (RootGroFactor).
        # Lateral root elongation does not occur in water logged soil.
        if self._[0].soil.cells[l][ktip].water_content < PoreSpace[l]:
            self.rlat2[l] += rlatr * temprg * (1 - p1 + self._[0].soil.cells[l][ktip].root.growth_factor * p1)
            # If the lateral reaches a new soil soil cell: a proportion (tran) of mass of roots is transferred to the new soil cell.
            if self.rlat2[l] > sumwk and ktip < nk - 1:
                # column into which the tip of the lateral grows to left.
                newktip = ktip + 1  # column into which the tip of the lateral grows to left.
                for i in range(NumRootAgeGroups):
                    tran = self.root_weights[l][ktip][i] * rtran
                    self.root_weights[l][ktip][i] -= tran
                    self.root_weights[l][newktip][i] += tran
                # RootAge is initialized for this soil cell.
                # RootColNumLeft of this layer is redefined.
                if self.root_age[l][newktip] == 0:
                    self.root_age[l][newktip] = 0.01
                if newktip > self._[0].soil.layers[l].number_of_right_columns_with_root:
                    self._[0].soil.layers[l].number_of_right_columns_with_root = newktip

    def potential_root_growth(self, NumRootAgeGroups, per_plant_area):
        """
        This function calculates the potential root growth rate.
        The return value is the sum of potential root growth rates for the whole slab.
        It is called from PlantGrowth().
        It calls: RootImpedance(), SoilNitrateOnRootGrowth(), SoilAirOnRootGrowth(), SoilMechanicResistance(), SoilTemOnRootGrowth() and root_psi().
        """
        # The following constant parameter is used:
        cdef double rgfac = 0.36  # potential relative growth rate of the roots (g/g/day).
        # Initialize to zero the PotGroRoots array.
        self._root_potential_growth[:] = 0
        self.soil.root_impedance()
        for l in range(self.soil.number_of_layers_with_root):
            for k in range(nk):
                # Check if this soil cell contains roots (if RootAge is greater than 0), and execute the following if this is true.
                # In each soil cell with roots, the root weight capable of growth rtwtcg is computed as the sum of RootWeight[l][k][i] * cgind[i] for all root classes.
                if self.root_age[l][k] > 0:
                    rtwtcg = 0  # root weight capable of growth in a soil soil cell.
                    for i in range(NumRootAgeGroups):
                        rtwtcg += self.root_weights[l][k][i] * cgind[i]
                    # Compute the temperature factor for root growth by calling function SoilTemOnRootGrowth() for this layer.
                    stday = self.soil_temperature[l][k] - 273.161  # soil temperature, C, this day's average for this cell.
                    temprg = SoilTemOnRootGrowth(stday)  # effect of soil temperature on root growth.
                    # Compute soil mechanical resistance for each soil cell by calling SoilMechanicResistance{}, the effect of soil aeration on root growth by calling SoilAirOnRootGrowth(), and the effect of soil nitrate on root growth by calling SoilNitrateOnRootGrowth().
                    rtpct # effect of soil mechanical resistance on root growth (returned from SoilMechanicResistance).

                    lp1 = l if l == nl - 1 else l + 1  # layer below l.

                    # columns to the left and to the right of k.
                    kp1 = min(k + 1, nk - 1)
                    km1 = max(k - 1, 0)

                    rtimpd0 = RootImpede[l][k]
                    rtimpdkm1 = RootImpede[l][km1]
                    rtimpdkp1 = RootImpede[l][kp1]
                    rtimpdlp1 = RootImpede[lp1][k]
                    rtimpdmin = min(rtimpd0, rtimpdkm1, rtimpdkp1, rtimpdlp1)  # minimum value of rtimpd
                    rtpct = SoilMechanicResistance(rtimpdmin)
                    # effect of oxygen deficiency on root growth (returned from SoilAirOnRootGrowth).
                    rtrdo = SoilAirOnRootGrowth(SoilPsi[l][k], PoreSpace[l], self._[0].soil.cells[l][k].water_content)
                    # effect of nitrate deficiency on root growth (returned from SoilNitrateOnRootGrowth).
                    rtrdn = SoilNitrateOnRootGrowth(self._[0].soil.cells[l][k].nitrate_nitrogen_content)
                    # The root growth resistance factor RootGroFactor(l,k), which can take a value between 0 and 1, is computed as the minimum of these resistance factors. It is further modified by multiplying it by the soil moisture function root_psi().
                    # Potential root growth PotGroRoots(l,k) in each cell is computed as a product of rtwtcg, rgfac, the temperature function temprg, and RootGroFactor(l,k). It is also multiplied by per_plant_area / 19.6, for the effect of plant population density on root growth: it is made comparable to a population of 5 plants per m in 38" rows.
                    self._[0].soil.cells[l][k].root.growth_factor = root_psi(SoilPsi[l][k]) * min(rtrdo, rtpct, rtrdn)
                    self._root_potential_growth[l][k] = rtwtcg * rgfac * temprg * self._[0].soil.cells[l][k].root.growth_factor * per_plant_area / 19.6
        return self._root_potential_growth.sum()

    def redist_root_new_growth(self, int l, int k, double addwt, double column_width, unsigned int plant_row_column):
        """This function computes the redistribution of new growth of roots into adjacent soil cells. It is called from ActualRootGrowth().

        Redistribution is affected by the factors rgfdn, rgfsd, rgfup.
        And the values of RootGroFactor(l,k) in this soil cell and in the adjacent cells.
        The values of ActualRootGrowth(l,k) for this and for the adjacent soil cells are computed.
        The code of this module is based, with major changes, on the code of GOSSYM."""
        # The following constant parameters are used. These are relative factors for root growth to adjoining cells, downwards, sideways, and upwards, respectively. These factors are relative to the volume of the soil cell from which growth originates.
        cdef double rgfdn = 900
        cdef double rgfsd = 600
        cdef double rgfup = 10
        # Set the number of layer above and below this layer, and the number of columns to the right and to the left of this column.
        cdef int lm1, lp1  # layer above and below layer l.
        lp1 = min(nl - 1, l + 1)
        lm1 = max(0, l - 1)

        cdef int km1, kp1  # column to the left and to the right of column k.
        kp1 = min(nk - 1, k + 1)
        km1 = max(0, k - 1)
        # Compute proportionality factors (efac1, efacl, efacr, efacu, efacd) as the product of RootGroFactor and the geotropic factors in the respective soil cells.
        # Note that the geotropic factors are relative to the volume of the soil cell.
        # Compute the sum srwp of the proportionality factors.
        cdef double efac1  # product of RootGroFactor and geotropic factor for this cell.
        cdef double efacd  # as efac1 for the cell below this cell.
        cdef double efacl  # as efac1 for the cell to the left of this cell.
        cdef double efacr  # as efac1 for the cell to the right of this cell.
        cdef double efacu  # as efac1 for the cell above this cell.
        cdef double srwp  # sum of all efac values.
        efac1 = dl(l) * column_width * self.soil.cells[l][k].root.growth_factor
        efacl = rgfsd * self.soil.cells[l][km1].root.growth_factor
        efacr = rgfsd * self.soil.cells[l][kp1].root.growth_factor
        efacu = rgfup * self.soil.cells[lm1][k].root.growth_factor
        efacd = rgfdn * self.soil.cells[lp1][k].root.growth_factor
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
            if self._[0].soil.layers[lp1].number_of_left_columns_with_root == 0 or k < self._[0].soil.layers[lp1].number_of_left_columns_with_root:
                self._[0].soil.layers[lp1].number_of_left_columns_with_root = k
            if self._[0].soil.layers[lp1].number_of_right_columns_with_root == 0 or k > self._[0].soil.layers[lp1].number_of_right_columns_with_root:
                self._[0].soil.layers[lp1].number_of_right_columns_with_root = k
        # If this is in the location of the taproot, and the roots reach a new soil layer, update the taproot parameters taproot_length, self.last_layer_with_root_depth, and self.taproot_layer_number.
        if k == plant_row_column or k == plant_row_column + 1:
            if lp1 > self.taproot_layer_number and efacd > 0:
                self.taproot_length = self.last_layer_with_root_depth + 0.01
                self.last_layer_with_root_depth += dl(lp1)
                self.taproot_layer_number = lp1
        # Update state.soil.number_of_layers_with_root, if necessary, and the values of RootColNumLeft and RootColNumRight for this layer.
        if self.soil.number_of_layers_with_root <= l and efacd > 0:
            self.soil.number_of_layers_with_root = l + 1
        if km1 < self._[0].soil.layers[l].number_of_left_columns_with_root:
            self._[0].soil.layers[l].number_of_left_columns_with_root = km1
        if kp1 > self._[0].soil.layers[l].number_of_right_columns_with_root:
            self._[0].soil.layers[l].number_of_right_columns_with_root = kp1

    def initialize_lateral_roots(self):
        """This function initiates lateral root growth."""
        cdef double distlr = 12  # the minimum distance, in cm, from the tip of the taproot, for a lateral root to be able to grow.
        cdef double sdl  # distance of a layer from tip of taproot, cm.
        sdl = self.taproot_length - self.last_layer_with_root_depth
        # Loop on soil layers, from the lowest layer with roots upward:
        for l in reversed(range(self.taproot_layer_number + 1)):
            # Compute distance from tip of taproot.
            sdl += dl(l)
            # If a layer is marked for a lateral (LateralRootFlag[l] = 1) and its distance from the tip is larger than distlr - initiate a lateral (LateralRootFlag[l] = 2).
            if sdl > distlr and LateralRootFlag[l] == 1:
                LateralRootFlag[l] = 2

    def lateral_root_growth(self, NumRootAgeGroups, plant_row_column, row_space):
        # Call functions for growth of lateral roots
        self.initialize_lateral_roots()
        for l in range(self.taproot_layer_number):
            if LateralRootFlag[l] == 2:
                self.lateral_root_growth_left(l, NumRootAgeGroups, plant_row_column, row_space)
                self.lateral_root_growth_right(l, NumRootAgeGroups, plant_row_column, row_space)
    #end root

    def pre_fruiting_node_leaf_abscission(self, droplf, first_square_date, defoliate_date):
        """This function simulates the abscission of prefruiting node leaves.

        Arguments
        ---------
        droplf
            leaf age until it is abscised.
        """
        # Loop over all prefruiting nodes. If it is after first square, node age is updated here.
        for j in range(self.number_of_pre_fruiting_nodes):
            if first_square_date is not None:
                self._[0].age_of_pre_fruiting_nodes[j] += self.day_inc
            # The leaf on this node is abscised if its age has reached droplf, and if there is a leaf here, and if LeafAreaIndex is not too small:
            # Update state.leaf_area, AbscisedLeafWeight, state.leaf_weight, state.petiole_weight, CumPlantNLoss.
            # Assign zero to state.leaf_area_pre_fruiting, PetioleWeightPreFru and state.leaf_weight_pre_fruiting of this leaf.
            # If a defoliation was applied.
            if self.age_of_pre_fruiting_nodes[j] >= droplf and self.leaf_area_pre_fruiting[j] > 0 and self.leaf_area_index > 0.1:
                self.leaf_area -= self.leaf_area_pre_fruiting[j]
                self.leaf_weight -= self.leaf_weight_pre_fruiting[j]
                self.petiole_weight -= PetioleWeightPreFru[j]
                self.leaf_nitrogen -= self.leaf_weight_pre_fruiting[j] * self.leaf_nitrogen_concentration
                self.petiole_nitrogen -= PetioleWeightPreFru[j] * self.petiole_nitrogen_concentration
                self.cumulative_nitrogen_loss += self.leaf_weight_pre_fruiting[j] * self.leaf_nitrogen_concentration + PetioleWeightPreFru[j] * self.petiole_nitrogen_concentration
                self._[0].leaf_area_pre_fruiting[j] = 0
                self._[0].leaf_weight_pre_fruiting[j] = 0
                PetioleWeightPreFru[j] = 0

    def defoliation_leaf_abscission(self, defoliate_date):
        """Simulates leaf abscission caused by defoliants."""
        # When this is the first day of defoliation - if there are any leaves left on the prefruiting nodes, they will be shed at this stage.
        if self.date == defoliate_date:
            for j in range(self.number_of_pre_fruiting_nodes):
                if self.leaf_area_pre_fruiting[j] > 0:
                    self.leaf_area -= self.leaf_area_pre_fruiting[j]
                    self.leaf_area_pre_fruiting[j] = 0
                    self.leaf_weight -= self.leaf_weight_pre_fruiting[j]
                    self.petiole_weight -= PetioleWeightPreFru[j]
                    self.leaf_nitrogen -= self.leaf_weight_pre_fruiting[j] * self.leaf_nitrogen_concentration
                    self.petiole_nitrogen -= PetioleWeightPreFru[j] * self.petiole_nitrogen_concentration
                    self.cumulative_nitrogen_loss += self.leaf_weight_pre_fruiting[j] * self.leaf_nitrogen_concentration + PetioleWeightPreFru[j] * self.petiole_nitrogen_concentration
                    self.leaf_weight_pre_fruiting[j] = 0
                    PetioleWeightPreFru[j] = 0
        # When this is after the first day of defoliation - count the number of existing leaves and sort them by age
        if self.date == defoliate_date:
            return
        leaves = []
        for k in range(self.number_of_vegetative_branches):
            for l in range(self.vegetative_branches[k].number_of_fruiting_branches):
                if self.vegetative_branches[k].fruiting_branches[l].main_stem_leaf.weight > 0:
                    leaves.append((self.vegetative_branches[k].fruiting_branches[l].nodes[0].age, k, l, 66))
                    # 66 indicates this leaf is at the base of the fruiting branch
                for m in range(self.vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes):
                    if self.vegetative_branches[k].fruiting_branches[l].nodes[m].leaf.weight > 0:
                        leaves.append((self.vegetative_branches[k].fruiting_branches[l].nodes[m].age, k, l, m))
        # Compute the number of leaves to be shed on this day (numLeavesToShed).
        numLeavesToShed = int(len(leaves) * PercentDefoliation / 100)  # the computed number of leaves to be shed.
        # Execute leaf shedding according to leaf age.
        for leaf in sorted(leaves, reverse=True):
            if numLeavesToShed <= 0:
                break
            if numLeavesToShed > 0 and leaf[0] > 0:
                k, l, m = leaf[1:]
                if m == 66:  # main stem leaves
                    main_stem_leaf = self.vegetative_branches[k].fruiting_branches[l].main_stem_leaf
                    self.leaf_weight -= main_stem_leaf.weight
                    self.petiole_weight -= main_stem_leaf.petiole_weight
                    self.leaf_nitrogen -= main_stem_leaf.weight * self.leaf_nitrogen_concentration
                    self.petiole_nitrogen -= main_stem_leaf.petiole_weight * self.petiole_nitrogen_concentration
                    self.cumulative_nitrogen_loss += main_stem_leaf.weight * self.leaf_nitrogen_concentration + main_stem_leaf.petiole_weight * self.petiole_nitrogen_concentration
                    self.leaf_area -= main_stem_leaf.area
                    main_stem_leaf.area = 0
                    main_stem_leaf.weight = 0
                    main_stem_leaf.petiole_weight = 0
                else:  # leaves on fruit nodes
                    site = self.vegetative_branches[k].fruiting_branches[l].nodes[m]
                    self.leaf_weight -= site.leaf.weight
                    self.petiole_weight -= site.petiole.weight
                    self.leaf_nitrogen -= site.leaf.weight * self.leaf_nitrogen_concentration
                    self.petiole_nitrogen -= site.petiole.weight * self.petiole_nitrogen_concentration
                    self.cumulative_nitrogen_loss += site.leaf.weight * self.leaf_nitrogen_concentration + site.petiole.weight * self.petiole_nitrogen_concentration
                    self.leaf_area -= site.leaf.area
                    site.leaf.area = 0
                    site.leaf.weight = 0
                    site.petiole.weight = 0
                numLeavesToShed -= 1

    #begin phenology
    def fruiting_sites_abscission(self):
        """This function simulates the abscission of squares and bolls."""
        global NumSheddingTags
        # The following constant parameters are used:
        cdef double[9] vabsfr = [21.0, 0.42, 30.0, 0.05, 6.0, 2.25, 0.60, 5.0, 0.20]
        # Update tags for shedding: Increment NumSheddingTags by 1, and move the array members of ShedByCarbonStress, ShedByNitrogenStress, ShedByWaterStress, and AbscissionLag.
        NumSheddingTags += 1
        if NumSheddingTags > 1:
            for lt in range(NumSheddingTags - 1, 0, -1):
                ltm1 = lt - 1
                ShedByCarbonStress[lt] = ShedByCarbonStress[ltm1]
                ShedByNitrogenStress[lt] = ShedByNitrogenStress[ltm1]
                ShedByWaterStress[lt] = ShedByWaterStress[ltm1]
                AbscissionLag[lt] = AbscissionLag[ltm1]
        # Calculate shedding intensity: The shedding intensity due to stresses of this day is assigned to the first members of the arrays ShedByCarbonStress, ShedByNitrogenStress, and ShedByWaterStress.
        if self.carbon_stress < self._sim.cultivar_parameters[43]:
            ShedByCarbonStress[0] = (self._sim.cultivar_parameters[43] - self.carbon_stress) / self._sim.cultivar_parameters[43]
        else:
            ShedByCarbonStress[0] = 0
        if self.nitrogen_stress < vabsfr[1]:
            ShedByNitrogenStress[0] = (vabsfr[1] - self.nitrogen_stress) / vabsfr[1]
        else:
            ShedByNitrogenStress[0] = 0
        if self.water_stress < self._sim.cultivar_parameters[44]:
            ShedByWaterStress[0] = (self._sim.cultivar_parameters[44] - self.water_stress) / self._sim.cultivar_parameters[44]
        else:
            ShedByWaterStress[0] = 0
        # Assign 0.01 to the first member of AbscissionLag.
        AbscissionLag[0] = 0.01
        # Updating age of tags for shedding: Each member of array AbscissionLag is incremented by physiological age of today. It is further increased (i.e., shedding will occur sooner) when maximum temperatures are high.
        cdef double tmax = self._sim.climate[(self.date - self._sim.start_date).days]["Tmax"]
        for lt in range(NumSheddingTags):
            AbscissionLag[lt] += max(self.day_inc, 0.40)
            if tmax > vabsfr[2]:
                AbscissionLag[lt] += (tmax - vabsfr[2]) * vabsfr[3]
        # Assign zero to idecr, and start do loop over all days since first relevant tagging. If AbscissionLag reaches a value of vabsfr[4], calculate actual shedding of each site:
        idecr = 0  # decrease in NumSheddingTags after shedding has been executed.
        for lt in range(NumSheddingTags):
            if AbscissionLag[lt] >= vabsfr[4] or lt >= 20:
            # Start loop over all possible fruiting sites. The abscission functions will be called for sites that are squares or green bolls.
                for k in range(self.number_of_vegetative_branches):
                    for l in range(self.vegetative_branches[k].number_of_fruiting_branches):
                        for m in range(self.vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes):
                            site = self.vegetative_branches[k].fruiting_branches[l].nodes[m]
                            if site.stage in (Stage.Square, Stage.YoungGreenBoll, Stage.GreenBoll):
                                # ratio of abscission for a fruiting site.
                                abscissionRatio = self.site_abscission_ratio(k, l, m, lt)
                                if abscissionRatio > 0:
                                    if site.stage == Stage.Square:
                                        self.square_abscission(site, abscissionRatio)
                                    else:
                                        self.boll_abscission(site, abscissionRatio, self.ginning_percent if self.ginning_percent > 0 else site.ginning_percent)
                # Assign zero to the array members for this day.
                ShedByCarbonStress[lt] = 0
                ShedByNitrogenStress[lt] = 0
                ShedByWaterStress[lt] = 0
                AbscissionLag[lt] = 0
                idecr += 1
        # Decrease NumSheddingTags. If plantmap adjustments are necessary for square number, or green boll number, or open boll number - call AdjustAbscission().
        NumSheddingTags -= idecr

        self.compute_site_numbers()

    def site_abscission_ratio(self, k, l, m, lt):
        """This function computes and returns the probability of abscission of a single site (k, l, m).

        Arguments
        ---------
        k, l, m
            indices defining position of this site.
        lt
            lag index for this node.
        """
        site = self.vegetative_branches[k].fruiting_branches[l].nodes[m]
        # The following constant parameters are used:
        cdef double[5] vabsc = [21.0, 2.25, 0.60, 5.0, 0.20]
        VarPar = self._sim.cultivar_parameters

        # For each site, compute the probability of its abscission (pabs) as afunction of site age, and the total shedding ratio (shedt) as a function of plant stresses that occurred when abscission was triggered.
        pabs = 0  # probability of abscission of a fruiting site.
        shedt = 0  # total shedding ratio, caused by various stresses.
        # (1) Squares (FruitingCode = 1).
        if site.stage == Stage.Square:
            if site.age < vabsc[3]:
                pabs = 0  # No abscission of very young squares (AgeOfSite less than vabsc(3))
            else:
                # square age after becoming susceptible to shedding.
                xsqage = site.age - vabsc[3]
                if xsqage >= vabsc[0]:
                    pabs = VarPar[46]  # Old squares have a constant probability of shedding.
                else:
                    # Between these limits, pabs is a function of xsqage.
                    pabs = VarPar[46] + (VarPar[45] - VarPar[46]) * pow(((vabsc[0] - xsqage) / vabsc[0]), vabsc[1])
            # Total shedding ratio (shedt) is a product of the effects of carbohydrate stress and nitrogen stress.
            shedt = 1 - (1 - ShedByCarbonStress[lt]) * (1 - ShedByNitrogenStress[lt])
        # (2) Very young bolls (FruitingCode = 7, and AgeOfBoll less than VarPar[47]).
        elif site.stage == Stage.YoungGreenBoll and site.boll.age <= VarPar[47]:
            # There is a constant probability of shedding (VarPar[48]), and shedt is a product of the effects carbohydrate, and nitrogen stresses. Note that nitrogen stress has only a partial effect in this case, as modified by vabsc[2].
            pabs = VarPar[48]
            shedt = 1 - (1 - ShedByCarbonStress[lt]) * (1 - vabsc[2] * ShedByNitrogenStress[lt])
        # (3) Medium age bolls (AgeOfBoll between VarPar[47] and VarPar[47] + VarPar[49]).
        elif site.boll.age > VarPar[47] and site.boll.age <= (VarPar[47] + VarPar[49]):
            # pabs is linearly decreasing with age, and shedt is a product of the effects carbohydrate, nitrogen and water stresses.  Note that nitrogen stress has only a partial effect in this case, as modified by vabsc[4].
            pabs = VarPar[48] - (VarPar[48] - VarPar[50]) * (site.boll.age - VarPar[47]) / VarPar[49]
            shedt = 1 - (1 - ShedByCarbonStress[lt]) * (1 - vabsc[4] * ShedByNitrogenStress[lt]) * (1 - ShedByWaterStress[lt])
        # (4) Older bolls (AgeOfBoll between VarPar[47] + VarPar[49] and VarPar[47] + 2*VarPar[49]).
        elif site.boll.age > (VarPar[47] + VarPar[49]) and site.boll.age <= (VarPar[47] + 2 * VarPar[49]):
            # pabs is linearly decreasing with age, and shedt is affected only by water stress.
            pabs = VarPar[50] / VarPar[49] * (VarPar[47] + 2 * VarPar[49] - site.boll.age)
            shedt = ShedByWaterStress[lt]
        # (5) bolls older than VarPar[47] + 2*VarPar[49]
        elif site.boll.age > (VarPar[47] + 2 * VarPar[49]):
            pabs = 0  # no abscission
        # Actual abscission of tagged sites (abscissionRatio) is a product of pabs, shedt and DayInc for this day. It can not be greater than 1.
        return min(pabs * shedt * self.day_inc, 1)

    def square_abscission(self, site, abscissionRatio):
        """Simulates the abscission of a single square at site (k, l, m).

        Arguments
        ---------
        abscissionRatio
            ratio of abscission of a fruiting site."""
        # Compute the square weight lost by shedding (wtlos) as a proportion of SquareWeight of this site. Update state.square_nitrogen, CumPlantNLoss, SquareWeight[k][l][m], BloomWeightLoss, state.square_weight and FruitFraction[k][l][m].
        cdef double wtlos = site.square.weight * abscissionRatio  # weight lost by shedding at this site.
        self.square_nitrogen -= wtlos * self.square_nitrogen_concentration
        self.cumulative_nitrogen_loss += wtlos * self.square_nitrogen_concentration
        site.square.weight -= wtlos
        self.square_weight -= wtlos
        site.fraction *= (1 - abscissionRatio)
        # If FruitFraction[k][l][m] is less than 0.001 make it zero, and update state.square_nitrogen, CumPlantNLoss, BloomWeightLoss, state.square_weight, SquareWeight[k][l][m], and assign 5 to FruitingCode.
        if site.fraction <= 0.001:
            site.fraction = 0
            self.square_nitrogen -= site.square.weight * self.square_nitrogen_concentration
            self.cumulative_nitrogen_loss += site.square.weight * self.square_nitrogen_concentration
            self.square_weight -= site.square.weight
            site.square.weight = 0
            site.stage = Stage.AbscisedAsSquare

    def boll_abscission(self, site, abscissionRatio, gin1):
        """This function simulates the abscission of a single green boll at site (k, l, m). It is called from function FruitingSitesAbscission() if this site is a green boll.

        Arguments
        ---------
        abscissionRatio
            ratio of abscission of a fruiting site.
        gin1
            percent of seeds in seedcotton, used to compute lost nitrogen.
        """
        # Update state.seed_nitrogen, state.burr_nitrogen, CumPlantNLoss, state.green_bolls_weight, state.green_bolls_burr_weight, BollWeight[k][l][m], state.site[k][l][m].burr.weight, and FruitFraction[k][l][m].
        self.seed_nitrogen -= site.boll.weight * abscissionRatio * (1 - gin1) * self.seed_nitrogen_concentration
        self.burr_nitrogen -= site.burr.weight * abscissionRatio * self.burr_nitrogen_concentration
        self.cumulative_nitrogen_loss += site.boll.weight * abscissionRatio * (1. - gin1) * self.seed_nitrogen_concentration
        self.cumulative_nitrogen_loss += site.burr.weight * abscissionRatio * self.burr_nitrogen_concentration
        self.green_bolls_weight -= site.boll.weight * abscissionRatio
        self.green_bolls_burr_weight -= site.burr.weight * abscissionRatio
        site.boll.weight -= site.boll.weight * abscissionRatio
        site.burr.weight -= site.burr.weight * abscissionRatio
        site.fraction -= site.fraction * abscissionRatio

        # If FruitFraction[k][l][m] is less than 0.001 make it zero, update state.seed_nitrogen, state.burr_nitrogen, CumPlantNLoss, state.green_bolls_weight, state.green_bolls_burr_weight, BollWeight[k][l][m], state.site[k][l][m].burr.weight, and assign 4 to FruitingCode.

        if site.fraction <= 0.001:
            site.stage = Stage.AbscisedAsBoll
            self.seed_nitrogen -= site.boll.weight * (1 - gin1) * self.seed_nitrogen_concentration
            self.burr_nitrogen -= site.burr.weight * self.burr_nitrogen_concentration
            self.cumulative_nitrogen_loss += site.boll.weight * (1 - gin1) * self.seed_nitrogen_concentration
            self.cumulative_nitrogen_loss += site.burr.weight * self.burr_nitrogen_concentration
            site.fraction = 0
            self.green_bolls_weight -= site.boll.weight
            self.green_bolls_burr_weight -= site.burr.weight
            site.boll.weight = 0
            site.burr.weight = 0

    def compute_site_numbers(self):
        """Calculates square, green boll, open boll, and abscised site numbers (NumSquares, NumGreenBolls, NumOpenBolls, and AbscisedFruitSites, respectively), as the sums of FruitFraction in all sites with appropriate FruitingCode."""
        self.number_of_squares = 0
        self.number_of_green_bolls = 0
        self.number_of_open_bolls = 0
        for k in range(self.number_of_vegetative_branches):
            for l in range(self.vegetative_branches[k].number_of_fruiting_branches):
                for m in range(self.vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes):
                    site = self.vegetative_branches[k].fruiting_branches[l].nodes[m]
                    if site.stage == Stage.Square:
                        self.number_of_squares += site.fraction
                    elif site.stage == Stage.YoungGreenBoll or site.stage == Stage.GreenBoll:
                        self.number_of_green_bolls += site.fraction
                    elif site.stage == Stage.MatureBoll:
                        self.number_of_open_bolls += site.fraction

    def new_boll_formation(self, site):
        """Simulates the formation of a new boll at a fruiting site."""
        # The following constant parameters are used:
        cdef double seedratio = 0.64  # ratio of seeds in seedcotton weight.
        cdef double[2] vnewboll = [0.31, 0.02]
        # If bPollinSwitch is false accumulate number of blooms to be dropped, and define FruitingCode as 6.
        if not self.pollination_switch:
            site.stage = Stage.AbscisedAsFlower
            site.fraction = 0
            site.square.weight = 0
            return
        # The initial weight of the new boll (BollWeight) and new burr (state.burr_weight) will be a fraction of the square weight, and the rest will be added to BloomWeightLoss. 80% of the initial weight will be in the burr.
        # The nitrogen in the square is partitioned in the same proportions. The nitrogen that was in the square is transferred to the burrs. Update state.green_bolls_weight, state.green_bolls_burr_weight and state.square_weight. assign zero to SquareWeight at this site.
        cdef double bolinit  # initial weight of boll after flowering.
        bolinit = vnewboll[0] * site.square.weight
        site.boll.weight = 0.2 * bolinit
        site.burr.weight = bolinit - site.boll.weight

        cdef double sqr1n  # the nitrogen content of one square before flowering.
        sqr1n = self.square_nitrogen_concentration * site.square.weight
        self.square_nitrogen -= sqr1n
        self.cumulative_nitrogen_loss += sqr1n * (1 - vnewboll[0])
        sqr1n = sqr1n * vnewboll[0]

        cdef double seed1n  # the nitrogen content of seeds in a new boll on flowering.
        seed1n = min(site.boll.weight * seedratio * vnewboll[1], sqr1n)
        self.seed_nitrogen += seed1n
        self.burr_nitrogen += sqr1n - seed1n

        self.green_bolls_weight += site.boll.weight
        self.green_bolls_burr_weight += site.burr.weight
        self.square_weight -= site.square.weight
        site.square.weight = 0
    #end phenology

    #begin soil
    def apply_fertilizer(self, row_space, plant_population):
        """This function simulates the application of nitrogen fertilizer on each date of application."""
        cdef double ferc = 0.01  # constant used to convert kgs per ha to mg cm-2
        # Loop over all fertilizer applications.
        for i in range(NumNitApps):
            # Check if fertilizer is to be applied on this day.
            if self.date.timetuple().tm_yday == NFertilizer[i].day:
                # If this is a BROADCAST fertilizer application:
                if NFertilizer[i].mthfrt == 0:
                    # Compute the number of layers affected by broadcast fertilizer incorporation (lplow), assuming that the depth of incorporation is 20 cm.
                    lplow = 0  # number of soil layers affected by cultivation
                    sdl = 0.0  # sum of depth of consecutive soil layers
                    for l in range(40):
                        sdl += dl(l)
                        if sdl >= 20:
                            lplow = l + 1
                            break
                    # Calculate the actual depth of fertilizer incorporation in the soil (fertdp) as the sum of all soil layers affected by incorporation.
                    fertdp = 0.0  # depth of broadcast fertilizer incorporation, cm
                    for l in range(lplow):
                        fertdp += dl(l)
                    # Update the nitrogen contents of all soil soil cells affected by this fertilizer application.
                    for l in range(lplow):
                        for k in range(20):
                            VolNh4NContent[l][k] += NFertilizer[i].amtamm * ferc / fertdp
                            self.soil.cells[l][k].nitrate_nitrogen_content += NFertilizer[i].amtnit * ferc / fertdp
                            VolUreaNContent[l][k] += NFertilizer[i].amtura * ferc / fertdp
                # If this is a FOLIAR fertilizer application:
                elif NFertilizer[i].mthfrt == 2:
                    # It is assumed that 70% of the amount of ammonium or urea intercepted by the canopy is added to the leaf N content (state.leaf_nitrogen).
                    self.leaf_nitrogen += 0.70 * self.light_interception * (NFertilizer[i].amtamm + NFertilizer[i].amtura) * 1000 / plant_population
                    # The amount not intercepted by the canopy is added to the soil. If the fertilizer is nitrate, it is assumed that all of it is added to the upper soil layer.
                    # Update nitrogen contents of the upper layer.
                    for k in range(20):
                        VolNh4NContent[0][k] += NFertilizer[i].amtamm * (1 - 0.70 * self.light_interception) * ferc / dl(0)
                        self.soil.cells[0][k].nitrate_nitrogen_content += NFertilizer[i].amtnit * ferc / dl(0)
                        VolUreaNContent[0][k] += NFertilizer[i].amtura * (1 - 0.70 * self.light_interception) * ferc / dl(0)
                # If this is a SIDE-DRESSING of N fertilizer:
                elif NFertilizer[i].mthfrt == 1:
                    # Define the soil column (ksdr) and the soil layer (lsdr) in which the side-dressed fertilizer is applied.
                    ksdr = NFertilizer[i].ksdr  # the column in which the side-dressed is applied
                    lsdr = NFertilizer[i].ksdr  # the layer in which the side-dressed is applied
                    n00 = 1  # number of soil soil cells in which side-dressed fertilizer is incorporated.
                    # If the volume of this soil cell is less than 100 cm3, it is assumed that the fertilizer is also incorporated in the soil cells below and to the sides of it.
                    if (dl(lsdr) * wk(ksdr, row_space)) < 100:
                        if ksdr < nk - 1:
                            n00 += 1
                        if ksdr > 0:
                            n00 += 1
                        if lsdr < nl - 1:
                            n00 += 1
                    # amount of ammonium N added to the soil by sidedressing (mg per cell)
                    addamm = NFertilizer[i].amtamm * ferc * row_space / n00
                    # amount of nitrate N added to the soil by sidedressing (mg per cell)
                    addnit = NFertilizer[i].amtnit * ferc * row_space / n00
                    # amount of urea N added to the soil by sidedressing (mg per cell)
                    addnur = NFertilizer[i].amtura * ferc * row_space / n00
                    # Update the nitrogen contents of these soil cells.
                    self.soil.cells[lsdr][ksdr].nitrate_nitrogen_content += addnit / (dl(lsdr) * wk(ksdr, row_space))
                    VolNh4NContent[lsdr][ksdr] += addamm / (dl(lsdr) * wk(ksdr, row_space))
                    VolUreaNContent[lsdr][ksdr] += addnur / (dl(lsdr) * wk(ksdr, row_space))
                    if (dl(lsdr) * wk(ksdr, row_space)) < 100:
                        if ksdr < nk - 1:
                            kp1 = ksdr + 1  # column to the right of ksdr.
                            self.soil.cells[lsdr][kp1].nitrate_nitrogen_content += addnit / (dl(lsdr) * wk(kp1, row_space))
                            VolNh4NContent[lsdr][kp1] += addamm / (dl(lsdr) * wk(kp1, row_space))
                            VolUreaNContent[lsdr][kp1] += addnur / (dl(lsdr) * wk(kp1, row_space))
                        if ksdr > 0:
                            km1 = ksdr - 1  # column to the left of ksdr.
                            self.soil.cells[lsdr][km1].nitrate_nitrogen_content += addnit / (dl(lsdr) * wk(km1, row_space))
                            VolNh4NContent[lsdr][km1] += addamm / (dl(lsdr) * wk(km1, row_space))
                            VolUreaNContent[lsdr][km1] += addnur / (dl(lsdr) * wk(km1, row_space))
                        if lsdr < nl - 1:
                            lp1 = lsdr + 1
                            self.soil.cells[lp1][ksdr].nitrate_nitrogen_content += addnit / (dl(lp1) * wk(ksdr, row_space))
                            VolNh4NContent[lp1][ksdr] += addamm / (dl(lp1) * wk(ksdr, row_space))
                            VolUreaNContent[lp1][ksdr] += addnur / (dl(lp1) * wk(ksdr, row_space))
                # If this is FERTIGATION (N fertilizer applied in drip irrigation):
                elif NFertilizer[i].mthfrt == 3:
                    # Convert amounts added to mg cm-3, and update the nitrogen content of the soil cell in which the drip outlet is situated.
                    VolNh4NContent[LocationLayerDrip][LocationColumnDrip] += NFertilizer[i].amtamm * ferc * row_space / (dl(LocationLayerDrip) * wk(LocationColumnDrip, row_space))
                    self.soil.cells[LocationLayerDrip][LocationColumnDrip].nitrate_nitrogen_content += NFertilizer[i].amtnit * ferc * row_space / (dl(LocationLayerDrip) * wk(LocationColumnDrip, row_space))
                    VolUreaNContent[LocationLayerDrip][LocationColumnDrip] += NFertilizer[i].amtura * ferc * row_space / (dl(LocationLayerDrip) * wk(LocationColumnDrip, row_space))

    def water_uptake(self, row_space, per_plant_area):
        """This function computes the uptake of water by plant roots from the soil (i.e., actual transpiration rate)."""
        # Compute the modified light interception factor (LightInter1) for use in computing transpiration rate.
        # modified light interception factor by canopy
        LightInter1 = min(max(self.light_interception * 1.55 - 0.32,  self.light_interception), 1)

        # The potential transpiration is the product of the daytime Penman equation and LightInter1.
        PotentialTranspiration = self.evapotranspiration * LightInter1
        upf = np.zeros((40, 20), dtype=np.float64)  # uptake factor, computed as a ratio, for each soil cell
        uptk = np.zeros((40, 20), dtype=np.float64)  # actual transpiration from each soil cell, cm3 per day
        sumep = 0  # sum of actual transpiration from all soil soil cells, cm3 per day.

        # Compute the reduction due to soil moisture supply by function PsiOnTranspiration().
        # the actual transpiration converted to cm3 per slab units.
        Transp = 0.10 * row_space * PotentialTranspiration * PsiOnTranspiration(AverageSoilPsi)
        while True:
            for l in range(self.soil.number_of_layers_with_root):
                j = SoilHorizonNum[l]
                # Compute, for each layer, the lower and upper water content limits for the transpiration function. These are set from limiting soil water potentials (-15 to -1 bars).
                vh2lo = qpsi(-15, thad[l], thts[l], alpha[j], vanGenuchtenBeta[j])  # lower limit of water content for the transpiration function
                vh2hi = qpsi(-1, thad[l], thts[l], alpha[j], vanGenuchtenBeta[j])  # upper limit of water content for the transpiration function
                for k in range(self.soil._[0].layers[l].number_of_left_columns_with_root, self.soil._[0].layers[l].number_of_right_columns_with_root + 1):
                    # reduction factor for water uptake, caused by low levels of soil water, as a linear function of cell.water_content, between vh2lo and vh2hi.
                    redfac = min(max((self.soil.cells[l][k].water_content - vh2lo) / (vh2hi - vh2lo), 0), 1)
                    # The computed 'uptake factor' (upf) for each soil cell is the product of 'root weight capable of uptake' and redfac.
                    upf[l][k] = self.soil.cells[l][k].root.weight_capable_uptake * redfac

            difupt = 0  # the cumulative difference between computed transpiration and actual transpiration, in cm3, due to limitation of PWP.
            for l in range(self.soil.number_of_layers_with_root):
                for k in range(self.soil._[0].layers[l].number_of_left_columns_with_root, self.soil._[0].layers[l].number_of_right_columns_with_root + 1):
                    if upf[l][k] > 0 and self.soil.cells[l][k].water_content > thetar[l]:
                        # The amount of water extracted from each cell is proportional to its 'uptake factor'.
                        upth2o = Transp * upf[l][k] / upf.sum()  # transpiration from a soil cell, cm3 per day
                        # Update cell.water_content, storing its previous value as vh2ocx.
                        vh2ocx = self.soil.cells[l][k].water_content  # previous value of water_content of this cell
                        self.soil.cells[l][k].water_content -= upth2o / (dl(l) * wk(k, row_space))
                        # If the new value of cell.water_content is less than the permanent wilting point, modify the value of upth2o so that water_content will be equal to it.
                        if self.soil.cells[l][k].water_content < thetar[l]:
                            self.soil.cells[l][k].water_content = thetar[l]

                            # Compute the difference due to this correction and add it to difupt.
                            xupt = (vh2ocx - thetar[l]) * dl(l) * wk(k, row_space)  # intermediate computation of upth2o
                            difupt += upth2o - xupt
                            upth2o = xupt
                        upth2o = max(upth2o, 0)

                        # Compute sumep as the sum of the actual amount of water extracted from all soil cells. Recalculate uptk of this soil cell as cumulative upth2o.
                        sumep += upth2o
                        uptk[l][k] += upth2o

            # If difupt is greater than zero, redefine the variable Transp as difuptfor use in next loop.
            if difupt > 0:
                Transp = difupt
            else:
                break

        # recompute SoilPsi for all soil cells with roots by calling function PSIQ,
        for l in range(self.soil.number_of_layers_with_root):
            j = SoilHorizonNum[l]
            for k in range(self.soil._[0].layers[l].number_of_left_columns_with_root, self.soil._[0].layers[l].number_of_right_columns_with_root + 1):
                SoilPsi[l][k] = (
                    psiq(self.soil.cells[l][k].water_content, thad[l], thts[l], alpha[j], vanGenuchtenBeta[j])
                    - PsiOsmotic(self.soil.cells[l][k].water_content, thts[l], ElCondSatSoilToday)
                )

        # compute ActualTranspiration as actual water transpired, in mm.
        self.actual_transpiration = sumep * 10 / row_space

        # Zeroize the amounts of NH4 and NO3 nitrogen taken up from the soil.
        self.supplied_nitrate_nitrogen = 0
        self.supplied_ammonium_nitrogen = 0

        # Compute the proportional N requirement from each soil cell with roots, and call function NitrogenUptake() to compute nitrogen uptake.
        if sumep > 0 and self.total_required_nitrogen > 0:
            for l in range(self.soil.number_of_layers_with_root):
                for k in range(self.soil._[0].layers[l].number_of_left_columns_with_root, self.soil._[0].layers[l].number_of_right_columns_with_root + 1):
                    if uptk[l][k] > 0:
                        # proportional allocation of TotalRequiredN to each cell
                        reqnc = self.total_required_nitrogen * uptk[l][k] / sumep
                        NitrogenUptake(self._[0], self._[0].soil.cells[l][k], l, k, reqnc, row_space, per_plant_area)

    def average_psi(self, row_space):
        """This function computes and returns the average soil water potential of the root zone of the soil slab. This average is weighted by the amount of active roots (roots capable of uptake) in each soil cell. Soil zones without roots are not included."""
        # Constants used:
        vrcumin = 0.1e-9
        vrcumax = 0.025

        psinum = np.zeros(9, dtype=np.float64)  # sum of weighting coefficients for computing avgwat.
        sumwat = np.zeros(9, dtype=np.float64)  # sum of weighted soil water content for computing avgwat.
        sumdl = np.zeros(9, dtype=np.float64)  # sum of thickness of all soil layers containing roots.
        # Compute sum of dl as sumdl for each soil horizon.
        for l in range(self.soil.number_of_layers_with_root):
            j = SoilHorizonNum[l]
            sumdl[j] += dl(l)
            for k in range(self.soil._[0].layers[l].number_of_left_columns_with_root, self.soil._[0].layers[l].number_of_right_columns_with_root + 1):
                # Check that RootWtCapblUptake in any cell is more than a minimum value vrcumin.
                if self.soil.cells[l][k].root.weight_capable_uptake >= vrcumin:
                    # Compute sumwat as the weighted sum of the water content, and psinum as the sum of these weights. Weighting is by root weight capable of uptake, or if it exceeds a maximum value (vrcumax) this maximum value is used for weighting.
                    sumwat[j] += self.soil.cells[l][k].water_content * dl(l) * wk(k, row_space) * min(self.soil.cells[l][k].root.weight_capable_uptake, vrcumax)
                    psinum[j] += dl(l) * wk(k, row_space) * min(self.soil.cells[l][k].root.weight_capable_uptake, vrcumax)
        sumpsi = 0  # weighted sum of avgpsi
        sumnum = 0  # sum of weighting coefficients for computing AverageSoilPsi.
        for j in range(9):
            if psinum[j] > 0 and sumdl[j] > 0:
                # Compute avgwat and the parameters to compute the soil water potential in each soil horizon
                avgwat = sumwat[j] / psinum[j]  # weighted average soil water content (V/V) in root zone
                # Soil water potential computed for a soil profile layer:
                avgpsi = psiq(avgwat, airdr[j], thetas[j], alpha[j], vanGenuchtenBeta[j]) - PsiOsmotic(avgwat, thetas[j], ElCondSatSoilToday)
                # Use this to compute the average for the whole root zone.
                sumpsi += avgpsi * psinum[j]
                sumnum += psinum[j]
        return sumpsi / sumnum if sumnum > 0 else 0  # average soil water potential for the whole root zone

    def soil_surface_balance(self, int ihr, int k, double ess, double rlzero, double rss, double sf, double hsg, double so, double so2, double so3, double thet, double tv) -> tuple[float, float, float]:
        """This function is called from EnergyBalance(). It calls function ThermalCondSoil().

        It solves the energy balance equations at the soil surface, and computes the resulting temperature of the soil surface.

        Units for all energy fluxes are: cal cm-2 sec-1.

        :param ihr: the time in hours.
        :param k: soil column number.
        :param ess: evaporation from soil surface (mm / sec).
        :param rlzero: incoming long wave
        :param rss: global radiation absorbed by soil surface
        :param sf: fraction of shaded soil area
        :param hsg: multiplier for computing sensible heat transfer from soil to air.
        :param thet: air temperature (K).
        :param tv: temperature of plant canopy (K).
        """
        # Constants:
        cdef double ef = 0.95  # emissivity of the foliage surface
        cdef double eg = 0.95  # emissivity of the soil surface
        cdef double stefa1 = 1.38e-12  # Stefan-Boltsman constant.
        # Long wave radiation reaching the soil:
        cdef double rls1  # long wave energy reaching soil surface
        if sf >= 0.05:  # haded column
            rls1 = (
                (1 - sf) * eg * rlzero  # from sky on unshaded soil surface
                + sf * eg * ef * stefa1 * tv ** 4  # from foliage on shaded soil surface
            )
        else:
            rls1 = eg * rlzero  # from sky in unshaded column
        # rls4 is the multiplier of so**4 for emitted long wave radiation from soil
        cdef double rls4 = eg * stefa1
        cdef double bbex  # previous value of bbadjust.
        cdef double soex = so  # previous value of so.
        # Start itrations for soil surface enegy balance.
        for mon in range(50):
            # Compute latent heat flux from soil evaporation: convert from mm sec-1 to cal cm-2 sec-1. Compute derivative of hlat
            # hlat is the energy used for evaporation from soil surface (latent heat)
            hlat = (75.5255 - 0.05752 * so) * ess
            dhlat = -0.05752 * ess  # derivative of hlat
            # Compute the thermal conductivity of layers 1 to 3 by function ThermalCondSoil().
            # heat conductivity of n-th soil layer in cal / (cm sec deg).
            rosoil1 = ThermalCondSoil(self._[0].soil.cells[0][k].water_content, so, 1)
            rosoil2 = ThermalCondSoil(self._[0].soil.cells[1][k].water_content, so2, 2)
            rosoil3 = ThermalCondSoil(self._[0].soil.cells[2][k].water_content, so3, 3)
            # Compute average rosoil between layers 1 to 3,and heat transfer from soil surface to 3rd soil layer.
            # multiplier for heat flux between 1st and 3rd soil layers.
            rosoil = (rosoil1 * dl(0) + rosoil2 * dl(1) + rosoil3 * dl(2)) / (dl(0) + dl(1) + dl(2)) / (.5 * dl(0) + dl(1) + .5 * dl(2))
            # bbsoil is the heat energy transfer by conductance from soil surface to soil
            bbsoil = rosoil * (so - so3)
            # emtlw is emitted long wave radiation from soil surface
            emtlw = rls4 * pow(so, 4)
            # Sensible heat transfer and its derivative
            # average air temperature above soil surface (K)
            tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)
            senheat = hsg * (so - tafk)  # sensible heat transfer from soil surface
            dsenheat = hsg * (1 - sf * 0.1)  # derivative of senheat
            # Compute the energy balance bb. (positive direction is upward)
            bb = (
                emtlw  # long wave radiation emitted from soil surface
                - rls1  # long wave radiation reaching the soil surface
                + bbsoil  #(b) heat transfer from soil surface to next soil layer
                + hlat  #(c) latent heat transfer
                - rss  # global radiation reaching the soil surface
                + senheat  # (d) heat transfer from soil surface to air
            )

            if abs(bb) < 10e-6:
                return so, so2, so3  # end computation for so
            # If bb is not small enough, compute its derivative by so.

            demtlw = 4 * rls4 * so ** 3 # The derivative of emitted long wave radiation (emtlw)
            # Compute derivative of bbsoil
            sop001 = so + 0.001  # soil surface temperature plus 0.001
            # heat conductivity of 1st soil layer for so+0.001
            rosoil1p = ThermalCondSoil(self._[0].soil.cells[0][k].water_content, sop001, 1)
            # rosoil for so+0.001
            rosoilp = (rosoil1p * dl(0) + rosoil2 * dl(1) + rosoil3 * dl(2)) / (dl(0) + dl(1) + dl(2)) / (.5 * dl(0) + dl(1) + .5 * dl(2))
            drosoil = (rosoilp - rosoil) / 0.001  # derivative of rosoil
            dbbsoil = rosoil + drosoil * (so - so3)  # derivative of bbsoil
            # The derivative of the energy balance function
            bbp = (
                demtlw  # (a)
                + dbbsoil  # (b)
                + dhlat  # (c)
                + dsenheat  # (d)
            )
            # Correct the upper soil temperature by the ratio of bb to bbp.
            # the adjustment of soil surface temperature before next iteration
            bbadjust = bb / bbp
            # If adjustment is small enough, no more iterations are needed.
            if abs(bbadjust) < 0.002:
                return so, so2, so3
            # If bbadjust is not the same sign as bbex, reduce fluctuations
            if mon <= 1:
                bbex = 0
            elif mon >= 2:
                if abs(bbadjust + bbex) < abs(bbadjust - bbex):
                    bbadjust = (bbadjust + bbex) / 2
                    so = (so + soex) / 2

            bbadjust = min(max(bbadjust, -10), 10)

            so -= bbadjust
            so2 += (so - soex) / 2
            so3 += (so - soex) / 3
            soex = so
            bbex = bbadjust
            mon += 1
        else:
            # If (mon >= 50) send message on error and end simulation.
            raise RuntimeError("\n".join((
                "Infinite loop in soil_surface_balance(). Abnormal stop!!",
                "Daynum, ihr, k = %s %3d %3d" % (self.date.isoformat(), ihr, k),
                "so = %10.3g" % so,
                "so2 = %10.3g" % so2,
                "so3 = %10.3g" % so3,
            )))
    #end soil

    #begin climate
    def compute_evapotranspiration(
        self,
        latitude: np.float64,
        elevation: np.float64,
        declination: np.float64,
        tmpisr: np.float64,
        site7: np.float64,
    ):
        """computes the rate of reference evapotranspiration and related variables."""
        stefb: np.float64 = 5.77944E-08  # the Stefan-Boltzman constant, in W m-2 K-4 (= 1.38E-12 * 41880)
        c12: np.float64 = 0.125  # c12 ... c15 are constant parameters.
        c13: np.float64 = 0.0439
        c14: np.float64 = 0.030
        c15: np.float64 = 0.0576
        iamhr = 0  # earliest time in day for computing cloud cover
        ipmhr = 0  # latest time in day for computing cloud cover
        cosz: np.float64 = 0  # cosine of sun angle from zenith for this hour
        suna: np.float64 = 0  # sun angle from horizon, degrees at this hour
        # Start hourly loop
        for ihr, hour in enumerate(self.hours):
            ti = ihr + 0.5  # middle of the hourly interval
            # The following subroutines and functions are called for each hour: sunangle, cloudcov, clcor, refalbed .
            cosz, suna = sunangle(
                ti,
                latitude,
                declination,
                self.solar_noon,
            )
            isr = tmpisr * cosz  # hourly extraterrestrial radiation in W / m**2
            hour.cloud_cov = cloudcov(hour.radiation, isr, cosz)
            # clcor is called to compute cloud-type correction.
            # iamhr and ipmhr are set.
            hour.cloud_cor = clcor(
                ihr,
                site7,
                isr,
                cosz,
                self.day_length,
                hour.radiation,
                self.solar_noon,
            )
            if cosz >= 0.1736 and iamhr == 0:
                iamhr = ihr
            if ihr >= 12 and cosz <= 0.1736 and ipmhr == 0:
                ipmhr = ihr - 1
            # refalbed is called to compute the reference albedo for each hour.
            hour.albedo = refalbed(isr, hour.radiation, cosz, suna)
        # Zero some variables that will later be used for summation.
        self.evapotranspiration = 0
        self.net_radiation = 0  # daily net radiation
        for ihr, hour in enumerate(self.hours):
            # Compute saturated vapor pressure (svp), using function VaporPressure().
            # The actual vapor pressure (vp) is computed from svp and the relative humidity. Compute vapor pressure deficit (vpd). This procedure is based on the CIMIS algorithm.
            svp = VaporPressure(hour.temperature)  # saturated vapor pressure, mb
            vp = 0.01 * hour.humidity * svp  # vapor pressure, mb
            vpd = svp - vp  # vapor pressure deficit, mb.
            # Get cloud cover and cloud correction for night hours
            if ihr < iamhr or ihr > ipmhr:
                hour.cloud_cov = 0
                hour.cloud_cor = 0
            # The hourly net radiation is computed using the CIMIS algorithm (Dong et al., 1988):
            # rlonin, the hourly incoming long wave radiation, is computed from ea0, cloud cover (CloudCoverRatio), air temperature (tk),  stefb, and cloud type correction (CloudTypeCorr).
            # rnet, the hourly net radiation, W m-2, is computed from the global radiation, the albedo, the incoming long wave radiation, and the outgoing longwave radiation.
            tk = hour.temperature + 273.161  # hourly air temperature in Kelvin.
            ea0 = clearskyemiss(vp, tk)  # clear sky emissivity for long wave radiation
            # Compute incoming long wave radiation:
            rlonin = (ea0 * (1 - hour.cloud_cov) + hour.cloud_cov) * stefb * tk ** 4 - hour.cloud_cor
            rnet = (1 - hour.albedo) * hour.radiation + rlonin - stefb * tk ** 4
            self.net_radiation += rnet
            # The hourly reference evapotranspiration ReferenceETP is computed by the CIMIS algorithm using the modified Penman equation:
            # The weighting ratio (w) is computed from the functions del() (the slope of the saturation vapor pressure versus air temperature) and gam() (the psychometric constant).
            w = delta(tk, svp) / (delta(tk, svp) + gamma(elevation, hour.temperature))  # coefficient of the Penman equation

            # The wind function (fu2) is computed using different sets of parameters for day-time and night-time. The parameter values are as suggested by CIMIS.
            fu2 = (  # wind function for computing evapotranspiration
                c12 + c13 * hour.wind_speed
                if hour.radiation <= 0 else
                c14 + c15 * hour.wind_speed
            )

            # hlathr, the latent heat for evaporation of water (W m-2 per mm at this hour) is computed as a function of temperature.
            hlathr = 878.61 - 0.66915 * (hour.temperature + 273.161)
            # ReferenceETP, the hourly reference evapotranspiration, is now computed by the modified Penman equation.
            hour.ref_et = w * rnet / hlathr + (1 - w) * vpd * fu2
            if hour.ref_et < 0:
                hour.ref_et = 0
            # ReferenceTransp is the sum of ReferenceETP
            self.evapotranspiration += hour.ref_et
            # es1hour and es2hour are computed as the hourly potential evapotranspiration due to radiative and aerodynamic factors, respectively.
            # es1hour and ReferenceTransp are not computed for periods of negative net radiation.
            hour.et2 = (1 - w) * vpd * fu2
            hour.et1 = max(w * rnet / hlathr, 0)

    def calculate_average_temperatures(self):
        self.average_temperature = 0
        self.daytime_temperature = 0
        self.nighttime_temperature = 0
        night_hours = 0
        for hour in self.hours:
            if hour.radiation <= 0:
                night_hours += 1
                self.nighttime_temperature += hour.temperature
            else:
                self.daytime_temperature += hour.temperature
            self.average_temperature += hour.temperature
        if night_hours == 0 or night_hours == 24:
            raise RuntimeError("Plant cotton in polar region?")
        self.average_temperature /= 24
        self.nighttime_temperature /= night_hours
        self.daytime_temperature /= (24 - night_hours)

    def initialize_soil_data(self):
        """Computes and sets the initial soil data. It is executed once at the beginning of the simulation, after the soil hydraulic data file has been read. It is called by ReadInput()."""
        cdef int j = 0  # horizon number
        cdef double sumdl = 0  # depth to the bottom this layer (cm);
        cdef double rm = 2.65  # density of the solid fraction of the soil (g / cm3)
        cdef double bdl[40]  # array of bulk density of soil layers
        for l in range(40):
            # Using the depth of each horizon layer, the horizon number (SoilHorizonNum) is computed for each soil layer.
            sumdl += dl(l)
            for j, layer_depth in enumerate(SOIL["depth"]):
                if sumdl <= layer_depth:
                    break
            SoilHorizonNum[l] = j
            # bdl, thad, thts are defined for each soil layer, using the respective input variables BulkDensity, airdr, thetas.
            # FieldCapacity, MaxWaterCapacity and thetar are computed for each layer, as water content (cm3 cm-3) of each layer corresponding to matric potentials of psisfc (for field capacity), psidra (for free drainage) and -15 bars (for permanent wilting point), respectively, using function qpsi.
            # pore space volume (PoreSpace) is also computed for each layer.
            # make sure that saturated water content is not more than pore space.
            bdl[l] = BulkDensity[j]
            PoreSpace[l] = 1 - BulkDensity[j] / rm
            if thetas[j] > PoreSpace[l]:
                thetas[j] = PoreSpace[l]
            thad[l] = airdr[j]
            thts[l] = thetas[j]
            FieldCapacity[l] = qpsi(psisfc, thad[l], thts[l], alpha[j], vanGenuchtenBeta[j])
            MaxWaterCapacity[l] = qpsi(psidra, thad[l], thts[l], alpha[j], vanGenuchtenBeta[j])
            thetar[l] = qpsi(-15., thad[l], thts[l], alpha[j], vanGenuchtenBeta[j])
            # When the saturated hydraulic conductivity (SaturatedHydCond) is not given, it is computed from the hydraulic conductivity at field capacity (condfc), using the wcond function.
            if SaturatedHydCond[j] <= 0:
                SaturatedHydCond[j] = condfc[j] / wcond(FieldCapacity[l], thad[l], thts[l], vanGenuchtenBeta[j], 1, 1)
        # Loop for all soil layers. Compute depth from soil surface to the end of each layer (sumdl).
        sumdl = 0
        for l in range(40):
            sumdl += dl(l)
            # At start of simulation compute estimated movable fraction of nitrates in each soil layer, following the work of:
            # Bowen, W.T., Jones, J.W., Carsky, R.J., and Quintana, J.O. 1993. Evaluation of the nitrogen submodel of CERES-maize following legume green manure incorporation. Agron. J. 85:153-159.
            # The fraction of total nitrate in a layer that is in solution and can move from one layer to the next with the downward flow of water, FLOWNO3[l], is a function of the adsorption coefficient, soil bulk density, and the volumetric soil water content at the drained upper limit.
            # Adsorption coefficients are assumed to be 0.0 up to 30 cm depth, and deeper than 30 cm - 0.2, 0.4, 0.8, 1.0, 1.2, and 1.6 for each successive 15 cm layer.
            coeff: float  # Adsorption coefficient
            if sumdl <= 30:
                coeff = 0
            elif sumdl <= 45:
                coeff = 0.2
            elif sumdl <= 60:
                coeff = 0.4
            elif sumdl <= 75:
                coeff = 0.6
            elif sumdl <= 90:
                coeff = 0.8
            elif sumdl <= 105:
                coeff = 1.0
            elif sumdl <= 120:
                coeff = 1.2
            else:
                coeff = 1.6
            NO3FlowFraction[l] = 1 / (1 + coeff * bdl[l] / MaxWaterCapacity[l])
            # Determine the corresponding 15 cm layer of the input file.
            # Compute the initial volumetric water content (cell.water_content) of each layer, and check that it will not be less than the air-dry value or more than pore space volume.
            j = int((sumdl - 1) / LayerDepth)
            if j > 13:
                j = 13
            n = SoilHorizonNum[l]
            self.soil.cells[l][0].water_content = FieldCapacity[l] * h2oint[j] / 100
            if self.soil.cells[l][0].water_content < airdr[n]:
                self.soil.cells[l][0].water_content = airdr[n]
            if self.soil.cells[l][0].water_content > PoreSpace[l]:
                self.soil.cells[l][0].water_content = PoreSpace[l]
            # Initial values of ammonium N (rnnh4, VolNh4NContent) and nitrate N (rnno3, VolNo3NContent) are converted from kgs per ha to mg / cm3 for each soil layer, after checking for minimal amounts.
            if rnno3[j] < 2.0:
                rnno3[j] = 2.0
            if rnnh4[j] < 0.2:
                rnnh4[j] = 0.2
            self.soil.cells[l][0].nitrate_nitrogen_content = rnno3[j] / LayerDepth * 0.01
            VolNh4NContent[l][0] = rnnh4[j] / LayerDepth * 0.01
            # organic matter in mg / cm3 units.
            om = (oma[j] / 100) * bdl[l] * 1000
            # potom is the proportion of readily mineralizable om. it is a function of soil depth (sumdl, in cm), modified from GOSSYM (where it probably includes the 0.4 factor for organic C in om).
            potom = max(0.0, 0.15125 - 0.02878 * log(sumdl))
            # FreshOrganicMatter is the readily mineralizable organic matter (= "fresh organic matter" in CERES models). HumusOrganicMatter is the remaining organic matter, which is mineralized very slowly.
            self.soil.cells[l][0].fresh_organic_matter = om * potom
            HumusOrganicMatter[l][0] = om * (1 - potom)
        # Since the initial value has been set for the first column only in each layer, these values are now assigned to all the other columns.
        for l in range(40):
            VolUreaNContent[l][0] = 0
            for k in range(1, 20):
                self.soil.cells[l][k].water_content = self.soil.cells[l][0].water_content
                self.soil.cells[l][k].nitrate_nitrogen_content = self.soil.cells[l][0].nitrate_nitrogen_content
                VolNh4NContent[l][k] = VolNh4NContent[l][0]
                self.soil.cells[l][k].fresh_organic_matter = self.soil.cells[l][0].fresh_organic_matter
                HumusOrganicMatter[l][k] = HumusOrganicMatter[l][0]
                VolUreaNContent[l][k] = 0


def SensibleHeatTransfer(tsf, tenviron, height, wndcanp) -> float:
    """This function computes the sensible heat transfer coefficient, using the friction potential (shear) temperature (thstar), and the surface friction (shear) velocity (ustar) at the atmospheric boundary. It is called three times from EnergyBalance(): for canopy or soil surface with their environment.

    Parameters
    ----------
    tenviron
        temperature (K) of the environment - air at 200 cm height for columns with no canopy, or tafk when canopy is present .
    tsf
        surface temperature, K, of soil or canopy.
    wndcanp
        wind speed in the canopy (if present), cm s-1.
    height
        canopy height, cm, or zero for soil surface.

    Returns
    -------
    float
        raw sensible heat transfer coefficient
    """
    # Constant values used:
    grav = 980  # acceleration due to gravity (980 cm sec-2).
    s40 = 0.13  # calibration constant.
    s42 = 0.63  # calibration constant.
    stmin = 5  # minimal value of ustar.
    vonkar = 0.40  # Von-Karman constant (0.40).
    zalit1 = 0.0962  # parameter .....
    # Wind velocity not allowed to be less than 100 cm s-1.
    cdef double u = max(wndcanp, 100)  # wind speed at 200 cm height, cm / s.
    # Assign initial values to z0 and gtop, and set dt.
    cdef double z0 = max(s40 * height, 1)  # surface roughness parameter, cm.
    cdef double gtop = log((200 - s42 * height) / z0)  # logarithm of ratio of height of measurement to surface roughness parameter.
    cdef double dt = tsf - tenviron  # temperature difference.
    # Set approximate initial values for ustar and thstar (to reduce iterations).
    cdef double thstar  # friction potential (shear) temperature.
    cdef double ustar  # Surface friction (shear) velocity (cm sec-1).
    if dt >= 0:
        ustar = 1.873 + 0.570172 * dt + .07438568 * u
        thstar = -0.05573 * dt + 1.987 / u - 6.657 * dt / u
    else:
        ustar = max(-4.4017 + 1.067 * dt + 0.25957 * u - 0.001683 * dt * u, 5)
        thstar = max(-0.0096 - 0.1149 * dt + 0.0000377 * u + 0.0002367 * dt * u, 0.03)
    cdef double tbot1 = tsf  # surface temperature corrected for friction (shear) potential temperature.
    cdef double g1  # temporary derived variable
    cdef double ug1chk  # previous value of ug1.
    cdef double ug1  # ratio of ustar to g1.
    cdef double ug1res  # residual value of ug1.
    # Start iterations.
    for mtest in range(100):
        ug1chk = 0  # previous value of UG1.
        if mtest > 0:
            # Assign values to tbot1, uchek, thekz, and ug1chk.
            tbot1 = tsf + zalit1 * thstar * pow((ustar * z0 / 15), 0.45) / vonkar
            uchek = ustar  # previous value of ustar.
            thekz = thstar  # previous value of thstar.
            if g1 != 0:
                ug1chk = ustar / g1
        # Compute air temperature at 1 cm,, and compute zl and lstar.
        # nondimensional height.
        if abs(thstar) < 1e-30:
            zl = 0
        else:
            thetmn = (tenviron + tbot1) * 0.5  # mean temperature (K) of air and surface.
            lstar = (thetmn * ustar * ustar) / (vonkar * grav * thstar)
            zl = min(max((200 - s42 * height) / lstar, -5), 0.5)
        # Compute g1u, and g2 temporary derived variables.
        if zl > 0:
            g1u = -4.7 * zl
            g2 = max(-6.35135 * zl, -1)
        else:
            tmp1 = pow((1 - 15 * zl), 0.25)  # intermediate variable.
            g1u = 2 * log((1 + tmp1) / 2) + log((1 + tmp1 * tmp1) / 2) - 2 * atan(tmp1 + 1.5708)
            g2 = 2 * log((1 + sqrt(1 - 9 * zl)) / 2)
        g2 = min(g2, gtop)
        # Compute ustar and check for minimum value.
        ustar = max(vonkar * u / (gtop - g1u), stmin)
        # Compute g1 and thstar.
        g1 = 0.74 * (gtop - g2) + zalit1 * pow((ustar * z0 / 0.15), 0.45)
        thstar = -dt * vonkar / g1
        # If more than 30 iterations, reduce fluctuations
        if mtest > 30:
            thstar = (thstar + thekz) / 2
            ustar = (ustar + uchek) / 2

        # Compute ug1 and  ug1res to check convergence
        ug1 = ustar / g1
        if abs(ug1chk) <= 1.e-30:
            ug1res = abs(ug1)
        else:
            ug1res = abs((ug1chk - ug1) / ug1chk)
        # If ug1 did not converge, go to next iteration.
        if abs(ug1 - ug1chk) <= 0.05 or ug1res <= 0.01:
            return ustar * vonkar / g1
    else:
        # Stop simulation if no convergence after 100 iterations.
        raise RuntimeError


cdef class Simulation:
    cdef cSimulation _sim
    cdef public unsigned int year
    cdef public unsigned int profile_id
    cdef public unsigned int version
    cdef public double latitude
    cdef public double longitude
    cdef public double elevation  # meter
    cdef uint32_t _emerge_day
    cdef uint32_t _start_day
    cdef uint32_t _stop_day
    cdef uint32_t _plant_day
    cdef uint32_t _topping_day
    cdef uint32_t _defoliate_day
    cdef uint32_t _first_bloom_day
    cdef uint32_t _first_square_day
    cdef public uint32_t plant_row_column  # column number to the left of plant row location.
    cdef public double max_leaf_area_index
    cdef public double ptsred  # The effect of moisture stress on the photosynthetic rate
    cdef public double density_factor  # empirical plant density factor.
    cdef double DaysTo1stSqare   # number of days from emergence to 1st square
    cdef double defkgh  # amount of defoliant applied, kg per ha
    cdef double tdfkgh  # total cumulative amount of defoliant
    cdef bool_t idsw  # switch indicating if predicted defoliation date was defined.
    cdef public double skip_row_width  # the smaller distance between skip rows, cm
    cdef public double plants_per_meter  # average number of plants pre meter of row.
    cdef public double per_plant_area  # average soil surface area per plant, dm2
    # switch affecting the method of computing soil temperature.
    # 0 = one dimensional (no horizontal flux) - used to predict emergence when emergence date is not known;
    # 1 = one dimensional - used before emergence when emergence date is given;
    # 2 = two dimensional - used after emergence.
    cdef public unsigned int emerge_switch
    cdef public State _current_state
    relative_radiation_received_by_a_soil_column = np.ones(20)  # the relative radiation received by a soil column, as affected by shading by plant canopy.

    def __init__(self, profile_id=0, version=0x0400, **kwargs):
        self.profile_id = profile_id
        self.version = version
        self.max_leaf_area_index = 0.001
        for attr in (
            "start_date",
            "stop_date",
            "emerge_date",
            "plant_date",
            "topping_date",
            "latitude",
            "longitude",
            "elevation",
            "site_parameters",
            "cultivar_parameters",
            "row_space",
            "skip_row_width",
            "plants_per_meter",
        ):
            if attr in kwargs:
                setattr(self, attr, kwargs.get(attr))

    @property
    def start_date(self):
        return date.fromordinal(self._start_day)

    @start_date.setter
    def start_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._start_day = d.toordinal()

    @property
    def stop_date(self):
        return date.fromordinal(self._stop_day)

    @stop_date.setter
    def stop_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._stop_day = d.toordinal()

    @property
    def emerge_date(self):
        return date.fromordinal(self._emerge_day) if self._emerge_day else None

    @emerge_date.setter
    def emerge_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._emerge_day = d.toordinal()

    @property
    def plant_date(self):
        return date.fromordinal(self._plant_day) if self._plant_day else None

    @plant_date.setter
    def plant_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._plant_day = d.toordinal()

    @property
    def topping_date(self):
        if self.version >= 0x500 and self._topping_day > 0:
            return date.fromordinal(self._topping_day)
        return None

    @topping_date.setter
    def topping_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._topping_day = d.toordinal()

    @property
    def site_parameters(self):
        return SitePar

    @site_parameters.setter
    def site_parameters(self, parameters):
        for i, p in enumerate(parameters):
            SitePar[i + 1] = p

    @property
    def cultivar_parameters(self):
        return self._sim.cultivar_parameters

    @cultivar_parameters.setter
    def cultivar_parameters(self, parameters):
        for i, p in enumerate(parameters):
            self._sim.cultivar_parameters[i + 1] = p

    @property
    def row_space(self):
        return self._sim.row_space

    @row_space.setter
    def row_space(self, value):
        self._sim.row_space = value or 0

    @property
    def plant_population(self):
        return self._sim.plant_population

    @plant_population.setter
    def plant_population(self, value):
        self._sim.plant_population = value

    @property
    def first_square_date(self):
        return date.fromordinal(self._first_square_day) if self._first_square_day else None

    @first_square_date.setter
    def first_square_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._first_square_day = d.toordinal()

    @property
    def first_bloom_date(self):
        return date.fromordinal(self._first_bloom_day) if self._first_bloom_day else None

    @first_bloom_date.setter
    def first_bloom_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._first_bloom_day = d.toordinal()

    @property
    def defoliate_date(self):
        return date.fromordinal(self._defoliate_day) if self._defoliate_day else None

    @defoliate_date.setter
    def defoliate_date(self, d):
        if not isinstance(d, date):
            d = date.fromisoformat(d)
        self._defoliate_day = d.toordinal()

    cpdef State _state(self, unsigned int i):
        return State.from_ptr(&self._sim.states[i], self, self.version)

    @property
    def climate(self):
        return self._sim.climate

    @climate.setter
    def climate(self, climate):
        alias = {
            "radiation": "Rad",
            "max": "Tmax",
            "min": "Tmin",
            "wind": "Wind",
            "rain": "Rain",
            "dewpoint": "Tdew",
        }
        for i, daily_climate in enumerate(climate):
            self._sim.climate[i] = {
                alias[k]: v for k, v in daily_climate.items()
            }

    def _init_state(self):
        cdef State state0 = self._current_state
        state0.date = self.start_date
        state0.lint_yield = 0
        state0.soil.number_of_layers_with_root = 7
        state0.plant_height = 4.0
        state0.stem_weight = 0.2
        state0.petiole_weight = 0
        state0.square_weight = 0
        state0.green_bolls_weight = 0
        state0.green_bolls_burr_weight = 0
        state0.open_bolls_weight = 0
        state0.open_bolls_burr_weight = 0
        state0.reserve_carbohydrate = 0.06
        state0.cumulative_nitrogen_loss = 0
        state0.water_stress = 1
        state0.water_stress_stem = 1
        state0.carbon_stress = 1
        state0.extra_carbon = 0
        state0.leaf_area_index = 0.001
        state0.leaf_area = 0
        state0.leaf_weight = 0.20
        state0.leaf_nitrogen = 0.0112
        state0.number_of_vegetative_branches = 1
        state0.number_of_squares = 0
        state0.number_of_green_bolls = 0
        state0.number_of_open_bolls = 0
        state0.fiber_length = 0
        state0.fiber_strength = 0
        state0.nitrogen_stress = 1
        state0.nitrogen_stress_vegetative = 1
        state0.nitrogen_stress_fruiting = 1
        state0.nitrogen_stress_root = 1
        state0.total_required_nitrogen = 0
        state0.leaf_nitrogen_concentration = .056
        state0.petiole_nitrogen_concentration = 0
        state0.seed_nitrogen_concentration = 0
        state0.burr_nitrogen_concentration = 0
        state0.burr_nitrogen = 0
        state0.seed_nitrogen = 0
        state0.root_nitrogen_concentration = .026
        state0.root_nitrogen = 0.0052
        state0.square_nitrogen_concentration = 0
        state0.square_nitrogen = 0
        state0.stem_nitrogen_concentration = 0.036
        state0.stem_nitrogen = 0.0072
        state0.fruit_growth_ratio = 1
        state0.ginning_percent = 0.35
        state0.number_of_pre_fruiting_nodes = 1
        state0.total_actual_leaf_growth = 0
        state0.total_actual_petiole_growth = 0
        state0.carbon_allocated_for_root_growth = 0
        state0.supplied_ammonium_nitrogen = 0
        state0.supplied_nitrate_nitrogen = 0
        state0.petiole_nitrate_nitrogen_concentration = 0
        for i in range(9):
            state0.age_of_pre_fruiting_nodes[i] = 0
            state0.leaf_area_pre_fruiting[i] = 0
            state0.leaf_weight_pre_fruiting[i] = 0
        for k in range(3):
            state0._[0].vegetative_branches[k].number_of_fruiting_branches = 0
            for l in range(30):
                state0._[0].vegetative_branches[k].fruiting_branches[
                    l].number_of_fruiting_nodes = 0
                state0._[0].vegetative_branches[k].fruiting_branches[l].delay_for_new_node = 0
                state0._[0].vegetative_branches[k].fruiting_branches[l].main_stem_leaf = dict(
                    leaf_area=0,
                    leaf_weight=0,
                    petiole_weight=0,
                    potential_growth_for_leaf_area=0,
                    potential_growth_for_leaf_weight=0,
                    potential_growth_for_petiole_weight=0,
                )
                for m in range(5):
                    state0._[0].vegetative_branches[k].fruiting_branches[l].nodes[m] = dict(
                        age=0,
                        fraction=0,
                        average_temperature=0,
                        ginning_percent=0.35,
                        stage=Stage.NotYetFormed,
                        leaf=dict(
                            age=0,
                            potential_growth=0,
                            area=0,
                            weight=0,
                        ),
                        square=dict(
                            potential_growth=0,
                            weight=0,
                        ),
                        boll=dict(
                            age=0,
                            potential_growth=0,
                            weight=0,
                            cumulative_temperature=0,
                        ),
                        burr=dict(
                            potential_growth=0,
                            weight=0,
                        ),
                        petiole=dict(
                            potential_growth=0,
                            weight=0,
                        ),
                    )

    def _initialize_root_data(self):
        """ This function initializes the root submodel parameters and variables."""
        state0 = self._current_state
        # The parameters of the root model are defined for each root class:
        # grind(i), cuind(i), thtrn(i), trn(i), thdth(i), dth(i).
        cdef double rlint = 10  # Vertical interval, in cm, along the taproot, for initiating lateral roots.
        cdef int ll = 1  # Counter for layers with lateral roots.
        cdef double sumdl = 0  # Distance from soil surface to the middle of a soil layer.
        for l in range(nl):
            # Using the value of rlint (interval between lateral roots), the layers from which lateral roots may be initiated are now computed.
            # LateralRootFlag[l] is assigned a value of 1 for these layers.
            LateralRootFlag[l] = 0
            if l > 0:
                sumdl += 0.5 * dl(l - 1)
            sumdl += 0.5 * dl(l)
            if sumdl >= ll * rlint:
                LateralRootFlag[l] = 1
                ll += 1
        # All the state variables of the root system are initialized to zero.
        for l in range(nl):
            if l < 3:
                state0._[0].soil.layers[l].number_of_left_columns_with_root = self.plant_row_column - 1
                state0._[0].soil.layers[l].number_of_right_columns_with_root = self.plant_row_column + 2
            elif l < 7:
                state0._[0].soil.layers[l].number_of_left_columns_with_root = self.plant_row_column
                state0._[0].soil.layers[l].number_of_right_columns_with_root = self.plant_row_column + 1
            else:
                state0._[0].soil.layers[l].number_of_left_columns_with_root = 0
                state0._[0].soil.layers[l].number_of_right_columns_with_root = 0

        state0.init_root_data(self.plant_row_column, 0.01 * self.row_space / self.per_plant_area)
        # Start loop for all soil layers containing roots.
        self.last_layer_with_root_depth = 0
        for l in range(7):
            self.last_layer_with_root_depth += dl(l)  # compute total depth to the last layer with roots (self.last_layer_with_root_depth).
        # Initial value of taproot length, taproot_length, is computed to the middle of the last layer with roots. The last soil layer with taproot, state.taproot_layer_number, is defined.
        state0.taproot_length = (self.last_layer_with_root_depth - 0.5 * dl(6))
        state0.taproot_layer_number = 6


    def _copy_state(self, i):
        cdef cState state = self._sim.states[i]
        self._sim.states[i + 1] = state

    def _energy_balance(self, u, ihr, k, ess, etp1):
        """
        This function solves the energy balance equations at the soil surface, and at the foliage / atmosphere interface. It computes the resulting temperatures of the soil surface and the plant canopy.

        Units for all energy fluxes are: cal cm-2 sec-1.
        It is called from SoilTemperature(), on each hourly time step and for each soil column.
        It calls functions clearskyemiss(), VaporPressure(), SensibleHeatTransfer(), soil_surface_balance() and canopy_balance()

        :param ihr: the time of day in hours.
        :param k: soil column number.
        :param ess: evaporation from surface of a soil column (mm / sec).
        :param etp1: actual transpiration rate (mm / sec).
        :param sf: fraction of shaded soil area
        """
        state = self._current_state
        hour = state.hours[ihr]
        # Constants used:
        cdef double wndfac = 0.60  # Ratio of wind speed under partial canopy cover.
        cdef double cswint = 0.75  # proportion of short wave radiation (on fully shaded soil surface) intercepted by the canopy.
        # Set initial values
        cdef double sf = 1 - self.relative_radiation_received_by_a_soil_column[k]
        cdef double thet = hour.temperature + 273.161  # air temperature, K
        cdef double so = SoilTemp[0][k]  # soil surface temperature, K
        cdef double so2 = SoilTemp[1][k]  # 2nd soil layer temperature, K
        cdef double so3 = SoilTemp[2][k]  # 3rd soil layer temperature, K
        # Compute soil surface albedo (based on Horton and Chung, 1991):
        ag = compute_soil_surface_albedo(state.soil.cells[0][k].water_content, FieldCapacity[0], thad[0], self.site_parameters[15], self.site_parameters[16])

        rzero, rss, rsup = compute_incoming_short_wave_radiation(hour.radiation, sf * cswint, ag)
        rlzero = compute_incoming_long_wave_radiation(hour.humidity, hour.temperature, hour.cloud_cov, hour.cloud_cor)

        # Set initial values of canopy temperature and air temperature in canopy.
        cdef double tv  # temperature of plant foliage (K)
        cdef double tafk  # temperature (K) of air inside the canopy.
        if sf < 0.05:  # no vegetation
            tv = thet
            tafk = thet
        # Wind velocity in canopy is converted to cm / s.
        cdef double wndhr  # wind speed in cm /sec
        wndhr = hour.wind_speed * 100
        cdef double rocp  # air density * specific heat at constant pressure = 0.24 * 2 * 1013 / 5740
        # divided by tafk.
        cdef double c2  # multiplier for sensible heat transfer (at plant surface).
        cdef double rsv  # global radiation absorbed by the vegetation
        if sf >= 0.05:  # a shaded soil column
            tv = FoliageTemp[k]  # vegetation temperature
            # Short wave radiation intercepted by the canopy:
            rsv = (
                    rzero * (1 - hour.albedo) * sf * cswint  # from above
                    + rsup * (1 - hour.albedo) * sf * cswint  # reflected from soil surface
            )
            # Air temperature inside canopy is the average of soil, air, and plant temperatures, weighted by 0.1, 0.3, and 0.6, respectively.
            tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)

            # Call SensibleHeatTransfer() to compute sensible heat transfer coefficient. Factor 2.2 for sensible heat transfer: 2 sides of leaf plus stems and petioles.
            # sensible heat transfer coefficient for soil
            varcc = SensibleHeatTransfer(tv, tafk, state.plant_height, wndhr)  # canopy to air
            rocp = 0.08471 / tafk
            c2 = 2.2 * sf * rocp * varcc
        cdef double soold = so  # previous value of soil surface temperature
        cdef double tvold = tv  # previous value of vegetation temperature
        # Starting iterations for soil and canopy energy balance
        for menit in range(30):
            soold = so
            wndcanp = (1 - sf * (1 - wndfac)) * wndhr  # estimated wind speed under canopy
            # Call SensibleHeatTransfer() to compute sensible heat transfer for soil surface to air
            tafk = (1 - sf) * thet + sf * (0.1 * so + 0.3 * thet + 0.6 * tv)
            # sensible heat transfer coefficientS for soil
            varc = SensibleHeatTransfer(so, tafk, 0, wndcanp)
            rocp = 0.08471 / tafk
            hsg = rocp * varc  # multiplier for computing sensible heat transfer soil to air.
            # Call soil_surface_balance() for energy balance in soil surface / air interface.
            so, so2, so3 = state.soil_surface_balance(ihr, k, ess, rlzero, rss, sf, hsg, so, so2, so3, thet, tv)

            if sf >= 0.05:
                # This section executed for shaded columns only.
                tvold = tv
                # Compute canopy energy balance for shaded columns
                tv = canopy_balance(etp1, rlzero, rsv, c2, sf, so, thet, tv)
                if menit >= 10:
                    # The following is used to reduce fluctuations.
                    so = (so + soold) / 2
                    tv = (tv + tvold) / 2
            if abs(tv - tvold) <= 0.05 and abs(so - soold) <= 0.05:
                break
        else:
            raise RuntimeError  # If more than 30 iterations are needed - stop simulation.
        # After convergence - set global variables for the following temperatures:
        if sf >= 0.05:
            FoliageTemp[k] = tv
        SoilTemp[0][k] = so
        SoilTemp[1][k] = so2
        SoilTemp[2][k] = so3


    def _soil_temperature_init(self):
        """This function is called from SoilTemperature() at the start of the simulation. It sets initial values to soil and canopy temperatures."""
        # Compute initial values of soil temperature: It is assumed that at the start of simulation
        # the temperature of the first soil layer (upper boundary) is equal to the average air temperature
        # of the previous five days (if climate data not available - start from first climate data).
        # NOTE: For a good simulation of soil temperature, it is recommended to start simulation at
        # least 10 days before planting date. This means that climate data should be available for
        # this period. This is especially important if emergence date has to be simulated.
        state0 = self._current_state
        idd = 0  # number of days minus 4 from start of simulation.
        tsi1 = 0  # Upper boundary (surface layer) initial soil temperature, C.
        for i in range(5):
            tsi1 += self.climate[i]["Tmax"] + self.climate[i]["Tmin"]
        tsi1 /= 10
        # The temperature of the last soil layer (lower boundary) is computed as a sinusoidal function of day of year, with site-specific parameters.
        state0.deep_soil_temperature = SitePar[9] + SitePar[10] * sin(2 * pi * (self.start_date.timetuple().tm_yday - SitePar[11]) / 365) + 273.161
        # SoilTemp is assigned to all columns, converted to degrees K.
        tsi1 += 273.161
        for l in range(40):
            # The temperatures of the other soil layers are linearly interpolated.
            # tsi = computed initial soil temperature, C, for each layer
            tsi = ((40 - l - 1) * tsi1 + l * state0.deep_soil_temperature) / (40 - 1)
            for k in range(20):
                SoilTemp[l][k] = tsi

    def _soil_temperature(self, u):
        """
        This is the main part of the soil temperature sub-model.
        It is called daily from self._simulate_this_day.
        It calls the following functions:
        _energy_balance(), predict_emergence(), SoilHeatFlux().

        References:

        Benjamin, J.G., Ghaffarzadeh, M.R. and Cruse, R.M. 1990. Coupled water and heat transport in ridged soils. Soil Sci. Soc. Am. J. 54:963-969.

        Chen, J. 1984. Uncoupled multi-layer model for the transfer of sensible and latent heat flux densities from vegetation. Boundary-Layer Meteorology 28:213-225.

        Chen, J. 1985. A graphical extrapolation method to determine canopy resistance from measured temperature and humidity profiles above a crop canopy. Agric. For. Meteorol. 37:75-88.

        Clothier, B.E., Clawson, K.L., Pinter, P.J.Jr., Moran, M.S., Reginato, R.J. and Jackson, R.D. 1986. Estimation of soil heat flux from net radiation during the growth of alfalfa. Agric. For. Meteorol. 37:319-329.

        Costello, T.A. and Braud, H.J. Jr. 1989. Thermal diffusivity of soil by nonlinear regression analysis of soil temperature data. Trans. ASAE 32:1281-1286.

        De Vries, D.A. 1963. Thermal properties of soils. In: W.R. Van Wijk (ed) Physics of plant environment, North Holland, Amsterdam, pp 210-235.

        Deardorff, J.W. 1978. Efficient prediction of ground surface temperature and moisture with inclusion of a layer of vegetation. J. Geophys. Res. 83 (C4):1889-1903.

        Dong, A., Prashar, C.K. and Grattan, S.R. 1988. Estimation of daily and hourly net radiation. CIMIS Final Report June 1988, pp. 58-79.

        Ephrath, J.E., Goudriaan, J. and Marani, A. 1996. Modelling diurnal patterns of air temperature, radiation, wind speed and relative humidity by equations from daily characteristics. Agricultural Systems 51:377-393.

        Hadas, A. 1974. Problem involved in measuring the soil thermal conductivity and diffusivity in a moist soil. Agric. Meteorol. 13:105-113.

        Hadas, A. 1977. Evaluation of theoretically predicted thermal conductivities of soils under field and laboratory conditions. Soil Sci. Soc. Am. J. 41:460-466.

        Hanks, R.J., Austin, D.D. and Ondrechen, W.T. 1971. Soil temperature estimation by a numerical method. Soil Sci. Soc. Am. Proc. 35:665-667.

        Hares, M.A. and Novak, M.D. 1992. Simulation of surface energy balance and soil temperature under strip tillage: I. Model description. Soil Sci. Soc. Am. J. 56:22-29.

        Hares, M.A. and Novak, M.D. 1992. Simulation of surface energy balance and soil temperature under strip tillage: II. Field test. Soil Sci. Soc. Am. J. 56:29-36.

        Horton, E. and Wierenga, P.J. 1983. Estimating the soil heat flux from observations of soil temperature near the surface. Soil Sci. Soc. Am. J. 47:14-20.

        Horton, E., Wierenga, P.J. and Nielsen, D.R. 1983. Evaluation of methods for determining apparent thermal diffusivity of soil near the surface. Soil Sci. Soc. Am. J. 47:25-32.

        Horton, R. 1989. Canopy shading effects on soil heat and water flow. Soil Sci. Soc. Am. J. 53:669-679.

        Horton, R., and Chung, S-O, 1991. Soil Heat Flow. Ch. 17 in: Hanks, J., and Ritchie, J.T., (Eds.) Modeling Plant and Soil Systems. Am. Soc. Agron., Madison, WI, pp 397-438.

        Iqbal, M. 1983. An Introduction to Solar Radiation. Academic Press.

        Kimball, B.A., Jackson, R.D., Reginato, R.J., Nakayama, F.S. and Idso, S.B. 1976. Comparison of field-measured and calculated soil heat fluxes. Soil Sci. Soc. Am. J. 40:18-28.

        Lettau, B. 1971. Determination of the thermal diffusivity in the upper layers of a natural ground cover. Soil Sci. 112:173-177.

        Monin, A.S. 1973. Boundary layers in planetary atmospheres. In: P. Morrel (ed.), Dynamic meteorology, D. Reidel Publishing Company, Boston, pp. 419-458.

        Spitters, C.J.T., Toussaint, H.A.J.M. and Goudriaan, J. 1986. Separating the diffuse and direct component of global radiation and its implications for modeling canopy photosynthesis. Part I. Components of incoming radiation. Agric. For. Meteorol. 38:217-229.

        Wierenga, P.J. and de Wit, C.T. 1970. Simulation of heat flow in soils. Soil Sci. Soc. Am. Proc. 34:845-848.

        Wierenga, P.J., Hagan, R.M. and Nielsen, D.R. 1970. Soil temperature profiles during infiltration and redistribution of cool and warm irrigation water. Water Resour. Res. 6:230-238.

        Wierenga, P.J., Nielsen, D.R. and Hagan, R.M. 1969. Thermal properties of soil based upon field and laboratory measurements. Soil Sci. Soc. Am. Proc. 33:354-360.
        """
        state = self._current_state
        # Compute dts, the daily change in deep soil temperature (C), as a site-dependent function of Daynum.
        cdef double dts = 2 * pi * SitePar[10] / 365 * cos(2 * pi * (state.date.timetuple().tm_yday - SitePar[11]) / 365)
        # Define iter1 and dlt for hourly time step.
        cdef int iter1 = 24  # number of iterations per day.
        cdef double dlt = 3600  # time (seconds) of one iteration.
        cdef int kk = 1  # number of soil columns for executing computations.
        # If there is no canopy cover, no horizontal heat flux is assumed, kk = 1.
        # Otherwise it is equal to the number of columns in the slab.
        cdef double shadeav = 0  # average shaded area in all shaded soil columns.
        # emerge_switch defines the type of soil temperature computation.
        if self.emerge_switch > 1:
            shadetot = 0  # sum of shaded area in all shaded soil columns.
            nshadedcol = 0  # number of at least partially shaded soil columns.
            kk = nk
            for k in range(nk):
                if self.relative_radiation_received_by_a_soil_column[k] <= 0.99:
                    shadetot += 1 - self.relative_radiation_received_by_a_soil_column[k]
                    nshadedcol += 1

            if nshadedcol > 0:
                shadeav = shadetot / nshadedcol
        # Set daily averages of soil temperature to zero.
        state.soil_temperature[:] = 0
        # es and ActualSoilEvaporation are computed as the average for the whole soil slab, weighted by column widths.
        cdef double es = 0  # potential evaporation rate, mm day-1
        state.potential_evaporation = 0
        state.actual_soil_evaporation = 0
        # Start hourly loop of iterations.
        for ihr in range(iter1):
            # Update the temperature of the last soil layer (lower boundary conditions).
            state.deep_soil_temperature += dts * dlt / 86400
            etp0 = 0  # actual transpiration (mm s-1) for this hour
            if state.evapotranspiration > 0.000001:
                etp0 = state.actual_transpiration * state.hours[ihr].ref_et / state.evapotranspiration / dlt
            # Compute vertical transport for each column
            for k in range(kk):
                #  Set SoilTemp for the lowest soil layer.
                SoilTemp[nl - 1][k] = state.deep_soil_temperature
                # Compute transpiration from each column, weighted by its relative shading.
                etp1 = 0  # actual hourly transpiration (mm s-1) for a column.
                if shadeav > 0.000001:
                    etp1 = etp0 * (1 - self.relative_radiation_received_by_a_soil_column[k]) / shadeav
                ess = 0  # evaporation rate from surface of a soil column (mm / sec).
                # The potential evaporation rate (escol1k) from a column is the sum of the radiation component of the Penman equation(es1hour), multiplied by the relative radiation reaching this column, and the wind and vapor deficit component of the Penman equation (es2hour).
                # potential evaporation fron soil surface of a column, mm per hour.
                escol1k = state.hours[ihr].et1 * self.relative_radiation_received_by_a_soil_column[k] + state.hours[ihr].et2
                es += escol1k * wk(k, self.row_space)
                # Compute actual evaporation from soil surface. update cell.water_content of the soil soil cell, and add to daily sum of actual evaporation.
                evapmax = 0.9 * (state._[0].soil.cells[0][k].water_content - thad[0]) * 10 * dl(0)  # maximum possible evaporatio from a soil cell near the surface.
                escol1k = min(evapmax, escol1k)
                state._[0].soil.cells[0][k].water_content -= 0.1 * escol1k / dl(0)
                state.actual_soil_evaporation += escol1k * wk(k, self.row_space)
                ess = escol1k / dlt
                # Call self._energy_balance to compute soil surface and canopy temperature.
                self._energy_balance(u, ihr, k, ess, etp1)
            # Compute soil temperature flux in the vertical direction.
            # Assign iv = 1, layer = 0, nn = nl.
            iv = 1  # indicates vertical (=1) or horizontal (=0) flux.
            nn = nl  # number of array members for heat flux.
            layer = 0  # soil layer number
            # Loop over kk columns, and call SoilHeatFlux().
            for k in range(kk):
                SoilHeatFlux(state._[0], dlt, iv, nn, layer, k, self.row_space)
            # If no horizontal heat flux is assumed, make all array members of SoilTemp equal to the value computed for the first column. Also, do the same for array memebers of cell.water_content.
            if self.emerge_switch <= 1:
                for l in range(nl):
                    for k in range(nk):
                        SoilTemp[l][k] = SoilTemp[l][0]
                        if l == 0:
                            state._[0].soil.cells[l][k].water_content = state._[0].soil.cells[l][0].water_content
            # Compute horizontal transport for each layer

            # Compute soil temperature flux in the horizontal direction, when self.emerge_switch = 2.
            # Assign iv = 0 and nn = nk. Start loop for soil layers, and call SoilHeatFlux.
            if self.emerge_switch > 1:
                iv = 0
                nn = nk
                for l in range(nl):
                    layer = l
                    SoilHeatFlux(state._[0], dlt, iv, nn, layer, l, self.row_space)
            # Compute average temperature of soil layers, in degrees C.
            tsolav = [0] * nl  # hourly average soil temperature C, of a soil layer.
            for l in range(nl):
                for k in range(nk):
                    state.soil_temperature[l][k] += SoilTemp[l][k]
                    tsolav[l] += SoilTemp[l][k] - 273.161
                tsolav[l] /= nk
            # Compute average temperature of foliage, in degrees C. The average is weighted by the canopy shading of each column, only columns which are shaded 5% or more by canopy are used.
            tfc = 0  # average foliage temperature, weighted by shading in each column
            shading = 0  # sum of shaded area in all shaded columns, used to compute TFC
            for k in range(nk):
                if self.relative_radiation_received_by_a_soil_column[k] <= 0.95:
                    tfc += (FoliageTemp[k] - 273.161) * (1 - self.relative_radiation_received_by_a_soil_column[k])
                    shading += 1 - self.relative_radiation_received_by_a_soil_column[k]
            if shading >= 0.01:
                tfc = tfc / shading
            # If emergence date is to be simulated, call predict_emergence().
            if self.emerge_switch == 0 and state.date >= self.plant_date:
                emerge_date = state.predict_emergence(self.plant_date, ihr, self.plant_row_column)
                if emerge_date is not None:
                    self.emerge_date = emerge_date
                    self.emerge_switch = 2
        # At the end of the day compute actual daily evaporation and its cumulative sum.
        if kk == 1:
            es /= wk(1, self.row_space)
            state.actual_soil_evaporation /= wk(1, self.row_space)
        else:
            es /= self.row_space
            state.actual_soil_evaporation /= self.row_space
        if state.kday > 0:
            state.potential_evaporation = es
        # compute daily averages.
        state.soil_temperature[:] /= iter1

    def _daily_climate(self, u):
        state = self._current_state
        cdef double declination  # daily declination angle, in radians
        cdef double sunr  # time of sunrise, hours.
        cdef double suns  # time of sunset, hours.
        cdef double tmpisr  # extraterrestrial radiation, \frac{W}{m^2}
        hour = timedelta(hours=1)
        result = compute_day_length((self.latitude, self.longitude), state.date)
        declination = result["declination"]
        zero = result["sunr"].replace(hour=0, minute=0, second=0, microsecond=0)
        sunr = (result["sunr"] - zero) / hour
        suns = (result["suns"] - zero) / hour
        tmpisr = result["tmpisr"]
        state.solar_noon = (result["solar_noon"] - zero) / hour
        state.day_length = result["day_length"] / hour

        cdef double xlat = self.latitude * pi / 180  # latitude converted to radians.
        cdef double cd = cos(xlat) * cos(declination)  # amplitude of the sine of the solar height.
        cdef double sd = sin(xlat) * sin(declination)  # seasonal offset of the sine of the solar height.
        # The computation of the daily integral of global radiation (from sunrise to sunset) is based on Spitters et al. (1986).
        cdef double c11 = 0.4  # constant parameter
        cdef double radsum
        if abs(sd / cd) >= 1:
            radsum = 0
        else:
            # dsbe is the integral of sinb * (1 + c11 * sinb) from sunrise to sunset.
            dsbe = acos(-sd / cd) * 24 / pi * (sd + c11 * sd * sd + 0.5 * c11 * cd * cd) + 12 * (
                    cd * (2 + 3 * c11 * sd)) * sqrt(1 - (sd / cd) * (sd / cd)) / pi
            # The daily radiation integral is computed for later use in function Radiation.
            # Daily radiation intedral is converted from langleys to Watt m - 2, and divided by dsbe.
            # 11.630287 = 1000000 / 3600 / 23.884
            radsum = self._sim.climate[u].Rad * 11.630287 / dsbe
        cdef double rainToday
        rainToday = self._sim.climate[u].Rain  # the amount of rain today, mm
        # Set 'pollination switch' for rainy days (as in GOSSYM).
        state.pollination_switch = rainToday < 2.5
        # Call SimulateRunoff() only if the daily rainfall is more than 2 mm.
        # Note: this is modified from the original GOSSYM - RRUNOFF routine. It is called here for rainfall only, but it is not activated when irrigation is applied.
        cdef double runoffToday = 0  # amount of runoff today, mm
        if rainToday >= 2.0:
            runoffToday = SimulateRunoff(self._sim, u, SandVolumeFraction[0], ClayVolumeFraction[0], NumIrrigations)
            if runoffToday < rainToday:
                rainToday -= runoffToday
            else:
                rainToday = 0
            self._sim.climate[u].Rain = rainToday
        self._sim.states[u].runoff = runoffToday
        # Parameters for the daily wind function are now computed:
        cdef double t1 = sunr + SitePar[1]  # the hour at which wind begins to blow (SitePar(1) hours after sunrise).
        cdef double t2 = state.solar_noon + SitePar[
            2]  # the hour at which wind speed is maximum (SitePar(2) hours after solar noon).
        cdef double t3 = suns + SitePar[3]  # the hour at which wind stops to blow (SitePar(3) hours after sunset).
        cdef double wnytf = SitePar[4]  # used for estimating night time wind (from time t3 to time t1 next day).

        for ihr in range(24):
            hour = state.hours[ihr]
            ti = ihr + 0.5
            sinb = sd + cd * cos(pi * (ti - state.solar_noon) / 12)
            hour.radiation = radiation(radsum, sinb, c11)
            hour.temperature = daytmp(self._sim, u, ti, SitePar[8], sunr, suns)
            hour.dew_point = tdewhour(self._sim, u, ti, hour.temperature, sunr, state.solar_noon, SitePar[8], SitePar[12], SitePar[13], SitePar[14])
            hour.humidity = dayrh(hour.temperature, hour.dew_point)
            hour.wind_speed = compute_hourly_wind_speed(ti, self._sim.climate[u].Wind * 1000 / 86400, t1, t2, t3, wnytf)
        # Compute average daily temperature, using function AverageAirTemperatures.
        state.calculate_average_temperatures()
        # Compute potential evapotranspiration.
        state.compute_evapotranspiration(self.latitude, self.elevation, declination, tmpisr, SitePar[7])

    def _stress(self, u):
        state = self._current_state
        global AverageLwp
        # The following constant parameters are used:
        cdef double[9] vstrs = [-3.0, 3.229, 1.907, 0.321, -0.10, 1.230, 0.340, 0.30, 0.05]
        # Call state.leaf_water_potential() to compute leaf water potentials.
        state.leaf_water_potential(self.row_space)
        # The running averages, for the last three days, are computed:
        # average_min_leaf_water_potential is the average of state.min_leaf_water_potential, and AverageLwp of state.min_leaf_water_potential + state.max_leaf_water_potential.
        state.average_min_leaf_water_potential += (state.min_leaf_water_potential - LwpMinX[2]) / 3
        AverageLwp += (state.min_leaf_water_potential + state.max_leaf_water_potential - LwpX[2]) / 3
        for i in (2, 1):
            LwpMinX[i] = LwpMinX[i - 1]
            LwpX[i] = LwpX[i - 1]
        LwpMinX[0] = state.min_leaf_water_potential
        LwpX[0] = state.min_leaf_water_potential + state.max_leaf_water_potential
        if state.kday < 5:
            self.ptsred = 1
            state.water_stress_stem = 1
            return
        # The computation of ptsred, the effect of moisture stress on the photosynthetic rate, is based on the following work:
        # Ephrath, J.E., Marani, A., Bravdo, B.A., 1990. Effects of moisture stress on stomatal resistance and photosynthetic rate in cotton (Gossypium hirsutum) 1. Controlled levels of stress. Field Crops Res.23:117-131.
        # It is a function of average_min_leaf_water_potential (average min_leaf_water_potential for the last three days).
        state.average_min_leaf_water_potential = max(state.average_min_leaf_water_potential, vstrs[0])
        self.ptsred = min(vstrs[1] + state.average_min_leaf_water_potential * (vstrs[2] + vstrs[3] * state.average_min_leaf_water_potential), 1)
        # The general moisture stress factor (WaterStress) is computed as an empirical function of AverageLwp. psilim, the value of AverageLwp at the maximum value of the function, is used for truncating it.
        # The minimum value of WaterStress is 0.05, and the maximum is 1.
        cdef double psilim  # limiting value of AverageLwp.
        cdef double WaterStress
        psilim = -0.5 * vstrs[5] / vstrs[6]
        WaterStress = 1 if AverageLwp > psilim else min(max(vstrs[4] - AverageLwp * (vstrs[5] + vstrs[6] * AverageLwp), 0.05), 1)
        # Water stress affecting plant height and stem growth(WaterStressStem) is assumed to be more severe than WaterStress, especially at low WaterStress values.
        state.water_stress_stem = max(WaterStress * (1 + vstrs[7] * (2 - WaterStress)) - vstrs[7], vstrs[8])
        state.water_stress = WaterStress

    def _potential_fruit_growth(self, u):
        """
        This function simulates the potential growth of fruiting sites of cotton plants. It is called from PlantGrowth(). It calls TemperatureOnFruitGrowthRate()

        The following global variables are set here:
            PotGroAllBolls, PotGroAllBurrs, PotGroAllSquares.

        References:
        Marani, A. 1979. Growth rate of cotton bolls and their components. Field Crops Res. 2:169-175.
        Marani, A., Phene, C.J. and Cardon, G.E. 1992. CALGOS, a version of GOSSYM adapted for irrigated cotton.  III. leaf and boll growth routines. Beltwide Cotton Grow, Res. Conf. 1992:1361-1363.
        """
        state = self._current_state
        global PotGroAllSquares, PotGroAllBolls, PotGroAllBurrs
        # The constant parameters used:
        cdef double[5] vpotfrt = [0.72, 0.30, 3.875, 0.125, 0.17]
        # Compute tfrt for the effect of temperature on boll and burr growth rates. Function TemperatureOnFruitGrowthRate() is used (with parameters derived from GOSSYM), for day time and night time temperatures, weighted by day and night lengths.
        cdef double tfrt  # the effect of temperature on rate of boll, burr or square growth.
        tfrt = (state.day_length * TemperatureOnFruitGrowthRate(state.daytime_temperature) + (24 - state.day_length) * TemperatureOnFruitGrowthRate(state.nighttime_temperature)) / 24
        # Assign zero to sums of potential growth of squares, bolls and burrs.
        PotGroAllSquares = 0
        PotGroAllBolls = 0
        PotGroAllBurrs = 0
        # Assign values for the boll growth equation parameters. These are cultivar - specific.
        cdef double agemax = self.cultivar_parameters[9]  # maximum boll growth period (physiological days).
        cdef double rbmax = self.cultivar_parameters[10]  # maximum rate of boll (seed and lint) growth, g per boll per physiological day.
        cdef double wbmax = self.cultivar_parameters[11]  # maximum possible boll (seed and lint) weight, g per boll.
        # Loop for all vegetative stems.
        for k in range(self._sim.states[u].number_of_vegetative_branches):  # loop of vegetative stems
            for l in range(self._sim.states[u].vegetative_branches[k].number_of_fruiting_branches):  # loop of fruiting branches
                for m in range(self._sim.states[u].vegetative_branches[k].fruiting_branches[l].number_of_fruiting_nodes):  # loop for nodes on a fruiting branch
                    # Calculate potential square growth for node (k,l,m).
                    # Sum potential growth rates of squares as PotGroAllSquares.
                    if self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].stage == Stage.Square:
                        # ratesqr is the rate of square growth, g per square per day.
                        # The routine for this is derived from GOSSYM, and so are the parameters used.
                        ratesqr = tfrt * vpotfrt[3] * exp(-vpotfrt[2] + vpotfrt[3] * self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].age)
                        self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].square.potential_growth = ratesqr * self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].fraction
                        PotGroAllSquares += self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].square.potential_growth
                    # Growth of seedcotton is simulated separately from the growth of burrs. The logistic function is used to simulate growth of seedcotton. The constants of this function for cultivar 'Acala-SJ2', are based on the data of Marani (1979); they are derived from calibration for other cultivars
                    # agemax is the age of the boll (in physiological days after bloom) at the time when the boll growth rate is maximal.
                    # rbmax is the potential maximum rate of boll growth (g seeds plus lint dry weight per physiological day) at this age.
                    # wbmax is the maximum potential weight of seed plus lint (g dry weight per boll).
                    # The auxiliary variable pex is computed as
                    #    pex = exp(-4 * rbmax * (t - agemax) / wbmax)
                    # where t is the physiological age of the boll after bloom (= agebol).
                    # Boll weight (seed plus lint) at age T, according to the logistic function is:
                    #    wbol = wbmax / (1 + pex)
                    # and the potential boll growth rate at this age will be the derivative of this function:
                    #    ratebol = 4 * rbmax * pex / (1. + pex)**2
                    elif self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].stage == Stage.YoungGreenBoll or self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].stage == Stage.GreenBoll:
                        # pex is an intermediate variable to compute boll growth.
                        pex = exp(-4 * rbmax * (self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].boll.age - agemax) / wbmax)
                        # ratebol is the rate of boll (seed and lint) growth, g per boll per day.
                        ratebol = 4 * tfrt * rbmax * pex / (1 + pex) ** 2
                        # Potential growth rate of the burrs is assumed to be constant (vpotfrt[4] g dry weight per day) until the boll reaches its final volume. This occurs at the age of 22 physiological days in 'Acala-SJ2'. Both ratebol and ratebur are modified by temperature (tfrt) and ratebur is also affected by water stress (wfdb).
                        # Compute wfdb for the effect of water stress on burr growth rate. wfdb is the effect of water stress on rate of burr growth.
                        wfdb = min(max(vpotfrt[0] + vpotfrt[1] * state.water_stress, 0), 1)
                        ratebur = None  # rate of burr growth, g per boll per day.
                        if self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].boll.age >= 22:
                            ratebur = 0
                        else:
                            ratebur = vpotfrt[4] * tfrt * wfdb
                        # Potential boll (seeds and lint) growth rate (ratebol) and potential burr growth rate (ratebur) are multiplied by FruitFraction to compute PotGroBolls and PotGroBurrs for node (k,l,m).
                        self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].boll.potential_growth = ratebol * self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].fraction
                        self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].burr.potential_growth = ratebur * self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].fraction
                        # Sum potential growth rates of bolls and burrs as PotGroAllBolls and PotGroAllBurrs, respectively.
                        PotGroAllBolls += self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].boll.potential_growth
                        PotGroAllBurrs += self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].burr.potential_growth

                    # If these are not green bolls, their potential growth is 0. End loop.
                    else:
                        self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].boll.potential_growth = 0
                        self._sim.states[u].vegetative_branches[k].fruiting_branches[l].nodes[m].burr.potential_growth = 0

    def _potential_leaf_growth(self, u):
        """
        This function simulates the potential growth of leaves of cotton plants. It is called from self._growth(). It calls function temperature_on_leaf_growth_rate().

        The following monomolecular growth function is used :
            leaf area = smax * (1 - exp(-c * pow(t,p)))
        where    smax = maximum leaf area.
            t = time (leaf age).
            c, p = constant parameters.
        Note: p is constant for all leaves, whereas smax and c depend on leaf position.
        The rate per day (the derivative of this function) is :
            r = smax * c * p * exp(-c * pow(t,p)) * pow(t, (p-1))
        """
        state = self._current_state
        # The following constant parameters. are used in this function:
        p = 1.6  # parameter of the leaf growth rate equation.
        vpotlf = [3.0, 0.95, 1.2, 13.5, -0.62143, 0.109365, 0.00137566, 0.025, 0.00005, 30., 0.02, 0.001, 2.50, 0.18]
        # Calculate water stress reduction factor for leaf growth rate (wstrlf). This has been empirically calibrated in COTTON2K.
        wstrlf = state.water_stress * (1 + vpotlf[0] * (2 - state.water_stress)) - vpotlf[0]
        if wstrlf < 0.05:
            wstrlf = 0.05
        # Calculate wtfstrs, the effect of leaf water stress on state.leaf_weight_area_ratio (the ratio of leaf dry weight to leaf area). This has also been empirically calibrated in COTTON2K.
        wtfstrs = vpotlf[1] + vpotlf[2] * (1 - wstrlf)
        # Compute the ratio of leaf dry weight increment to leaf area increment (g per dm2), as a function of average daily temperature and water stress. Parameters for the effect of temperature are adapted from GOSSYM.
        tdday = state.average_temperature  # limited value of today's average temperature.
        if tdday < vpotlf[3]:
            tdday = vpotlf[3]
        state.leaf_weight_area_ratio = wtfstrs / (vpotlf[4] + tdday * (vpotlf[5] - tdday * vpotlf[6]))
        # Assign zero to total potential growth of leaf and petiole.
        state.leaf_potential_growth = 0
        state.petiole_potential_growth = 0
        c = 0  # parameter of the leaf growth rate equation.
        smax = 0  # maximum possible leaf area, a parameter of the leaf growth rate equation.
        rate = 0  # growth rate of area of a leaf.
        # Compute the potential growth rate of prefruiting leaves. smax and c are functions of prefruiting node number.
        for j in range(state.number_of_pre_fruiting_nodes):
            if state.leaf_area_pre_fruiting[j] <= 0:
                PotGroLeafAreaPreFru[j] = 0
                PotGroLeafWeightPreFru[j] = 0
                PotGroPetioleWeightPreFru[j] = 0
            else:
                jp1 = j + 1
                smax = max(self.cultivar_parameters[4], jp1 * (self.cultivar_parameters[2] - self.cultivar_parameters[3] * jp1))
                c = vpotlf[7] + vpotlf[8] * jp1 * (jp1 - vpotlf[9])
                rate = smax * c * p * exp(-c * pow(state.pre_fruiting_nodes[j].age, p)) * pow(state.pre_fruiting_nodes[j].age, (p - 1))
                # Growth rate is modified by water stress and a function of average temperature.
                # Compute potential growth of leaf area, leaf weight and petiole weight for leaf on node j. Add leaf weight potential growth to leaf_potential_growth.
                # Add potential growth of petiole weight to petiole_potential_growth.
                if rate >= 1e-12:
                    PotGroLeafAreaPreFru[j] = rate * wstrlf * temperature_on_leaf_growth_rate(state.average_temperature)
                    PotGroLeafWeightPreFru[j] = PotGroLeafAreaPreFru[j] * state.leaf_weight_area_ratio
                    PotGroPetioleWeightPreFru[j] = PotGroLeafAreaPreFru[j] * state.leaf_weight_area_ratio * vpotlf[13]
                    state.leaf_potential_growth += PotGroLeafWeightPreFru[j]
                    state.petiole_potential_growth += PotGroPetioleWeightPreFru[j]
        # denfac is the effect of plant density on leaf growth rate.
        cdef double denfac = 1 - vpotlf[12] * (1 - self.density_factor)
        for vegetative_branch in state.vegetative_branches:
            for l, fruiting_branch in enumerate(vegetative_branch.fruiting_branches):
                # smax and c are  functions of fruiting branch number.
                # smax is modified by plant density, using the density factor denfac.
                # Compute potential main stem leaf growth, assuming that the main stem leaf is initiated at the same time as leaf (k,l,0).
                main_stem_leaf = fruiting_branch.main_stem_leaf
                if main_stem_leaf.area <= 0:
                    main_stem_leaf.potential_growth_of_area = 0
                    main_stem_leaf.potential_growth_of_weight = 0
                    main_stem_leaf.potential_growth_of_petiole = 0
                else:
                    lp1 = l + 1
                    smax = denfac * (self.cultivar_parameters[5] + self.cultivar_parameters[6] * lp1 * (self.cultivar_parameters[7] - lp1))
                    smax = max(self.cultivar_parameters[4], smax)
                    c = vpotlf[10] + lp1 * vpotlf[11]
                    if fruiting_branch.nodes[0].leaf.age > 70:
                        rate = 0
                    else:
                        rate = smax * c * p * exp(-c * pow(fruiting_branch.nodes[0].leaf.age, p)) * pow(fruiting_branch.nodes[0].leaf.age, (p - 1))
                    # Add leaf and petiole weight potential growth to SPDWL and SPDWP.
                    if rate >= 1e-12:
                        main_stem_leaf.potential_growth_of_area = rate * wstrlf * temperature_on_leaf_growth_rate(state.average_temperature)
                        main_stem_leaf.potential_growth_of_weight = main_stem_leaf.potential_growth_of_area * state.leaf_weight_area_ratio
                        main_stem_leaf.potential_growth_of_petiole = main_stem_leaf.potential_growth_of_area * state.leaf_weight_area_ratio * vpotlf[13]
                        state.leaf_potential_growth += main_stem_leaf.potential_growth_of_weight
                        state.petiole_potential_growth += main_stem_leaf.potential_growth_of_petiole
                # Assign smax value of this main stem leaf to smaxx, c to cc.
                # Loop over the nodes of this fruiting branch.
                smaxx = smax  # value of smax for the corresponding main stem leaf.
                cc = c  # value of c for the corresponding main stem leaf.
                for m, node in enumerate(fruiting_branch.nodes):
                    if node.leaf.area <= 0:
                        node.leaf.potential_growth = 0
                        node.petiole.potential_growth = 0
                    # Compute potential growth of leaf area and leaf weight for leaf on fruiting branch node (k,l,m).
                    # Add leaf and petiole weight potential growth to spdwl and spdwp.
                    else:
                        mp1 = m + 1
                        # smax and c are reduced as a function of node number on this fruiting branch.
                        smax = smaxx * (1 - self.cultivar_parameters[8] * mp1)
                        c = cc * (1 - self.cultivar_parameters[8] * mp1)
                        # Compute potential growth for the leaves on fruiting branches.
                        if node.leaf.age > 70:
                            rate = 0
                        else:
                            rate = smax * c * p * exp(-c * pow(node.leaf.age, p)) * pow(node.leaf.age, (p - 1))
                        if rate >= 1e-12:
                            # Growth rate is modified by water stress. Potential growth is computed as a function of average temperature.
                            node.leaf.potential_growth = rate * wstrlf * temperature_on_leaf_growth_rate(state.average_temperature)
                            node.petiole.potential_growth = node.leaf.potential_growth * state.leaf_weight_area_ratio * vpotlf[13]
                            state.leaf_potential_growth += node.leaf.potential_growth * state.leaf_weight_area_ratio
                            state.petiole_potential_growth += node.petiole.potential_growth

    def _add_vegetative_branch(self, u, stemNRatio, DaysTo1stSqare):
        """
        This function decides whether a new vegetative branch is to be added, and then forms it. It is called from CottonPhenology().
        """
        state = self._current_state
        if len(state.vegetative_branches) == 3:
            return
        # TimeToNextVegBranch is computed as a function of this average temperature.
        cdef double TimeToNextVegBranch  # time, in physiological days, for the next vegetative branch to be formed.
        node = state.vegetative_branches[-1].fruiting_branches[0].nodes[0]
        TimeToNextVegBranch = np.polynomial.Polynomial([13.39, -0.696, 0.012])(node.average_temperature)
        # Compare the age of the first fruiting site of the last formed vegetative branch with TimeToNextVegBranch plus DaysTo1stSqare and the delays caused by stresses, in order to decide if a new vegetative branch is to be formed.
        if node.age < TimeToNextVegBranch + state.phenological_delay_for_vegetative_by_carbon_stress + state.phenological_delay_by_nitrogen_stress + DaysTo1stSqare:
            return
        vb = VegetativeBranch.from_ptr(&state._[0].vegetative_branches[state.number_of_vegetative_branches], state.number_of_vegetative_branches)
        # Assign 1 to FruitFraction and FruitingCode of the first site of this branch.
        vb.fruiting_branches[0].nodes[0].fraction = 1
        vb.fruiting_branches[0].nodes[0].stage = Stage.Square
        # Add a new leaf to the first site of this branch.
        vb.fruiting_branches[0].nodes[0].leaf.area = self.cultivar_parameters[34]
        vb.fruiting_branches[0].nodes[0].leaf.weight = self.cultivar_parameters[34] * state.leaf_weight_area_ratio
        # Add a new mainstem leaf to the first node of this branch.
        vb.fruiting_branches[0].main_stem_leaf.area = self.cultivar_parameters[34]
        vb.fruiting_branches[0].main_stem_leaf.weight = vb.fruiting_branches[0].main_stem_leaf.area * state.leaf_weight_area_ratio
        # The initial mass and nitrogen in the new leaves are substracted from the stem.
        state.stem_weight -= vb.fruiting_branches[0].nodes[0].leaf.weight + vb.fruiting_branches[0].main_stem_leaf.weight
        state.leaf_weight += vb.fruiting_branches[0].nodes[0].leaf.weight + vb.fruiting_branches[0].main_stem_leaf.weight
        cdef double addlfn  # nitrogen moved to new leaves from stem.
        addlfn = (vb.fruiting_branches[0].nodes[0].leaf.weight + vb.fruiting_branches[0].main_stem_leaf.weight) * stemNRatio
        state.leaf_nitrogen += addlfn
        state.stem_nitrogen -= addlfn
        # Assign the initial value of the average temperature of the first site.
        # Define initial NumFruitBranches and NumNodes for the new vegetative branch.
        vb.fruiting_branches[0].nodes[0].average_temperature = state.average_temperature
        vb.number_of_fruiting_branches = 1
        vb.fruiting_branches[0].number_of_fruiting_nodes = 1
        # When a new vegetative branch is formed, increase NumVegBranches by 1.
        state.number_of_vegetative_branches += 1

    def _defoliate(self, u):
        """This function simulates the effects of defoliating chemicals applied on the cotton. It is called from SimulateThisDay()."""
        global PercentDefoliation
        cdef State state = self._current_state
        # constant parameters:
        cdef double p1 = -50.0
        cdef double p2 = 0.525
        cdef double p3 = 7.06
        cdef double p4 = 0.85
        cdef double p5 = 2.48
        cdef double p6 = 0.0374
        cdef double p7 = 0.0020

        # If this is first day set initial values of tdfkgh, defkgh to 0.
        if state.date <= self.emerge_date:
            self.tdfkgh = 0
            self.defkgh = 0
            self.idsw = False
        # Start a loop for five possible defoliant applications.
        for i in range(5):
            # If there are open bolls and defoliation prediction has been set, execute the following.
            if state.number_of_open_bolls > 0 and DefoliantAppRate[i] <= -99.9:
                # percentage of open bolls in total boll number
                OpenRatio = <int>(100 * state.number_of_open_bolls / (state.number_of_open_bolls + state.number_of_green_bolls))
                if i == 0 and not self.idsw:
                    # If this is first defoliation - check the percentage of boll opening.
                    # If it is after the defined date, or the percent boll opening is greater than the defined threshold - set defoliation date as this day and set a second prediction.
                    if (state.date.timetuple().tm_yday >= DefoliationDate[0] and DefoliationDate[0] > 0) or OpenRatio > DefoliationMethod[i]:
                        self.idsw = True
                        DefoliationDate[0] = state.date.timetuple().tm_yday
                        DefoliantAppRate[1] = -99.9
                        if self.defoliate_date is None or state.date < self.defoliate_date:
                            self.defoliate_date = state.date
                        DefoliationMethod[0] = 0
                # If 10 days have passed since the last defoliation, and the leaf area index is still greater than 0.2, set another defoliation.
                if i >= 1:
                    if state.date == doy2date(self.year, DefoliationDate[i - 1] + 10) and state.leaf_area_index >= 0.2:
                        DefoliationDate[i] = state.date.timetuple().tm_yday
                        if i < 4:
                            DefoliantAppRate[i + 1] = -99.9
                        DefoliationMethod[i] = 0
            if state.date.timetuple().tm_yday == DefoliationDate[i]:
                # If it is a predicted defoliation, assign tdfkgh as 2.5 .
                # Else, compute the amount intercepted by the plants in kg per ha (defkgh), and add it to tdfkgh.
                if DefoliantAppRate[i] < -99:
                    self.tdfkgh = 2.5
                else:
                    if DefoliationMethod[i] == 0:
                        self.defkgh += DefoliantAppRate[i] * 0.95 * 1.12085 * 0.75
                    else:
                        self.defkgh += DefoliantAppRate[i] * state.light_interception * 1.12085 * 0.75
                    self.tdfkgh += self.defkgh
            # If this is after the first day of defoliant application, compute the percent of leaves to be defoliated (PercentDefoliation), as a function of average daily temperature, leaf water potential, days after first defoliation application, and tdfkgh. The regression equation is modified from the equation suggested in GOSSYM.
            if DefoliationDate[i] > 0 and state.date > self.defoliate_date:
                dum = -state.min_leaf_water_potential * 10  # value of min_leaf_water_potential in bars.
                PercentDefoliation = p1 + p2 * state.average_temperature + p3 * self.tdfkgh + p4 * (state.date - self.defoliate_date).days + p5 * dum - p6 * dum * dum + p7 * state.average_temperature * self.tdfkgh * (state.date - self.defoliate_date).days * dum
                PercentDefoliation = min(max(PercentDefoliation, 0), 40)

    def _soil_procedures(self, u):
        """This function manages all the soil related processes, and is executed once each day."""
        global AverageSoilPsi, LocationColumnDrip, LocationLayerDrip, noitr
        state = self._current_state
        # The following constant parameters are used:
        cdef double cpardrip = 0.2
        cdef double cparelse = 0.4
        # Call function ApplyFertilizer() for nitrogen fertilizer application.
        state.apply_fertilizer(self.row_space, self.plant_population)
        cdef double DripWaterAmount = 0  # amount of water applied by drip irrigation
        cdef double WaterToApply  # amount of water applied by non-drip irrigation or rainfall
        # Check if there is rain on this day
        WaterToApply = self._sim.climate[u].Rain
        # When water is added by an irrigation defined in the input: update the amount of applied water.
        for i in range(NumIrrigations):
            if state.date.timetuple().tm_yday == self._sim.irrigation[i].day:
                if self._sim.irrigation[i].method == 2:
                    DripWaterAmount += self._sim.irrigation[i].amount
                    LocationColumnDrip = self._sim.irrigation[i].LocationColumnDrip
                    LocationLayerDrip = self._sim.irrigation[i].LocationLayerDrip
                else:
                    WaterToApply += self._sim.irrigation[i].amount
                break
        # The following will be executed only after plant emergence
        if state.date >= self.emerge_date and self.emerge_switch > 0:
            state.roots_capable_of_uptake()  # function computes roots capable of uptake for each soil cell
            AverageSoilPsi = state.average_psi(self.row_space)  # function computes the average matric soil water
            # potential in the root zone, weighted by the roots-capable-of-uptake.
            state.water_uptake(self.row_space, self.per_plant_area)  # function  computes water and nitrogen uptake by plants.
        if WaterToApply > 0:
            # For rain or surface irrigation.
            # The number of iterations is computed from the thickness of the first soil layer.
            noitr = <int>(cparelse * WaterToApply / (dl(0) + 2) + 1)
            # the amount of water applied, mm per iteration.
            applywat = WaterToApply / noitr
            # The following subroutines are called noitr times per day:
            # If water is applied, GravityFlow() is called when the method of irrigation is not by drippers, followed by CapillaryFlow().
            for iter in range(noitr):
                GravityFlow(self._sim.states[u].soil.cells, applywat, self.row_space)
                CapillaryFlow(self._sim, u)
        if DripWaterAmount > 0:
            # For drip irrigation.
            # The number of iterations is computed from the volume of the soil cell in which the water is applied.
            noitr = <int>(cpardrip * DripWaterAmount / (dl(LocationLayerDrip) * wk(LocationColumnDrip, self.row_space)) + 1)
            # the amount of water applied, mm per iteration.
            applywat = DripWaterAmount / noitr
            # If water is applied, DripFlow() is called followed by CapillaryFlow().
            for iter in range(noitr):
                DripFlow(state._[0].soil.cells, applywat, self.row_space)
                CapillaryFlow(self._sim, u)
        # When no water is added, there is only one iteration in this day.
        if WaterToApply + DripWaterAmount <= 0:
            noitr = 1
            CapillaryFlow(self._sim, u)

    def _soil_nitrogen(self, u):
        """This function computes the transformations of the nitrogen compounds in the soil."""
        state = self._current_state
        cdef double depth[40]  # depth to the end of each layer.
        # At start compute depth[l] as the depth to the bottom of each layer, cm.
        sumdl = 0  # sum of layer thicknesses.
        for l in range(nl):
            sumdl += dl(l)
            depth[l] = sumdl
        # For each soil cell: call functions UreaHydrolysis(), MineralizeNitrogen(), Nitrification() and Denitrification().
        for l in range(nl):
            for k in range(nk):
                if VolUreaNContent[l][k] > 0:
                    UreaHydrolysis(self._sim.states[u].soil.cells[l][k], l, k, state.soil_temperature[l][k])
                MineralizeNitrogen(self._sim.states[u].soil.cells[l][k], l, k, state.date.timetuple().tm_yday, self.start_date.timetuple().tm_yday, self.row_space, state.soil_temperature[l][k])
                if VolNh4NContent[l][k] > 0.00001:
                    Nitrification(self._sim.states[u].soil.cells[l][k], l, k, depth[l], state.soil_temperature[l][k])
                # Denitrification() is called if there are enough water and nitrates in the soil cell. cparmin is the minimum temperature C for denitrification.
                cparmin = 5
                if self._sim.states[u].soil.cells[l][k].nitrate_nitrogen_content > 0.001 and self._sim.states[u].soil.cells[l][k].water_content > FieldCapacity[l] and state.soil_temperature[l][k] >= (cparmin + 273.161):
                    Denitrification(self._sim.states[u].soil.cells[l][k], l, k, self.row_space, state.soil_temperature[l][k])

    def _initialize_globals(self):
        # Define the numbers of rows and columns in the soil slab (nl, nk).
        # Define the depth, in cm, of consecutive nl layers.
        # NOTE: maxl and maxk are defined as constants in file "global.h".
        global nl, nk
        nl = maxl
        nk = maxk

    def _read_agricultural_input(self, inputs):
        global NumNitApps, NumIrrigations
        NumNitApps = 0
        idef = 0
        cdef Irrigation irrigation
        cdef NitrogenFertilizer nf
        for i in inputs:
            if i["type"] == "irrigation":
                irrigation.day = date2doy(i["date"])  # day of year of this irrigation
                irrigation.amount = i["amount"]  # net amount of water applied, mm
                irrigation.method = i.get("method", 0)  # method of irrigation: 1=  2=drip
                isdhrz = i.get("drip_horizontal_place", 0)  # horizontal placement cm
                isddph = i.get("drip_depth", 0)  # vertical placement cm
                # If this is a drip irrigation, convert distances to soil
                # layer and column numbers by calling SlabLoc.
                if irrigation.method == 2:
                    irrigation.LocationColumnDrip = SlabLoc(isdhrz, self.row_space)
                    irrigation.LocationLayerDrip = SlabLoc(isddph, 0)
                self._sim.irrigation[NumIrrigations] = irrigation
                NumIrrigations += 1
            elif i["type"] == "fertilization":
                nf.day = date2doy(i["date"])
                nf.amtamm = i.get("ammonium", 0)
                nf.amtnit = i.get("nitrate", 0)
                nf.amtura = i.get("urea", 0)
                nf.mthfrt = i.get("method", 0)
                isdhrz = i.get("drip_horizontal_place",
                            0)  # horizontal placement of DRIP, cm from left edge of soil slab.
                isddph = i.get("drip_depth",
                            0)  # vertical placement of DRIP, cm from soil surface.
                if nf.mthfrt == 1 or nf.mthfrt == 3:
                    nf.ksdr = SlabLoc(isdhrz, self.row_space)
                    nf.lsdr = SlabLoc(isddph, 0)
                else:
                    nf.ksdr = 0
                    nf.lsdr = 0
                NFertilizer[NumNitApps] = nf
                NumNitApps += 1
            elif i["type"] == "defoliation prediction":
                DefoliationDate[idef] = date2doy(i["date"])
                DefoliantAppRate[idef] = -99.9
                if idef == 0:
                    self.defoliate_date = doy2date(self.start_date.year, DefoliationDate[0])
                DefoliationMethod[idef] = i.get("method", 0)
                idef += 1

    def _initialize_soil_data(self):
        self._current_state.initialize_soil_data()
        InitializeSoilTemperature()
