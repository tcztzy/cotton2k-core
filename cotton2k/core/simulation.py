from __future__ import annotations

import datetime
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from ._simulation import Simulation as CySimulation  # pylint: disable=E0611
from ._simulation import SoilInit  # pylint: disable=E0611
from ._simulation import State as CyState  # pylint: disable=E0611
from .growth import Growth
from .meteorology import METEOROLOGY, Meteorology
from .nitrogen import PlantNitrogen
from .phenology import Phenology, Stage
from .photo import Photosynthesis
from .root import RootGrowth
from .soil import SoilProcedure, form
from .stem import StemGrowth
from .thermology import Thermology


class State(
    Growth,
    Meteorology,
    Photosynthesis,
    Phenology,
    PlantNitrogen,
    RootGrowth,
    SoilProcedure,
    StemGrowth,
    Thermology,
):  # pylint: disable=too-many-instance-attributes,too-many-ancestors
    _: CyState
    sim: "Simulation"

    def __init__(
        self,
        state: CyState,
        sim,
        pre_state=None,
    ) -> None:
        self._ = state
        self._sim = sim
        if pre_state is None:
            self.fruiting_nodes_boll_cumulative_temperature = np.zeros(
                (3, 30, 5), dtype=np.double
            )
        else:
            for attr in (
                "fruiting_nodes_boll_cumulative_temperature",
                "soil_heat_flux_numiter",
            ):
                value = getattr(pre_state, attr)
                if hasattr(value, "copy") and callable(value.copy):
                    value = value.copy()
                setattr(self, attr, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._, name)
        except AttributeError:
            try:
                return getattr(self._sim, name)
            except AttributeError:
                return getattr(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_", "_sim"):
            object.__setattr__(self, name, value)
        else:
            try:
                setattr(self._, name, value)
            except AttributeError:
                object.__setattr__(self, name, value)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def keys(self):  # pylint: disable=no-self-use
        return [
            "date",
            "lint_yield",
            "ginning_percent",
            "leaf_area_index",
            "light_interception",
            "plant_height",
            *(
                org + "_weight"
                for org in (
                    "plant",
                    "stem",
                    "leaf",
                    "root",
                    "petiole",
                    "square",
                    "green_bolls",
                    "open_bolls",
                )
            ),
            "above_ground_biomass",
            "evapotranspiration",
            "actual_transpiration",
            "actual_soil_evaporation",
            *("number_of_" + org for org in ("squares", "green_bolls", "open_bolls")),
            "vegetative_branches",
        ]

    @property
    def plant_weight(self):
        # pylint: disable=no-member
        return self.root_weight + self.above_ground_biomass

    @property
    def square_weight(self):
        """total square weight, g per plant."""
        return self.square_weights.sum()

    @property
    def above_ground_biomass(self):
        # pylint: disable=no-member
        return (
            self.above_ground_maintenance_weight
            + self.open_bolls_weight
            + self.open_bolls_burr_weight
        )

    @property
    def above_ground_maintenance_weight(self):
        return (
            self.stem_weight  # pylint: disable=no-member
            + self.green_bolls_weight  # pylint: disable=no-member
            + self.green_bolls_burr_weight  # pylint: disable=no-member
            + self.leaf_weight  # pylint: disable=no-member
            + self.petiole_weight  # pylint: disable=no-member
            + self.square_weight
            + self.reserve_carbohydrate
        )

    @property
    def maintenance_weight(self):
        return self.root_weight + self.above_ground_maintenance_weight

    @property
    def open_bolls_weight(self):
        return (
            self.fruiting_nodes_boll_weight[
                self.fruiting_nodes_stage == Stage.MatureBoll
            ]
        ).sum()

    @property
    def lint_yield(self) -> float:
        """yield of lint, kgs per hectare."""
        return (
            (self.fruiting_nodes_boll_weight * self.fruiting_nodes_ginning_percent)[
                self.fruiting_nodes_stage == Stage.MatureBoll
            ].sum()
            * self._sim.plant_population
            * 0.001
        )

    @property
    def number_of_squares(self):
        return (
            self.fruiting_nodes_fraction[self.fruiting_nodes_stage == Stage.Square]
        ).sum()

    @property
    def agetop(self):
        l = len(self.vegetative_branches[0].fruiting_branches) - 1
        # average physiological age of top three nodes.
        if l < 0:
            return 0
        if l == 0:
            return self.fruiting_nodes_age[0, 0, 0]
        if l == 1:
            return (
                self.fruiting_nodes_age[0, 0, 0] * 2 + self.fruiting_nodes_age[0, 1, 0]
            ) / 3
        return (self.fruiting_nodes_age[0, l - 2 : l + 1, 0]).mean()

    @property
    def root_weight(self):
        """total root weight, g per plant."""
        return (
            self.root_weights.sum()
            * 100
            * self._sim.per_plant_area
            / self._sim.row_space
        )


class Simulation(CySimulation):  # pylint: disable=too-many-instance-attributes
    states: list[State] = []

    def __init__(self, path: Union[Path, str, dict]):
        if isinstance(path, dict):
            kwargs = path
        else:
            kwargs = json.loads(Path(path).read_text())
        coord = (kwargs["latitude"], kwargs["longitude"])
        if coord not in METEOROLOGY:
            METEOROLOGY[coord] = {
                datetime.date.fromisoformat(c.pop("date")): c for c in kwargs["climate"]
            }
        super().__init__(
            version=kwargs.pop("version", 0x0400), meteor=METEOROLOGY[coord], **kwargs
        )
        self.column_width = np.ones(20) * self.row_space / 20
        self.column_width_cumsum = self.column_width.cumsum()
        self.layer_depth = np.ones(40) * 5
        self.layer_depth[:3] = 2
        self.layer_depth[3] = 4
        self.layer_depth[-2:] = 10
        self.layer_depth_cumsum = self.layer_depth.cumsum()
        self.cell_area = self.layer_depth[:, None] * self.column_width[None, :]
        self.ratio_implicit = (
            kwargs.get("soil", {}).get("hydrology", {}).get("ratio_implicit", 0)
        )
        self.soil_hydrology = np.array(
            [
                (layer["depth"], layer["bulk_density"], layer["saturated_hydraulic_conductivity"])
                for layer in kwargs.get("soil", {})
                .get("hydrology", {})
                .get("layers", [])
            ],
            dtype=[
                # depth from soil surface to the end of horizon layers, cm.
                ("depth", np.double),
                # bulk density of soil in a horizon, g cm-3.
                ("bulk_density", np.double),
                # saturated hydraulic conductivity, cm per day.
                ("saturated_hydraulic_conductivity", np.double)
            ],
        )
        self.soil_horizon_number = np.searchsorted(
            self.soil_hydrology["depth"], self.layer_depth_cumsum
        )
        self.soil_bulk_density = self.soil_hydrology["bulk_density"][
            self.soil_horizon_number
        ]
        self.pclay = np.array(
            [
                l["clay"]
                for l in kwargs.get("soil", {}).get("hydrology", {}).get("layers", [])
            ]
        )
        self.psand = np.array(
            [
                l["sand"]
                for l in kwargs.get("soil", {}).get("hydrology", {}).get("layers", [])
            ]
        )
        self.oma = np.array(
            [l["organic_matter"] for l in kwargs.get("soil", {}).get("initial", [])]
        )
        SoilInit(**kwargs.pop("soil", {}))  # type: ignore[arg-type]
        start_date = kwargs["start_date"]
        if not isinstance(start_date, (datetime.date, str)):
            raise ValueError
        self.year = (
            start_date.year
            if isinstance(start_date, datetime.date)
            else int(start_date[:4])
        )
        self.thad = np.zeros(40, dtype=np.double)
        self.field_capacity = np.zeros(40, dtype=np.double)
        self.pore_space = (
            1
            - self.soil_bulk_density
            / 2.65  # density of the solid fraction of the soil (g / cm3)
        )
        self.heat_capacity_soil_solid = np.zeros(40, dtype=np.double)
        self.initialize_state0()
        self.read_input(**kwargs)

    def state(self, i):
        if isinstance(i, datetime.date):
            i = (i - self.start_date).days
        return self.states[i]

    def _copy_state(self, i):
        pre = self._current_state
        post = CyState(self, self.version)
        super()._copy_state(pre, post)
        post.date = pre.date + datetime.timedelta(days=1)
        for attr in (
            "_leaf_nitrogen_concentration",
            "pre_fruiting_nodes_age",
            "average_min_leaf_water_potential",
            "average_soil_psi",
            "carbon_allocated_for_root_growth",
            "delay_of_emergence",
            "delay_of_new_fruiting_branch",
            "extra_carbon",
            "fiber_length",
            "fiber_strength",
            "ginning_percent",
            "hypocotyl_length",
            "leaf_area_index",
            "leaf_nitrogen",
            "leaf_weight",
            "leaf_weight_pre_fruiting",
            "min_leaf_water_potential",
            "net_photosynthesis",
            "nitrogen_stress",
            "nitrogen_stress_vegetative",
            "nitrogen_stress_fruiting",
            "nitrogen_stress_root",
            "number_of_vegetative_branches",
            "numiter",
            "pavail",
            "plant_height",
            "pre_fruiting_nodes_age",
            "pre_fruiting_leaf_area",
            "seed_layer_number",
            "seed_moisture",
            "stem_weight",
            "taproot_layer_number",
            "taproot_length",
            "total_required_nitrogen",
            # ndarrays
            "burr_weight",
            "burr_potential_growth",
            "foliage_temperature",
            "fruiting_nodes_age",
            "fruiting_nodes_average_temperature",
            "fruiting_nodes_boll_age",
            "fruiting_nodes_boll_potential_growth",
            "fruiting_nodes_boll_weight",
            "fruiting_nodes_fraction",
            "fruiting_nodes_ginning_percent",
            "fruiting_nodes_stage",
            "main_stem_leaf_area",
            "main_stem_leaf_area_potential_growth",
            "main_stem_leaf_weight",
            "main_stem_leaf_potential_growth",
            "main_stem_leaf_petiole_weight",
            "main_stem_leaf_petiole_potential_growth",
            "node_delay",
            "node_leaf_age",
            "node_leaf_area",
            "node_leaf_area_potential_growth",
            "node_leaf_weight",
            "node_petiole_potential_growth",
            "node_petiole_weight",
            "root_weights",
            "root_growth_factor",
            "root_weight_capable_uptake",
            "soil_fresh_organic_matter",
            "soil_nitrate_content",
            "soil_psi",
            "soil_urea_content",
            "soil_water_content",
            "square_potential_growth",
            "square_weights",
        ):
            value = getattr(pre, attr)
            if hasattr(value, "copy") and callable(value.copy):
                value = value.copy()
            setattr(post, attr, value)
        post.hourly_soil_temperature = np.zeros((24, 40, 20), dtype=np.double)
        post.hourly_soil_temperature[0] = pre.hourly_soil_temperature[23]
        self.states.append(State(post, self, self.state(i)))
        self._current_state = post  # pylint: disable=attribute-defined-outside-init

    def _initialize_switch(self):
        """If the date of emergence has not been given, emergence will be simulated by
        the model. In this case, emerge_switch = 0, and a check is performed to make
        sure that the date of planting has been given."""

        if self.emerge_date is None:
            if self.plant_date is None:
                raise ValueError("planting date or emergence date must be given")
            self.emerge_switch = 0  # pylint: disable=attribute-defined-outside-init
        # If the date of emergence has been given in the input: emerge_switch = 1 if
        # simulation starts before emergence, or emerge_switch = 2 if simulation starts
        # at emergence.
        elif self.emerge_date > self.start_date:
            self.emerge_switch = 1  # pylint: disable=attribute-defined-outside-init
        else:
            self.emerge_switch = 2  # pylint: disable=attribute-defined-outside-init
            self._current_state.kday = 1

    def _init_grid(self):
        """Initializes the soil grid variables. It is executed once at the beginning of
        the simulation.
        """
        # plant_location is the distance from edge of slab, cm, of the plant row.
        # pylint: disable=access-member-before-definition
        plant_location = self.row_space / 2
        if self.skip_row_width > 0:
            # If there is a skiprow arrangement, row_space and plant_location are
            # redefined.
            # pylint: disable=attribute-defined-outside-init
            self.row_space = (self.row_space + self.skip_row_width) / 2
            plant_location = self.skip_row_width / 2
        # Compute plant_population - number of plants per hectar, and per_plant_area --
        # the average surface area per plant, in dm2, and the empirical plant density
        # factor (density_factor). This factor will be used to express the effect of
        # plant density on some plant growth rate functions.
        # NOTE: density_factor = 1 for 5 plants per sq m (or 50000 per ha).
        # pylint: disable=attribute-defined-outside-init
        self.plant_population = self.plants_per_meter / self.row_space * 1000000
        # pylint: disable=attribute-defined-outside-init
        self.per_plant_area = 1000000 / self.plant_population
        # pylint: disable=attribute-defined-outside-init
        self.density_factor = np.exp(
            self.cultivar_parameters[1] * (5 - self.plant_population / 10000)
        )

        # The width of the slab columns is computed by dividing the row spacing by the
        # number of columns. It is assumed that slab width is equal to the average row
        # spacing, and column widths are uniform.
        # NOTE: wk is an array - to enable the option of non-uniform column widths in
        # the future.
        # plant_row_column (the column including the plant row) is now computed from
        # plant_location (the distance of the plant row from the edge of the slab).
        self.plant_row_column = 0  # pylint: disable=attribute-defined-outside-init
        for k in range(20):
            sumwk = (k + 1) * self._column_width
            if self.plant_row_column == 0 and sumwk > plant_location:
                self.plant_row_column = k - int(
                    (sumwk - plant_location) > self._column_width / 2
                )

    def initialize_state0(self):
        state0 = CyState(self, self.version)
        state0.main_stem_leaf_area = np.zeros((3, 30), dtype=np.double)
        state0.main_stem_leaf_area_potential_growth = np.zeros((3, 30), dtype=np.double)
        state0.main_stem_leaf_weight = np.zeros((3, 30), dtype=np.double)
        state0.main_stem_leaf_potential_growth = np.zeros((3, 30), dtype=np.double)
        state0.main_stem_leaf_petiole_weight = np.zeros((3, 30), dtype=np.double)
        state0.main_stem_leaf_petiole_potential_growth = np.zeros(
            (3, 30), dtype=np.double
        )
        state0.square_potential_growth = np.zeros((3, 30, 5), dtype=np.double)
        state0.square_weights = np.zeros((3, 30, 5), dtype=np.double)
        state0.burr_weight = np.zeros((3, 30, 5), dtype=np.double)
        state0.burr_potential_growth = np.zeros((3, 30, 5), dtype=np.double)
        state0.node_delay = np.zeros((3, 30), dtype=np.double)
        state0.node_petiole_weight = np.zeros((3, 30, 5), dtype=np.double)
        state0.node_petiole_potential_growth = np.zeros((3, 30, 5), dtype=np.double)
        state0.node_leaf_age = np.zeros((3, 30, 5), dtype=np.double)
        state0.node_leaf_area = np.zeros((3, 30, 5), dtype=np.double)
        state0.node_leaf_area_potential_growth = np.zeros((3, 30, 5), dtype=np.double)
        state0.node_leaf_weight = np.zeros((3, 30, 5), dtype=np.double)
        state0.fruiting_nodes_average_temperature = np.zeros(
            (3, 30, 5), dtype=np.double
        )
        state0.fruiting_nodes_boll_age = np.zeros((3, 30, 5), dtype=np.double)
        state0.fruiting_nodes_boll_potential_growth = np.zeros(
            (3, 30, 5), dtype=np.double
        )
        state0.soil_psi = np.zeros((40, 20), dtype=np.double)
        state0.soil_urea_content = np.zeros((40, 20), dtype=np.double)
        state0.hourly_soil_temperature = np.zeros((24, 40, 20), dtype=np.double)
        state0.foliage_temperature = np.ones(20, dtype=np.double) * 295
        self._current_state = state0
        super()._init_state()
        self.states = [State(state0, self)]

    def read_input(
        self, agricultural_inputs=None, **kwargs
    ):  # pylint: disable=unused-argument
        """This is the main function for reading input."""
        if agricultural_inputs is None:
            agricultural_inputs = []
        # pylint: disable=attribute-defined-outside-init
        self._soil_temperature_init()
        self._initialize_globals()
        self._initialize_switch()
        self._init_grid()
        for ao in agricultural_inputs:
            if ao["type"] == "irrigation":
                del ao["type"]
                if not isinstance(d := ao.pop("date"), datetime.date):
                    d = datetime.date.fromisoformat(d)
                self.irrigation[d] = ao
        self._read_agricultural_input(agricultural_inputs)
        self._initialize_soil_data()
        self.initialize_soil_temperature()
        self._initialize_root_data()

    def run(self):
        try:
            self._simulate()
        except RuntimeError:
            pass
        return self

    def _simulate(self):
        days = (self.stop_date - self.start_date).days
        for i in range(days):
            self._simulate_this_day(i)
            self._copy_state(i)
        self._simulate_this_day(days)

    # pylint: disable=attribute-defined-outside-init,no-member
    def _simulate_this_day(self, u):
        state = self.states[-1]
        if state.date >= self.emerge_date:
            state.kday = (state.date - self.emerge_date).days + 1
            # pylint: disable=access-member-before-definition
            if state.leaf_area_index > self.max_leaf_area_index:
                self.max_leaf_area_index = state.leaf_area_index
            state.light_interception = state.compute_light_interception(
                self.max_leaf_area_index,
                self.row_space,
            )
            state.column_shading(
                self.row_space,
                self.plant_row_column,
                self._column_width,
                self.max_leaf_area_index,
            )
        else:
            state.kday = 0
            state.light_interception = 0
            self.irradiation_soil_surface[:] = 1
        # The following functions are executed each day (also before emergence).
        state.daily_meteorology()  # computes climate variables for today.
        state.soil_thermology()  # executes all modules of soil and canopy temperature.
        state.soil_procedures()  # executes all other soil processes.
        state.soil_nitrogen()  # computes nitrogen transformations in the soil.
        # The following is executed each day after plant emergence:
        if (
            state.date >= self.emerge_date
            # pylint: disable=access-member-before-definition
            and self.emerge_switch > 0
        ):
            # If this day is after emergence, assign to emerge_switch the value of 2.
            self.emerge_switch = 2
            self._defoliate(u)  # effects of defoliants applied.
            self._stress(u)  # computes water stress factors.
            old_stem_days = 32
            if state.kday > old_stem_days:
                old_stem_weight = self.state(u - 32).stem_weight
                new_stem_weight = state.stem_weight - old_stem_weight
                growing_stem_weight = new_stem_weight
            else:
                old_stem_weight = 0
                new_stem_weight = (
                    state.stem_weight - self.state(self.emerge_date).stem_weight
                )
                growing_stem_weight = state.stem_weight
            state.get_net_photosynthesis(
                self.meteor[state.date]["irradiation"],
                self.per_plant_area,
                self.ptsred,
                old_stem_weight,
            )  # computes net photosynthesis.
            state.growth(new_stem_weight)  # executes all modules of plant growth.
            state.phenology()  # executes all modules of plant phenology.
            state.plant_nitrogen(
                self.emerge_date, growing_stem_weight
            )  # computes plant nitrogen allocation.
        # Check if the date to stop simulation has been reached, or if this is the last
        # day with available weather data. Simulation will also stop when no leaves
        # remain on the plant.
        if state.kday > 10 and state.leaf_area_index < 0.0002:
            raise RuntimeError

    @property
    def _column_width(self):
        return self.row_space / 20

    bclay = 7  # heat conductivity of clay (mcal cm-1 s-1 C-1).
    bsand = 20  # heat conductivity of sand and silt (mcal cm-1 s-1 C-1).
    ckw = 1.45  # heat conductivity of water (mcal cm-1 s-1 C-1).
    cka = 0.0615  # heat conductivity of air (mcal cm-1 s-1 C-1).
    cmin = 0.46  # heat capacity of the mineral fraction of the soil.
    corg = 0.6  # heat capacity of the organic fraction of the soil.
    ga = 0.144  # shape factor for air in pore spaces.

    @cached_property
    def dclay(self):
        """aggregation factor for clay in water."""
        return form(self.bclay, self.ckw, self.ga)

    @cached_property
    def dsand(self):
        """aggregation factor for sand in water."""
        return form(self.bsand, self.ckw, self.ga)

    # marginal soil water content (as a function of soil texture) for computing soil
    # heat conductivity.
    marginal_water_content = np.zeros(40, dtype=np.double)
    # the heat conductivity of dry soil.
    heat_conductivity_dry_soil: npt.NDArray[np.double]
    input_layer_depth = 15

    pclay: npt.NDArray[np.double]  # percentage of clay in soil of horizon layers.
    psand: npt.NDArray[np.double]  # percentage of sand in soil of horizon layers.

    def initialize_soil_temperature(self):
        """Initializes the variables needed for the simulation of soil temperature, and
        variables used by functions soil_thermal_conductivity() and SoilHeatFlux().
        """
        rm = 2.65  # specific weight of mineral fraction of soil.
        ro = 1.3  # specific weight of organic fraction of soil.
        # Compute aggregation factors:
        # aggregation factor for sand in air
        dsandair = form(self.bsand, self.cka, self.ga)
        # aggregation factor for clay in air
        dclayair = form(self.bclay, self.cka, self.ga)
        # Loop over all soil layers, and define indices for some soil arrays.
        self.soil_sand_volume_fraction = np.zeros(40, dtype=np.double)
        self.soil_clay_volume_fraction = np.zeros(40, dtype=np.double)
        for l, sumdl in enumerate(self.layer_depth_cumsum):
            j = min(
                int((sumdl + self.input_layer_depth - 1) / self.input_layer_depth) - 1,
                13,
            )  # layer definition for oma
            # Using the values of the clay and organic matter percentages in the soil,
            # compute mineral and organic fractions of the soil, by weight and volume.
            mmo = self.oma[j] / 100  # organic matter fraction of dry soil (by weight).
            mm = 1 - mmo  # mineral fraction of dry soil (by weight).
            # MarginalWaterContent is set as a function of the sand fraction of the
            # soil.
            i1 = self.soil_horizon_number[
                l
            ]  # layer definition as in soil hydrology input file.
            self.marginal_water_content[l] = 0.1 - 0.07 * self.psand[i1] / 100
            # The volume fractions of clay (self.soil_clay_volume_fraction) and of sand
            # plus silt (self.soil_sand_volume_fraction), are calculated.
            ra = (mmo / ro) / (
                mm / rm
            )  # volume ratio of organic to mineral soil fractions.
            xo = (
                (1 - self.pore_space[l]) * ra / (1 + ra)
            )  # organic fraction of soil (by volume).
            xm = (1 - self.pore_space[l]) - xo  # mineral fraction of soil (by volume).
            self.soil_clay_volume_fraction[l] = self.pclay[i1] * xm / mm / 100
            self.soil_sand_volume_fraction[l] = (
                1 - self.pore_space[l] - self.soil_clay_volume_fraction[l]
            )
            # Heat capacity of the solid soil fractions (mineral + organic, by volume)
            self.heat_capacity_soil_solid[l] = xm * self.cmin + xo * self.corg
        # The heat conductivity of dry soil is computed using the procedure suggested
        # by De Vries.
        self.heat_conductivity_dry_soil = (
            1.25
            * (
                self.pore_space * self.cka
                + dsandair * self.bsand * self.soil_sand_volume_fraction
                + dclayair * self.bclay * self.soil_clay_volume_fraction
            )
            / (
                self.pore_space
                + dsandair * self.soil_sand_volume_fraction
                + dclayair * self.soil_clay_volume_fraction
            )
        )
