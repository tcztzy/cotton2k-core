# pylint: disable=no-name-in-module, import-error
import csv
import datetime
import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from _cotton2k import Climate, SoilImpedance, SoilInit  # type: ignore[import]
from _cotton2k.simulation import Simulation as CySimulation  # type: ignore[import]
from _cotton2k.simulation import State as CyState

from .meteorology import METEOROLOGY
from .nitrogen import PlantNitrogen
from .phenology import Phenology, Stage
from .photo import Photosynthesis
from .root import RootGrowth
from .stem import StemGrowth

SOIL_IMPEDANCE = SoilImpedance()
with open(Path(__file__).parent / "soil_imp.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    SOIL_IMPEDANCE.curves = list(
        map(
            lambda row: {
                (k if k == "water" else float(k)): float(v) for k, v in row.items()
            },
            reader,
        )
    )


class State(
    Photosynthesis, Phenology, PlantNitrogen, RootGrowth, StemGrowth
):  # pylint: disable=too-many-instance-attributes
    _: CyState

    def __init__(self, state: CyState, sim) -> None:
        self._ = state
        self._sim = sim

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._, name)
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
        return (
            self.root_weight
            + self.stem_weight
            + self.green_bolls_weight
            + self.green_bolls_burr_weight
            + self.leaf_weight
            + self.petiole_weight
            + self.square_weight
            + self.open_bolls_weight
            + self.open_bolls_burr_weight
            + self.reserve_carbohydrate
        )

    @property
    def above_ground_biomass(self):
        # pylint: disable=no-member
        return (
            self.stem_weight
            + self.green_bolls_weight
            + self.green_bolls_burr_weight
            + self.leaf_weight
            + self.petiole_weight
            + self.square_weight
            + self.open_bolls_weight
            + self.open_bolls_burr_weight
            + self.reserve_carbohydrate
        )

    @property
    def agetop(self):
        l = len(self.vegetative_branches[0].fruiting_branches) - 1
        # average physiological age of top three nodes.
        if l < 0:
            return 0
        if l == 0:
            return self.vegetative_branches[0].fruiting_branches[0].nodes[0].age
        if l == 1:
            return (
                self.vegetative_branches[0].fruiting_branches[0].nodes[0].age * 2
                + self.vegetative_branches[0].fruiting_branches[1].nodes[0].age
            ) / 3
        return (
            self.vegetative_branches[0].fruiting_branches[l].nodes[0].age
            + self.vegetative_branches[0].fruiting_branches[l - 1].nodes[0].age
            + self.vegetative_branches[0].fruiting_branches[l - 2].nodes[0].age
        ) / 3

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
        super().__init__(kwargs.pop("version", 0x0400), **kwargs)
        SoilInit(**kwargs.pop("soil", {}))  # type: ignore[arg-type]
        start_date = kwargs["start_date"]
        if not isinstance(start_date, (datetime.date, str)):
            raise ValueError
        self.year = (
            start_date.year
            if isinstance(start_date, datetime.date)
            else int(start_date[:4])
        )
        self.read_input(**kwargs)
        METEOROLOGY[(kwargs["latitude"], kwargs["longitude"])] = {
            datetime.date.fromisoformat(kwargs["climate_start_date"])
            + datetime.timedelta(days=i): c
            for i, c in enumerate(kwargs["climate"])
        }
        climate_start_date = kwargs.pop("climate_start_date", 0)
        self.climate = Climate(climate_start_date, kwargs.pop("climate"))[self.start_date :]  # type: ignore[misc]  # pylint: disable=line-too-long

    def state(self, i):
        if isinstance(i, datetime.date):
            i = (i - self.start_date).days
        return self.states[i]

    def _copy_state(self, i):
        super()._copy_state(i)
        pre = self._current_state
        post = self._state(i + 1)
        post.date = pre.date + datetime.timedelta(days=1)
        for attr in (
            "average_min_leaf_water_potential",
            "carbon_allocated_for_root_growth",
            "delay_of_emergence",
            "delay_of_new_fruiting_branch",
            "extra_carbon",
            "fiber_length",
            "fiber_strength",
            "ginning_percent",
            "hypocotyl_length",
            "leaf_area_index",
            "leaf_weight",
            "lint_yield",
            "min_leaf_water_potential",
            "net_photosynthesis",
            "nitrogen_stress",
            "nitrogen_stress_vegetative",
            "nitrogen_stress_fruiting",
            "nitrogen_stress_root",
            "number_of_vegetative_branches",
            "pavail",
            "plant_height",
            "seed_layer_number",
            "seed_moisture",
            "stem_weight",
            "taproot_layer_number",
            "taproot_length",
        ):
            setattr(post, attr, getattr(pre, attr))
        root_weights = np.zeros((40, 20, 3), dtype=np.float64)
        np.copyto(root_weights, pre.root_weights)
        post.root_weights = root_weights
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

    def read_input(
        self, agricultural_inputs=None, **kwargs
    ):  # pylint: disable=unused-argument
        """This is the main function for reading input."""
        # pylint: disable=attribute-defined-outside-init
        self._current_state = self._state(0)
        self._init_state()
        self._soil_temperature_init()
        self._initialize_globals()
        self._initialize_switch()
        self._init_grid()
        self._read_agricultural_input(agricultural_inputs or [])
        self._initialize_soil_data()
        self._initialize_root_data()

    def run(self):
        try:
            self._simulate()
        except RuntimeError:
            pass
        return self

    def _simulate(self):
        self.states = []
        days = (self.stop_date - self.start_date).days
        for i in range(days):
            self._simulate_this_day(i)
            self._copy_state(i)
        self._simulate_this_day(days)

    # pylint: disable=attribute-defined-outside-init,no-member
    def _simulate_this_day(self, u):
        state = State(self._current_state, self)
        self.states.append(state)
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
                self.relative_radiation_received_by_a_soil_column,
            )
        else:
            state.kday = 0
            state.light_interception = 0
            self.relative_radiation_received_by_a_soil_column[:] = 1
        # The following functions are executed each day (also before emergence).
        self._daily_climate(u)  # computes climate variables for today.
        self._soil_temperature(
            u
        )  # executes all modules of soil and canopy temperature.
        self._soil_procedures(u)  # executes all other soil processes.
        self._soil_nitrogen(u)  # computes nitrogen transformations in the soil.
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
                self.climate[u]["Rad"],
                self.per_plant_area,
                self.ptsred,
                old_stem_weight,
            )  # computes net photosynthesis.
            self._growth(u, new_stem_weight)  # executes all modules of plant growth.
            state.phenology()  # executes all modules of plant phenology.
            state.plant_nitrogen(
                self.emerge_date, growing_stem_weight
            )  # computes plant nitrogen allocation.
        # Check if the date to stop simulation has been reached, or if this is the last
        # day with available weather data. Simulation will also stop when no leaves
        # remain on the plant.
        if state.kday > 10 and state.leaf_area_index < 0.0002:
            raise RuntimeError

    def _growth(self, u, new_stem_weight):
        state = State(self._current_state, self)
        # Call _potential_leaf_growth() to compute potential growth rate of leaves.
        self._potential_leaf_growth(u)
        # If it is after first square, call _potential_fruit_growth() to compute
        # potential growth rate of squares and bolls.
        if (
            len(state.vegetative_branches[0].fruiting_branches) > 0
            and len(state.vegetative_branches[0].fruiting_branches[0].nodes) > 0
            and state.vegetative_branches[0].fruiting_branches[0].nodes[0].stage
            != Stage.NotYetFormed
        ):
            self._potential_fruit_growth(u)
        # Call PotentialStemGrowth() to compute PotGroStem, potential growth rate of
        # stems.
        # The effect of temperature is introduced, by multiplying potential growth rate
        # by day_inc.
        # Stem growth is also affected by water stress(water_stress_stem).
        # stem_potential_growth is limited by (maxstmgr * per_plant_area) g per plant
        # per day.
        maxstmgr = 0.067  # maximum posible potential stem growth, g dm - 2 day - 1.
        state.stem_potential_growth = min(
            maxstmgr * self.per_plant_area,
            state.potential_stem_growth(
                new_stem_weight, self.density_factor, *self.cultivar_parameters[12:19]
            )
            * state.day_inc
            * state.water_stress_stem,
        )
        # Call PotentialRootGrowth() to compute potential growth rate on roots.
        # total potential growth rate of roots in g per slab.this is computed in
        # potential_root_growth() and used in actual_root_growth().
        sumpdr = state.potential_root_growth(3, self.per_plant_area)
        # Total potential growth rate of roots is converted from g per slab(sumpdr) to
        # g per plant (state.root_potential_growth).
        # Limit state.root_potential_growth to(maxrtgr * per_plant_area) g per plant
        # per day.
        maxrtgr = 0.045  # maximum possible potential root growth, g dm - 2 day - 1.
        state.root_potential_growth = min(
            maxrtgr * self.per_plant_area,
            sumpdr * 100 * self.per_plant_area / self.row_space,
        )
        # Call dry_matter_balance() to compute carbon balance, allocation of carbon to
        # plant parts, and carbon stress.
        vratio = state.dry_matter_balance(self.per_plant_area)
        # If it is after first square, call actual_fruit_growth() to compute actual
        # growth rate of squares and bolls.
        if (
            len(state.vegetative_branches[0].fruiting_branches) > 0
            and len(state.vegetative_branches[0].fruiting_branches[0].nodes) > 0
            and state.vegetative_branches[0].fruiting_branches[0].nodes[0].stage
            != Stage.NotYetFormed
        ):
            state.actual_fruit_growth()
        # Initialize state.leaf_weight.It is assumed that cotyledons fall off at time
        # of first square. Also initialize state.leaf_area and state.petiole_weight.
        if self.first_square_date is not None:
            state.leaf_weight = 0
            state.leaf_area = 0
        else:
            cotylwt = 0.20  # weight of cotyledons dry matter.
            state.leaf_weight = cotylwt
            state.leaf_area = 0.6 * cotylwt
        state.petiole_weight = 0
        # Call actual_leaf_growth to compute actual growth rate of leaves and compute
        # leaf area index.
        state.actual_leaf_growth(vratio)
        state.leaf_area_index = state.leaf_area / self.per_plant_area
        # Add actual_stem_growth to state.stem_weight.
        state.stem_weight += state.actual_stem_growth
        # Plant density affects growth in height of tall plants.
        htdenf = (
            55  # minimum plant height for plant density affecting growth in height.
        )
        z1 = min(
            max((state.plant_height - htdenf) / htdenf, 0),
            1,
        )  # intermediate variable to compute denf2.
        denf2 = 1 + z1 * (
            self.density_factor - 1
        )  # effect of plant density on plant growth in height.
        # Call add_plant_height to compute PlantHeight.
        if self.version < 0x500 or not state.date >= self.topping_date:
            # node numbers of top node.

            if len(state.vegetative_branches[0].fruiting_branches) >= 2:
                stage = state.vegetative_branches[0].fruiting_branches[1].nodes[0].stage
            else:
                stage = Stage.NotYetFormed
            state.plant_height += state.add_plant_height(
                denf2,
                state.day_inc,
                state.number_of_pre_fruiting_nodes,
                stage,
                state.age_of_pre_fruiting_nodes[state.number_of_pre_fruiting_nodes - 1],
                state.age_of_pre_fruiting_nodes[state.number_of_pre_fruiting_nodes - 2],
                state.agetop,
                state.water_stress_stem,
                state.carbon_stress,
                state.nitrogen_stress_vegetative,
                *self.cultivar_parameters[19:27]
            )
        # Call ActualRootGrowth() to compute actual root growth.
        state.compute_actual_root_growth(
            sumpdr,
            self.row_space,
            self._column_width,
            self.per_plant_area,
            3,
            self.emerge_date,
            self.plant_row_column,
        )

    @property
    def _column_width(self):
        return self.row_space / 20
