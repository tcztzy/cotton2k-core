from .phenology import Stage


class Growth:  # pylint: disable=no-member,too-few-public-methods,W0201
    def growth(self, new_stem_weight):
        u = (self.date - self.start_date).days
        # Call _potential_leaf_growth() to compute potential growth rate of leaves.
        self._potential_leaf_growth(u)
        # If it is after first square, call _potential_fruit_growth() to compute
        # potential growth rate of squares and bolls.
        if self.fruiting_nodes_stage[0, 0, 0] != Stage.NotYetFormed:
            self._potential_fruit_growth(u)
        # Call PotentialStemGrowth() to compute PotGroStem, potential growth rate of
        # stems.
        # The effect of temperature is introduced, by multiplying potential growth rate
        # by day_inc.
        # Stem growth is also affected by water stress(water_stress_stem).
        # stem_potential_growth is limited by (maxstmgr * per_plant_area) g per plant
        # per day.
        maxstmgr = 0.067  # maximum posible potential stem growth, g dm - 2 day - 1.
        self.stem_potential_growth = min(
            maxstmgr * self.per_plant_area,
            self.potential_stem_growth(
                new_stem_weight, self.density_factor, *self.cultivar_parameters[12:19]
            )
            * self.day_inc
            * self.water_stress_stem,
        )
        # Call PotentialRootGrowth() to compute potential growth rate on roots.
        # total potential growth rate of roots in g per slab.this is computed in
        # potential_root_growth() and used in actual_root_growth().
        sumpdr = self.potential_root_growth(3, self.per_plant_area)
        # Total potential growth rate of roots is converted from g per slab(sumpdr) to
        # g per plant (state.root_potential_growth).
        # Limit state.root_potential_growth to(maxrtgr * per_plant_area) g per plant
        # per day.
        maxrtgr = 0.045  # maximum possible potential root growth, g dm - 2 day - 1.
        self.root_potential_growth = min(
            maxrtgr * self.per_plant_area,
            sumpdr * 100 * self.per_plant_area / self.row_space,
        )
        # Call dry_matter_balance() to compute carbon balance, allocation of carbon to
        # plant parts, and carbon stress.
        vratio = self.dry_matter_balance(self.per_plant_area)
        # If it is after first square, call actual_fruit_growth() to compute actual
        # growth rate of squares and bolls.
        if self.fruiting_nodes_stage[0, 0, 0] != Stage.NotYetFormed:
            self.actual_fruit_growth()
        # Initialize state.leaf_weight.It is assumed that cotyledons fall off at time
        # of first square. Also initialize state.petiole_weight.
        if self.first_square_date is not None:
            self.leaf_weight = 0
        else:
            cotylwt = 0.20  # weight of cotyledons dry matter.
            self.leaf_weight = cotylwt
        self.petiole_weight = 0
        # Call actual_leaf_growth to compute actual growth rate of leaves and compute
        # leaf area index.
        self.actual_leaf_growth(vratio)
        self.leaf_area_index = self.leaf_area / self.per_plant_area
        # Add actual_stem_growth to state.stem_weight.
        self.stem_weight += self.actual_stem_growth
        # Plant density affects growth in height of tall plants.
        htdenf = (
            55  # minimum plant height for plant density affecting growth in height.
        )
        z1 = min(
            max((self.plant_height - htdenf) / htdenf, 0),
            1,
        )  # intermediate variable to compute denf2.
        denf2 = 1 + z1 * (
            self.density_factor - 1
        )  # effect of plant density on plant growth in height.
        # Call add_plant_height to compute PlantHeight.
        if self.version < 0x500 or not self.date >= self.topping_date:
            # node numbers of top node.

            if len(self.vegetative_branches[0].fruiting_branches) >= 2:
                stage = self.fruiting_nodes_stage[0, 1, 0]
            else:
                stage = Stage.NotYetFormed
            self.plant_height += self.add_plant_height(
                denf2,
                self.day_inc,
                self.number_of_pre_fruiting_nodes,
                stage,
                self.pre_fruiting_nodes_age[self.number_of_pre_fruiting_nodes - 1],
                self.pre_fruiting_nodes_age[self.number_of_pre_fruiting_nodes - 2],
                self.agetop,
                self.water_stress_stem,
                self.carbon_stress,
                self.nitrogen_stress_vegetative,
                *self.cultivar_parameters[19:27],
            )
        # Call ActualRootGrowth() to compute actual root growth.
        self.compute_actual_root_growth(
            sumpdr,
            self.row_space,
            self._column_width,
            self.per_plant_area,
            3,
            self.emerge_date,
            self.plant_row_column,
        )
